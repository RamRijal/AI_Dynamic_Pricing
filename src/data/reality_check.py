"""Auditable diagnostics for the data foundation and ingestion run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


OBSERVED_COLUMNS = [
    "product_id",
    "date",
    "current_price",
    "units_sold",
]
ENRICHED_COLUMNS = [
    "category",
    "stock_level",
    "competitor_price",
    "page_views",
    "unit_cost",
    "promotion_flag",
    "base_price",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def price_variation_diagnostics(experiment: pd.DataFrame) -> dict:
    """Measure whether product-level observed prices actually change over time."""
    ordered = experiment.sort_values(["product_id", "date"])
    previous_price = ordered.groupby("product_id")["current_price"].shift()
    changes = ordered["current_price"].ne(previous_price) & previous_price.notna()
    comparable = ordered.groupby("product_id").size().gt(1)
    products_with_changes = (
        ordered.assign(price_changed=changes)
        .groupby("product_id")["price_changed"]
        .any()
    )
    comparable_transitions = int(previous_price.notna().sum())
    price_change_amount = (
        ordered["current_price"]
        .sub(ordered.groupby("product_id")["current_price"].shift())
        .dropna()
    )
    return {
        "price_observations": int(len(ordered)),
        "comparable_product_observations": int(comparable.sum()),
        "price_change_observations": int(changes.sum()),
        "price_change_rate": round(
            float(changes.sum() / comparable_transitions) if comparable_transitions else 0.0,
            6,
        ),
        "products_with_multiple_dates": int(comparable.sum()),
        "products_with_price_changes": int(products_with_changes.sum()),
        "product_price_variation_rate": round(
            float(products_with_changes[comparable].mean()), 6
        ),
        "price_change_amount_p05": round(float(price_change_amount.quantile(0.05)), 4),
        "price_change_amount_median": round(float(price_change_amount.quantile(0.50)), 4),
        "price_change_amount_p95": round(float(price_change_amount.quantile(0.95)), 4),
    }


def build_run_manifest(
    source_path: str | Path,
    experiment: pd.DataFrame,
    output_paths: dict[str, str | Path],
) -> dict:
    """Build a reproducibility manifest for one successful ingestion run."""
    source = Path(source_path)
    manifest = {
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
        "outputs": {
            name: {
                "path": str(Path(path)),
                "sha256": sha256_file(path),
                "bytes": Path(path).stat().st_size,
            }
            for name, path in output_paths.items()
        },
        "dataset": {
            "rows": int(len(experiment)),
            "columns": list(experiment.columns),
            "unique_products": int(experiment["product_id"].nunique()),
            "date_min": str(experiment["date"].min().date()),
            "date_max": str(experiment["date"].max().date()),
            "observed_columns": OBSERVED_COLUMNS,
            "enriched_columns": ENRICHED_COLUMNS,
            "price_variation": price_variation_diagnostics(experiment),
        },
    }
    return manifest


def write_run_manifest(manifest: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
