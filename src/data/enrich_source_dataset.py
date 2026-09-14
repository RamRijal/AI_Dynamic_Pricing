from __future__ import annotations

from pathlib import Path
import sys
from zlib import crc32

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.app.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.cleaning import clean_dataset
from src.data.load_dataset import load_source_dataset
from src.features.build_features import build_features
from src.data.reality_check import build_run_manifest, write_run_manifest


SOURCE_PATH = RAW_DATA_DIR / "uci_online_retail.xlsx"
ENRICHED_PATH = RAW_DATA_DIR / "ecommerce_pricing.csv"
CLEANED_PATH = PROCESSED_DATA_DIR / "cleaned_pricing_data.csv"
FEATURED_PATH = PROCESSED_DATA_DIR / "featured_pricing_data.csv"
SUMMARY_PATH = PROCESSED_DATA_DIR / "dataset_summary.md"
RUN_MANIFEST_PATH = PROCESSED_DATA_DIR / "pipeline_run_manifest.json"


def _category_from_description(description: str) -> str:
    text = str(description).upper()
    if any(token in text for token in ("LAMP", "LIGHT", "LANTERN", "CANDLE")):
        return "Home Decor"
    if any(token in text for token in ("BAG", "PAPER", "BOX", "WRAP", "CARD")):
        return "Gift Packaging"
    if any(token in text for token in ("MUG", "BOWL", "PLATE", "JAR", "CUP")):
        return "Kitchen"
    if any(token in text for token in ("NECKLACE", "BRACELET", "RING", "CHARM")):
        return "Accessories"
    if any(token in text for token in ("CHRISTMAS", "EASTER", "HALLOWEEN", "PARTY")):
        return "Seasonal"
    return "General Merchandise"


def _noise_ratio(key: str, floor: float, ceiling: float) -> float:
    span = ceiling - floor
    value = crc32(key.encode("utf-8")) % 10_000
    return floor + (value / 10_000) * span


def build_experiment_dataset(source: pd.DataFrame) -> pd.DataFrame:
    filtered = source.copy()
    filtered["InvoiceDate"] = pd.to_datetime(filtered["InvoiceDate"], errors="coerce")
    filtered = filtered.dropna(subset=["StockCode", "Description", "InvoiceDate", "Quantity", "UnitPrice"])
    filtered = filtered[~filtered["InvoiceNo"].astype(str).str.startswith("C", na=False)]
    filtered = filtered[(filtered["Quantity"] > 0) & (filtered["UnitPrice"] > 0)]
    filtered["date"] = filtered["InvoiceDate"].dt.floor("D")
    filtered["product_id"] = filtered["StockCode"].astype(str).str.strip()
    filtered["category"] = filtered["Description"].map(_category_from_description)

    grouped = (
        filtered.groupby(["product_id", "Description", "category", "date"], as_index=False)
        .agg(units_sold=("Quantity", "sum"), current_price=("UnitPrice", "mean"))
        .sort_values(["product_id", "date"])
    )

    grouped["base_price"] = grouped.groupby("product_id")["current_price"].transform("max")
    grouped["promotion_flag"] = (
        grouped["current_price"] < grouped["base_price"] * 0.98
    ).astype(int)

    grouped["competitor_price"] = grouped.apply(
        lambda row: round(
            row["current_price"]
            * _noise_ratio(f"{row['product_id']}-competitor", 0.94, 1.04),
            2,
        ),
        axis=1,
    )

    grouped["unit_cost"] = grouped.apply(
        lambda row: round(
            row["current_price"]
            * _noise_ratio(f"{row['category']}-cost", 0.52, 0.68),
            2,
        ),
        axis=1,
    )

    grouped["page_views"] = grouped.apply(
        lambda row: int(
            max(
                row["units_sold"] * _noise_ratio(f"{row['product_id']}-views", 18, 36),
                row["units_sold"] + 5,
            )
        ),
        axis=1,
    )

    grouped["restock_buffer"] = grouped.apply(
        lambda row: int(row["units_sold"] * _noise_ratio(f"{row['product_id']}-buffer", 2.5, 5.5)),
        axis=1,
    )
    cumulative_sales = grouped.groupby("product_id")["units_sold"].cumsum()
    grouped["opening_stock"] = grouped.groupby("product_id")["units_sold"].transform("sum") + grouped["restock_buffer"]
    grouped["stock_level"] = (grouped["opening_stock"] - cumulative_sales).clip(lower=0)

    experiment = grouped[
        [
            "product_id",
            "category",
            "date",
            "current_price",
            "units_sold",
            "stock_level",
            "competitor_price",
            "page_views",
            "unit_cost",
            "promotion_flag",
            "base_price",
        ]
    ].copy()
    experiment["current_price"] = experiment["current_price"].round(2)
    return experiment.reset_index(drop=True)


def write_dataset_summary(source: pd.DataFrame, experiment: pd.DataFrame) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = "\n".join(
        [
            "# Dataset Summary",
            "",
            "## Provenance",
            "",
            "- Source dataset: `data/raw/uci_online_retail.xlsx`",
            "- Source: UCI Machine Learning Repository Online Retail",
            "- Retrieved: August 8, 2026",
            "- License: CC BY 4.0",
            "",
            "## Enrichment Assumptions",
            "",
            "- Product category inferred from description keywords",
            "- Competitor price simulated as a deterministic band around current price",
            "- Unit cost simulated from category-level cost ratios",
            "- Page views simulated from units sold with deterministic product-level multipliers",
            "- Stock level simulated from total product sales plus a deterministic restock buffer",
            "",
            "## Counts",
            "",
            f"- Source rows after source loading: {len(source):,}",
            f"- Enriched experiment rows: {len(experiment):,}",
            f"- Unique products: {experiment['product_id'].nunique():,}",
            f"- Categories: {experiment['category'].nunique():,}",
            f"- Date range: {experiment['date'].min().date()} to {experiment['date'].max().date()}",
        ]
    )
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")


def materialize_experiment_dataset(source_path: str | Path = SOURCE_PATH) -> dict:
    source = load_source_dataset(source_path)
    experiment = build_experiment_dataset(source)
    cleaned = clean_dataset(experiment)
    featured = build_features(cleaned)

    ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURED_PATH.parent.mkdir(parents=True, exist_ok=True)

    experiment.to_csv(ENRICHED_PATH, index=False)
    cleaned.to_csv(CLEANED_PATH, index=False)
    featured.to_csv(FEATURED_PATH, index=False)
    write_dataset_summary(source, experiment)
    manifest = build_run_manifest(
        source_path,
        experiment,
        {
            "enriched": ENRICHED_PATH,
            "cleaned": CLEANED_PATH,
            "featured": FEATURED_PATH,
            "summary": SUMMARY_PATH,
        },
    )
    write_run_manifest(manifest, RUN_MANIFEST_PATH)

    return {
        "source_rows": len(source),
        "experiment_rows": len(experiment),
        "unique_products": int(experiment["product_id"].nunique()),
        "categories": int(experiment["category"].nunique()),
        "enriched_path": str(ENRICHED_PATH),
        "cleaned_path": str(CLEANED_PATH),
        "featured_path": str(FEATURED_PATH),
        "summary_path": str(SUMMARY_PATH),
        "manifest_path": str(RUN_MANIFEST_PATH),
    }


if __name__ == "__main__":
    result = materialize_experiment_dataset()
    for key, value in result.items():
        print(f"{key}={value}")
