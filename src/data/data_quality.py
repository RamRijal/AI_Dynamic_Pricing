from __future__ import annotations

from pathlib import Path

import pandas as pd


def _required_column_check(df: pd.DataFrame, required_columns: list[str]) -> dict[str, bool]:
    return {column: column in df.columns for column in required_columns}


def _missing_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for column in columns:
        if column in df.columns:
            summary[column] = int(df[column].isna().sum())
    return summary


def _negative_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for column in columns:
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            summary[column] = int((numeric < 0).sum())
    return summary


def _iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        q1 = float(numeric.quantile(0.25))
        q3 = float(numeric.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((numeric < lower) | (numeric > upper)).sum())
        summary[column] = {
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": outlier_count,
        }
    return summary


def build_data_quality_report(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    featured: pd.DataFrame,
    dataset_path: str | Path,
) -> dict:
    dataset_path = Path(dataset_path)
    required_columns = [
        "product_id",
        "category",
        "date",
        "current_price",
        "units_sold",
        "stock_level",
        "competitor_price",
        "page_views",
    ]
    monitored_numeric_columns = [
        "current_price",
        "units_sold",
        "stock_level",
        "competitor_price",
        "page_views",
        "unit_cost",
    ]

    cleaned_date_min = str(cleaned["date"].min().date()) if not cleaned.empty else None
    cleaned_date_max = str(cleaned["date"].max().date()) if not cleaned.empty else None

    return {
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.name,
        "row_counts": {
            "raw_input_rows": int(len(raw)),
            "cleaned_rows": int(len(cleaned)),
            "featured_rows": int(len(featured)),
            "rows_removed_during_cleaning": int(len(raw) - len(cleaned)),
            "rows_removed_before_featuring": int(len(cleaned) - len(featured)),
        },
        "coverage": {
            "unique_products": int(cleaned["product_id"].nunique()) if "product_id" in cleaned.columns else 0,
            "unique_categories": int(cleaned["category"].nunique()) if "category" in cleaned.columns else 0,
            "date_min": cleaned_date_min,
            "date_max": cleaned_date_max,
        },
        "schema_checks": {
            "required_columns_present_in_raw": _required_column_check(raw, required_columns),
            "required_columns_present_in_cleaned": _required_column_check(cleaned, required_columns),
            "required_columns_present_in_featured": _required_column_check(featured, required_columns),
        },
        "missing_values": {
            "raw": _missing_summary(raw, required_columns),
            "cleaned": _missing_summary(cleaned, required_columns),
            "featured": _missing_summary(featured, required_columns),
        },
        "negative_value_checks": {
            "cleaned": _negative_summary(cleaned, monitored_numeric_columns),
            "featured": _negative_summary(featured, monitored_numeric_columns),
        },
        "outlier_summary": _iqr_outlier_summary(cleaned, ["current_price", "units_sold"]),
        "provenance": {
            "quality_check_generated_from": "run_real_experiment",
            "enrichment_type": "semi-synthetic enrichment over real retail transactions",
            "deterministic_enrichment": True,
        },
    }
