import pandas as pd


NUMERIC_COLUMNS = [
    "current_price",
    "units_sold",
    "stock_level",
    "competitor_price",
    "page_views",
    "unit_cost",
]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "unit_cost" not in cleaned.columns:
        cleaned["unit_cost"] = cleaned["current_price"] * 0.7

    cleaned = cleaned.dropna(subset=["product_id", "date", "current_price", "units_sold"])
    cleaned["competitor_price"] = cleaned["competitor_price"].fillna(cleaned["current_price"])
    cleaned["page_views"] = cleaned["page_views"].fillna(0)
    cleaned["stock_level"] = cleaned["stock_level"].fillna(0)
    cleaned["unit_cost"] = cleaned["unit_cost"].fillna(cleaned["current_price"] * 0.7)

    non_negative = ["current_price", "units_sold", "stock_level", "competitor_price", "page_views", "unit_cost"]
    for column in non_negative:
        cleaned[column] = cleaned[column].clip(lower=0)

    return cleaned.sort_values(["product_id", "date"]).reset_index(drop=True)
