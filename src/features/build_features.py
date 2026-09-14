import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.sort_values(["product_id", "date"]).copy()
    featured["price_gap"] = featured["current_price"] - featured["competitor_price"]
    featured["prev_price"] = featured.groupby("product_id")["current_price"].shift(1)
    featured["prev_units_sold"] = featured.groupby("product_id")["units_sold"].shift(1)
    featured["rolling_avg_sales"] = (
        featured.groupby("product_id")["units_sold"]
        .transform(lambda series: series.shift(1).rolling(3, min_periods=1).mean())
    )
    max_stock = max(featured["stock_level"].max(), 1)
    featured["stock_ratio"] = featured["stock_level"] / max_stock
    denominator = featured["units_sold"].replace(0, 1)
    featured["view_to_sales_ratio"] = featured["page_views"] / denominator
    featured["day_of_week"] = featured["date"].dt.dayofweek
    featured["month"] = featured["date"].dt.month

    if "promotion_flag" in featured.columns:
        featured["promotion_flag"] = featured["promotion_flag"].fillna(0)
    else:
        featured["promotion_flag"] = 0

    if "base_price" in featured.columns:
        base = featured["base_price"].where(featured["base_price"] != 0, featured["current_price"])
        featured["discount_pct"] = 1 - (featured["current_price"] / base)
    else:
        featured["discount_pct"] = 0.0

    return featured.fillna(0)
