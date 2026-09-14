from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


TARGET = "units_sold"
DEFAULT_FEATURES = [
    "current_price",
    "competitor_price",
    "stock_level",
    "page_views",
    "price_gap",
    "prev_price",
    "prev_units_sold",
    "rolling_avg_sales",
    "stock_ratio",
    "view_to_sales_ratio",
]


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in DEFAULT_FEATURES if column in df.columns]


def static_baseline_predictions(df: pd.DataFrame) -> pd.Series:
    return df["prev_units_sold"].where(df["prev_units_sold"] > 0, df["units_sold"].median())


def rule_based_price_adjustment(df: pd.DataFrame) -> pd.Series:
    adjustments = []
    for row in df.itertuples(index=False):
        price = float(row.current_price)
        stock_ratio = float(getattr(row, "stock_ratio", 0))
        view_ratio = float(getattr(row, "view_to_sales_ratio", 0))
        if stock_ratio > 0.7 and view_ratio > 15:
            price *= 0.97
        elif stock_ratio < 0.3 and getattr(row, "units_sold", 0) > 0:
            price *= 1.03
        adjustments.append(price)
    return pd.Series(adjustments, index=df.index, name="rule_based_price")


def train_baselines(
    df: pd.DataFrame,
    *,
    rf_n_estimators: int = 50,
    random_state: int = 42,
) -> dict:
    feature_columns = select_feature_columns(df)
    X = df[feature_columns]
    y = df[TARGET]

    linear = LinearRegression().fit(X, y)
    forest = RandomForestRegressor(random_state=random_state, n_estimators=rf_n_estimators).fit(X, y)

    return {
        "static_baseline": {"predictions": static_baseline_predictions(df)},
        "rule_based_baseline": {"prices": rule_based_price_adjustment(df)},
        "linear_regression": linear,
        "random_forest": forest,
        "feature_columns": feature_columns,
    }
