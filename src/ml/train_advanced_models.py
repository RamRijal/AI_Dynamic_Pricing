from __future__ import annotations

import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.ml.train_baselines import TARGET, select_feature_columns


def train_advanced_models(
    df: pd.DataFrame,
    *,
    xgb_n_estimators: int = 300,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.05,
    xgb_subsample: float = 0.85,
    xgb_colsample_bytree: float = 0.85,
    xgb_reg_lambda: float = 1.0,
    cat_iterations: int = 300,
    cat_depth: int = 6,
    cat_learning_rate: float = 0.05,
    random_state: int = 42,
) -> dict:
    feature_columns = select_feature_columns(df)
    X = df[feature_columns]
    y = df[TARGET]

    xgboost = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
        reg_lambda=xgb_reg_lambda,
        n_jobs=1,
    )
    xgboost.fit(X, y)

    catboost = CatBoostRegressor(
        random_seed=random_state,
        iterations=cat_iterations,
        depth=cat_depth,
        learning_rate=cat_learning_rate,
        loss_function="RMSE",
        verbose=False,
        thread_count=1,
    )
    catboost.fit(X, y)

    return {
        "xgboost": xgboost,
        "catboost": catboost,
        "feature_columns": feature_columns,
    }
