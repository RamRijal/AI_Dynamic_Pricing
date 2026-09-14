import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.ml.train_baselines import TARGET, select_feature_columns


def train_main_model(
    df: pd.DataFrame,
    *,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    n_estimators: int = 300,
    min_samples_leaf: int = 5,
    subsample: float = 0.85,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    feature_columns = select_feature_columns(df)
    X = df[feature_columns]
    y = df[TARGET]
    model = GradientBoostingRegressor(
        random_state=random_state,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
    )
    model.fit(X, y)
    return model
