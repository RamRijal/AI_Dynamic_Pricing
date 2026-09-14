from __future__ import annotations

from math import sqrt

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.ml.train_baselines import TARGET, select_feature_columns, static_baseline_predictions

MAIN_MODEL_LABEL = "gradient_boosting_tuned"


def score_predictions(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
    }


def evaluate_regression_model(model, df: pd.DataFrame) -> dict:
    feature_columns = select_feature_columns(df)
    predictions = model.predict(df[feature_columns])
    return score_predictions(df[TARGET], predictions)


def compare_model_metrics(df: pd.DataFrame, baselines: dict, main_model) -> dict:
    metrics = {
        "static_baseline": score_predictions(df[TARGET], static_baseline_predictions(df)),
        "linear_regression": evaluate_regression_model(baselines["linear_regression"], df),
        "random_forest": evaluate_regression_model(baselines["random_forest"], df),
        MAIN_MODEL_LABEL: evaluate_regression_model(main_model, df),
    }
    return metrics


def extend_metrics_with_advanced_models(metrics: dict[str, dict], df: pd.DataFrame, advanced_models: dict) -> dict[str, dict]:
    extended = dict(metrics)
    for model_name in ("xgboost", "catboost"):
        extended[model_name] = evaluate_regression_model(advanced_models[model_name], df)
    return extended


def select_best_model(metrics: dict[str, dict], pricing_policy_metrics: dict[str, dict] | None = None) -> tuple[str, dict]:
    def rank_key(item):
        name, values = item
        projected_profit = 0.0
        if pricing_policy_metrics and name in pricing_policy_metrics:
            projected_profit = pricing_policy_metrics[name].get("projected_profit", 0.0)
        return (values["rmse"], values["mae"], -projected_profit)

    ranked = sorted(metrics.items(), key=rank_key)
    return ranked[0]


def build_model_selection_summary(
    metrics: dict[str, dict],
    selected_model: str,
    pricing_policy_metrics: dict[str, dict] | None = None,
) -> str:
    lines = [
        "# Model Selection Summary",
        "",
        "## Candidate Ladder",
        "",
        "- `linear_regression`: interpretability baseline",
        "- `random_forest`: non-linear ensemble benchmark",
        f"- `{MAIN_MODEL_LABEL}`: stronger boosting-based final candidate",
        "- `xgboost`: dissertation-grade scalable boosting benchmark",
        "- `catboost`: ordered boosting benchmark with strong tabular performance",
        "",
        "## Technical Comparison",
        "",
    ]
    for name, values in sorted(metrics.items(), key=lambda item: (item[1]["rmse"], item[1]["mae"])):
        lines.append(f"- {name}: RMSE={values['rmse']:.4f}, MAE={values['mae']:.4f}")
    if pricing_policy_metrics:
        lines.extend(
            [
                "",
                "## Pricing Policy Comparison",
                "",
            ]
        )
        for name, values in sorted(
            pricing_policy_metrics.items(),
            key=lambda item: item[1].get("projected_profit", 0.0),
            reverse=True,
        ):
            lines.append(
                f"- {name}: projected_profit={values['projected_profit']:.2f}, projected_revenue={values['projected_revenue']:.2f}, guardrail_trigger_rate_pct={values['guardrail_trigger_rate_pct']:.4f}"
            )
    lines.extend(
        [
            "",
            "## Selection Decision",
            "",
            f"- Selected final model: `{selected_model}`",
            "- Decision rule: prefer the lowest RMSE, then use MAE as the tie-breaker, then use downstream projected profit under shared guardrails as supporting evidence.",
            "- Dissertation framing: linear regression remains the interpretable baseline, while the final model should be strong enough to justify non-linear pricing relationships in tabular demand data.",
        ]
    )
    return "\n".join(lines) + "\n"


def export_feature_importance(model, feature_columns: list[str]) -> list[dict]:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = [0.0 for _ in feature_columns]
    return [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_columns, values, strict=False)
    ]
