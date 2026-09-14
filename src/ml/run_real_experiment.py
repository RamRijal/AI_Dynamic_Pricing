from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.app.config import MODELS_DIR, REPORTS_DIR
from src.data.cleaning import clean_dataset
from src.data.data_quality import build_data_quality_report
from src.data.load_dataset import load_dataset
from src.features.build_features import build_features
from src.ml.evaluate_models import (
    MAIN_MODEL_LABEL,
    build_model_selection_summary,
    compare_model_metrics,
    extend_metrics_with_advanced_models,
    export_feature_importance,
    select_best_model,
)
from src.ml.save_artifacts import save_csv_report, save_json_report, save_model_artifact
from src.ml.train_advanced_models import train_advanced_models
from src.ml.train_baselines import rule_based_price_adjustment, select_feature_columns, train_baselines
from src.ml.train_main_model import train_main_model
from src.pricing.guardrails import evaluate_guardrails
from src.pricing.run_pricing_cycle import _estimate_units, run_pricing_cycle

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable


def _time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ordered_dates = sorted(pd.Series(df["date"].unique()).tolist())
    split_index = max(1, int(len(ordered_dates) * (1 - test_ratio)))
    split_date = ordered_dates[min(split_index, len(ordered_dates) - 1)]
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    if train.empty or test.empty:
        midpoint = max(1, len(df) // 2)
        train = df.iloc[:midpoint].copy()
        test = df.iloc[midpoint:].copy()
        split_date = str(test["date"].min())
    return train, test, str(split_date)


def _business_metrics(priced: pd.DataFrame) -> dict:
    current_revenue = float((priced["current_price"] * priced["units_sold"]).sum())
    current_profit = float(((priced["current_price"] - priced["unit_cost"]) * priced["units_sold"]).sum())
    projected_revenue = float((priced["new_price"] * priced["predicted_units"]).sum())
    projected_profit = float(((priced["new_price"] - priced["unit_cost"]) * priced["predicted_units"]).sum())
    return {
        "current_revenue": round(current_revenue, 2),
        "current_profit": round(current_profit, 2),
        "projected_revenue": round(projected_revenue, 2),
        "projected_profit": round(projected_profit, 2),
        "revenue_uplift_pct": round(((projected_revenue - current_revenue) / current_revenue) * 100, 2)
        if current_revenue
        else 0.0,
        "profit_uplift_pct": round(((projected_profit - current_profit) / current_profit) * 100, 2)
        if current_profit
        else 0.0,
    }


def _stability_metrics(priced: pd.DataFrame) -> dict:
    changes = priced["price_change_pct"].astype(float)
    upward = int((changes > 0).sum())
    downward = int((changes < 0).sum())
    unchanged = int((changes == 0).sum())
    absolute_changes = changes.abs()
    return {
        "average_price_change_pct": round(float(changes.mean()), 4),
        "average_absolute_price_change_pct": round(float(absolute_changes.mean()), 4),
        "median_absolute_price_change_pct": round(float(absolute_changes.median()), 4),
        "max_absolute_price_change_pct": round(float(absolute_changes.max()), 4),
        "upward_changes": upward,
        "downward_changes": downward,
        "unchanged_prices": unchanged,
    }


def _guardrail_summary(priced: pd.DataFrame) -> dict:
    reason_counts: dict[str, int] = {}
    triggered_rows = 0
    for value in priced["guardrail_reasons"].astype(str):
        if not value:
            continue
        triggered_rows += 1
        for reason in [item for item in value.split("|") if item]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "rows_evaluated": int(len(priced)),
        "rows_with_guardrail_trigger": triggered_rows,
        "guardrail_trigger_rate_pct": round((triggered_rows / len(priced)) * 100, 4) if len(priced) else 0.0,
        "reason_counts": reason_counts,
        "selection_reason_counts": {
            key: int(value)
            for key, value in priced["selection_reason"].value_counts().to_dict().items()
        },
        "stability_metrics": _stability_metrics(priced),
    }


def _policy_summary(priced: pd.DataFrame) -> dict:
    business = _business_metrics(priced)
    stability = _stability_metrics(priced)
    guardrails = _guardrail_summary(priced)
    return {
        **business,
        "average_absolute_price_change_pct": stability["average_absolute_price_change_pct"],
        "guardrail_trigger_rate_pct": guardrails["guardrail_trigger_rate_pct"],
    }


def _build_model_batch_estimator(model, feature_columns: list[str]):
    def estimate_batch(candidate_frame: pd.DataFrame) -> pd.Series:
        features = candidate_frame.reindex(columns=feature_columns, fill_value=0)
        predictions = model.predict(features)
        return pd.Series(predictions, index=candidate_frame.index)

    return estimate_batch


def _estimate_runtime_seconds(row_count: int, train_rows: int, test_rows: int, model_count: int, epochs: int) -> float:
    per_epoch_seconds = (
        8.0
        + (train_rows * 0.0012)
        + (test_rows * model_count * 0.0009)
        + (row_count * 0.0002)
    )
    return round(max(10.0, per_epoch_seconds * epochs), 2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dissertation-grade dynamic pricing experiment.")
    parser.add_argument("--dataset-path", default="data/raw/ecommerce_pricing.csv")
    parser.add_argument("--model-dir", default=str(MODELS_DIR))
    parser.add_argument("--report-dir", default=str(REPORTS_DIR))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--rf-n-estimators", type=int, default=50)
    parser.add_argument("--gb-learning-rate", type=float, default=0.05)
    parser.add_argument("--gb-max-depth", type=int, default=4)
    parser.add_argument("--gb-n-estimators", type=int, default=300)
    parser.add_argument("--gb-min-samples-leaf", type=int, default=5)
    parser.add_argument("--gb-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.85)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--cat-iterations", type=int, default=300)
    parser.add_argument("--cat-depth", type=int, default=6)
    parser.add_argument("--cat-learning-rate", type=float, default=0.05)
    return parser.parse_args()


def _epoch_seed(base_seed: int, epoch_index: int) -> int:
    return base_seed + epoch_index



def _run_rule_based_policy(df: pd.DataFrame, min_price: float, max_price: float, max_change_pct: float, min_margin_pct: float) -> pd.DataFrame:
    suggested_prices = rule_based_price_adjustment(df)
    run_timestamp = pd.Timestamp.now("UTC").isoformat()
    rows = []
    for row, suggested_price in zip(df.itertuples(index=False), suggested_prices, strict=False):
        unit_cost = float(getattr(row, "unit_cost", row.current_price * 0.7))
        guardrail_result = evaluate_guardrails(
            current_price=float(row.current_price),
            suggested_price=float(suggested_price),
            min_price=min_price,
            max_price=max_price,
            max_change_pct=max_change_pct,
            unit_cost=unit_cost,
            min_margin_pct=min_margin_pct,
        )
        final_price = float(guardrail_result["final_price"])
        predicted_units = float(_estimate_units(pd.Series(row._asdict()), final_price))
        rows.append(
            {
                **row._asdict(),
                "suggested_price": float(suggested_price),
                "new_price": final_price,
                "predicted_units": predicted_units,
                "predicted_profit": round((final_price - unit_cost) * predicted_units, 4),
                "predicted_revenue": round(final_price * predicted_units, 4),
                "price_change_pct": round(((final_price - float(row.current_price)) / float(row.current_price)) * 100, 4)
                if float(row.current_price)
                else 0.0,
                "candidate_count": 1,
                "feasible_candidate_count": 1 if not guardrail_result["reasons"] else 0,
                "selection_reason": "rule_based_policy",
                "guardrail_feasible": not guardrail_result["reasons"],
                "guardrail_reasons": "|".join(guardrail_result["reasons"]),
                "run_timestamp": run_timestamp,
            }
        )
    return pd.DataFrame(rows)


def run_real_experiment(
    dataset_path: str | Path = "data/raw/ecommerce_pricing.csv",
    model_dir: str | Path = MODELS_DIR,
    report_dir: str | Path = REPORTS_DIR,
    epochs: int = 2,
    base_seed: int = 42,
    rf_n_estimators: int = 50,
    gb_learning_rate: float = 0.05,
    gb_max_depth: int = 4,
    gb_n_estimators: int = 300,
    gb_min_samples_leaf: int = 5,
    gb_subsample: float = 0.85,
    xgb_n_estimators: int = 300,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.05,
    xgb_subsample: float = 0.85,
    xgb_colsample_bytree: float = 0.85,
    xgb_reg_lambda: float = 1.0,
    cat_iterations: int = 300,
    cat_depth: int = 6,
    cat_learning_rate: float = 0.05,
) -> dict:
    start_time = time.perf_counter()
    raw = load_dataset(dataset_path)
    cleaned = clean_dataset(raw)
    featured = build_features(cleaned)
    train_df, test_df, split_date = _time_split(featured)
    runtime_estimate_seconds = _estimate_runtime_seconds(
        row_count=len(featured),
        train_rows=len(train_df),
        test_rows=len(test_df),
        model_count=6,
        epochs=epochs,
    )

    epoch_summaries: list[dict] = []
    latest_epoch_outputs: dict | None = None

    for epoch_index in tqdm(range(epochs), total=epochs, desc="experiment-epochs"):
        epoch_number = epoch_index + 1
        epoch_seed = _epoch_seed(base_seed, epoch_index)
        epoch_started = time.perf_counter()

        baselines = train_baselines(train_df, rf_n_estimators=rf_n_estimators, random_state=epoch_seed)
        main_model = train_main_model(
            train_df,
            learning_rate=gb_learning_rate,
            max_depth=gb_max_depth,
            n_estimators=gb_n_estimators,
            min_samples_leaf=gb_min_samples_leaf,
            subsample=gb_subsample,
            random_state=epoch_seed,
        )
        advanced_models = train_advanced_models(
            train_df,
            xgb_n_estimators=xgb_n_estimators,
            xgb_max_depth=xgb_max_depth,
            xgb_learning_rate=xgb_learning_rate,
            xgb_subsample=xgb_subsample,
            xgb_colsample_bytree=xgb_colsample_bytree,
            xgb_reg_lambda=xgb_reg_lambda,
            cat_iterations=cat_iterations,
            cat_depth=cat_depth,
            cat_learning_rate=cat_learning_rate,
            random_state=epoch_seed,
        )
        feature_columns = select_feature_columns(train_df)

        test_baselines = {
            **baselines,
            "feature_columns": feature_columns,
        }
        metrics = compare_model_metrics(test_df, test_baselines, main_model)
        metrics = extend_metrics_with_advanced_models(metrics, test_df, advanced_models)

        model_estimators = {
            "linear_regression": _build_model_batch_estimator(baselines["linear_regression"], feature_columns),
            "random_forest": _build_model_batch_estimator(baselines["random_forest"], feature_columns),
            MAIN_MODEL_LABEL: _build_model_batch_estimator(main_model, feature_columns),
            "xgboost": _build_model_batch_estimator(advanced_models["xgboost"], feature_columns),
            "catboost": _build_model_batch_estimator(advanced_models["catboost"], feature_columns),
        }

        policy_runs = {
            "rule_based_baseline": _run_rule_based_policy(
                test_df,
                min_price=1.0,
                max_price=10_000.0,
                max_change_pct=0.10,
                min_margin_pct=0.05,
            )
        }
        for model_name, estimator in tqdm(model_estimators.items(), total=len(model_estimators), desc=f"epoch-{epoch_number}-policy"):
            policy_runs[model_name] = run_pricing_cycle(
                test_df,
                persist_history=False,
                unit_estimator_batch=estimator,
                show_progress=False,
                progress_desc=f"{model_name}-pricing",
            )

        pricing_policy_metrics = {name: _policy_summary(frame) for name, frame in policy_runs.items()}

        priced = run_pricing_cycle(test_df, persist_history=False, show_progress=False)
        business_metrics = _business_metrics(priced)
        stability_metrics = _stability_metrics(priced)
        guardrail_summary = _guardrail_summary(priced)
        data_quality_report = build_data_quality_report(raw, cleaned, featured, dataset_path)
        selected_model, selected_metrics = select_best_model(metrics, pricing_policy_metrics)
        selected_model_lookup = {
            "linear_regression": baselines["linear_regression"],
            "random_forest": baselines["random_forest"],
            MAIN_MODEL_LABEL: main_model,
            "xgboost": advanced_models["xgboost"],
            "catboost": advanced_models["catboost"],
        }
        feature_importance = export_feature_importance(selected_model_lookup.get(selected_model, main_model), feature_columns)
        epoch_duration_seconds = round(time.perf_counter() - epoch_started, 2)
        epoch_summary = {
            "epoch": epoch_number,
            "seed": epoch_seed,
            "duration_seconds": epoch_duration_seconds,
            "selected_model": selected_model,
            "selected_model_metrics": selected_metrics,
            "projected_profit": pricing_policy_metrics[selected_model]["projected_profit"],
            "guardrail_trigger_rate_pct": pricing_policy_metrics[selected_model]["guardrail_trigger_rate_pct"],
        }
        epoch_summaries.append(epoch_summary)
        latest_epoch_outputs = {
            "baselines": baselines,
            "main_model": main_model,
            "advanced_models": advanced_models,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "policy_runs": policy_runs,
            "pricing_policy_metrics": pricing_policy_metrics,
            "priced": priced,
            "business_metrics": business_metrics,
            "stability_metrics": stability_metrics,
            "guardrail_summary": guardrail_summary,
            "data_quality_report": data_quality_report,
            "selected_model": selected_model,
            "selected_model_metrics": selected_metrics,
            "feature_importance": feature_importance,
        }

    if latest_epoch_outputs is None:
        raise RuntimeError("No experiment epochs were executed.")

    baselines = latest_epoch_outputs["baselines"]
    main_model = latest_epoch_outputs["main_model"]
    advanced_models = latest_epoch_outputs["advanced_models"]
    feature_columns = latest_epoch_outputs["feature_columns"]
    metrics = latest_epoch_outputs["metrics"]
    pricing_policy_metrics = latest_epoch_outputs["pricing_policy_metrics"]
    priced = latest_epoch_outputs["priced"]
    business_metrics = latest_epoch_outputs["business_metrics"]
    stability_metrics = latest_epoch_outputs["stability_metrics"]
    guardrail_summary = latest_epoch_outputs["guardrail_summary"]
    data_quality_report = latest_epoch_outputs["data_quality_report"]
    selected_model = latest_epoch_outputs["selected_model"]
    selected_metrics = latest_epoch_outputs["selected_model_metrics"]
    feature_importance = latest_epoch_outputs["feature_importance"]
    total_runtime_seconds = round(time.perf_counter() - start_time, 2)

    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    save_model_artifact(model_dir / "linear_regression.pkl", baselines["linear_regression"])
    save_model_artifact(model_dir / "random_forest.pkl", baselines["random_forest"])
    save_model_artifact(model_dir / f"{MAIN_MODEL_LABEL}.pkl", main_model)
    save_model_artifact(model_dir / "xgboost.pkl", advanced_models["xgboost"])
    save_model_artifact(model_dir / "catboost.pkl", advanced_models["catboost"])

    save_json_report(report_dir / "model_metrics.json", metrics)
    save_json_report(report_dir / "data_quality_report.json", data_quality_report)
    save_json_report(report_dir / "guardrail_summary.json", guardrail_summary)
    save_json_report(report_dir / "pricing_policy_comparison.json", pricing_policy_metrics)
    save_json_report(
        report_dir / "final_metrics.json",
        {
            "dataset_path": str(dataset_path),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "split_date": split_date,
            "epochs": epochs,
            "epoch_summaries": epoch_summaries,
            "runtime_estimate_seconds": runtime_estimate_seconds,
            "actual_runtime_seconds": total_runtime_seconds,
            "technical_metrics": metrics,
            "business_metrics": business_metrics,
            "stability_metrics": stability_metrics,
            "guardrail_summary": guardrail_summary,
            "pricing_policy_metrics": pricing_policy_metrics,
            "selected_model": selected_model,
            "selected_model_metrics": selected_metrics,
        },
    )
    save_json_report(
        report_dir / "experiment_summary.json",
        {
            "dataset_path": str(dataset_path),
            "rows": len(featured),
            "unique_products": int(featured["product_id"].nunique()),
            "categories": int(featured["category"].nunique()),
            "date_min": str(featured["date"].min().date()),
            "date_max": str(featured["date"].max().date()),
            "split_date": split_date,
            "feature_columns": feature_columns,
            "epochs": epochs,
            "runtime_estimate_seconds": runtime_estimate_seconds,
            "actual_runtime_seconds": total_runtime_seconds,
        },
    )
    save_json_report(
        report_dir / "experiment_configuration.json",
        {
            "dataset_path": str(dataset_path),
            "split_strategy": "time-based 80/20 by ordered dates",
            "target": "units_sold",
            "epochs": epochs,
            "base_seed": base_seed,
            "runtime_estimate_seconds": runtime_estimate_seconds,
            "feature_columns": feature_columns,
            "models": ["static_baseline", "linear_regression", "random_forest", MAIN_MODEL_LABEL, "xgboost", "catboost"],
            "selected_model": selected_model,
            "training_parameters": {
                "rf_n_estimators": rf_n_estimators,
                "gb_learning_rate": gb_learning_rate,
                "gb_max_depth": gb_max_depth,
                "gb_n_estimators": gb_n_estimators,
                "gb_min_samples_leaf": gb_min_samples_leaf,
                "gb_subsample": gb_subsample,
                "xgb_n_estimators": xgb_n_estimators,
                "xgb_max_depth": xgb_max_depth,
                "xgb_learning_rate": xgb_learning_rate,
                "xgb_subsample": xgb_subsample,
                "xgb_colsample_bytree": xgb_colsample_bytree,
                "xgb_reg_lambda": xgb_reg_lambda,
                "cat_iterations": cat_iterations,
                "cat_depth": cat_depth,
                "cat_learning_rate": cat_learning_rate,
            },
            "pricing_objective": "maximize predicted profit under guardrails",
            "guardrails": {
                "min_price": 1.0,
                "max_price": 10000.0,
                "max_change_pct": 0.10,
                "min_margin_pct": 0.05,
            },
        },
    )
    (report_dir / "model_selection_summary.md").write_text(
        build_model_selection_summary(metrics, selected_model, pricing_policy_metrics),
        encoding="utf-8",
    )
    save_csv_report(report_dir / "feature_importance.csv", feature_importance)
    save_csv_report(
        report_dir / "price_changes.csv",
        priced[
            [
                "product_id",
                "date",
                "current_price",
                "suggested_price",
                "new_price",
                "predicted_units",
                "predicted_profit",
                "predicted_revenue",
                "price_change_pct",
                "candidate_count",
                "feasible_candidate_count",
                "selection_reason",
                "guardrail_feasible",
                "guardrail_reasons",
                "run_timestamp",
            ]
        ].to_dict(orient="records"),
    )
    from src.reporting.generate_figures import generate_figures

    generate_figures(report_dir)
    comparison_lines = [
        "# Baseline Comparison Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Train rows: {len(train_df):,}",
        f"- Test rows: {len(test_df):,}",
        f"- Split date: `{split_date}`",
        f"- Epochs: {epochs}",
        f"- Runtime estimate (seconds): {runtime_estimate_seconds}",
        f"- Actual runtime (seconds): {total_runtime_seconds}",
        "",
        "## Technical Metrics",
        "",
    ]
    for name, values in metrics.items():
        comparison_lines.append(f"- {name}: MAE={values['mae']:.4f}, RMSE={values['rmse']:.4f}")
    comparison_lines.extend(
        [
        "",
        "## Business Metrics",
        "",
        f"- Selected model: `{selected_model}`",
        f"- Current revenue: {business_metrics['current_revenue']}",
        f"- Projected revenue: {business_metrics['projected_revenue']}",
        f"- Revenue uplift %: {business_metrics['revenue_uplift_pct']}",
        f"- Current profit: {business_metrics['current_profit']}",
        f"- Projected profit: {business_metrics['projected_profit']}",
        f"- Profit uplift %: {business_metrics['profit_uplift_pct']}",
        "",
        "## Pricing Stability",
        "",
        f"- Average price change %: {stability_metrics['average_price_change_pct']}",
        f"- Average absolute price change %: {stability_metrics['average_absolute_price_change_pct']}",
        f"- Max absolute price change %: {stability_metrics['max_absolute_price_change_pct']}",
        f"- Guardrail trigger rate %: {guardrail_summary['guardrail_trigger_rate_pct']}",
        "",
        "## Downstream Pricing Policy Comparison",
        "",
        ]
    )
    for name, values in sorted(
        pricing_policy_metrics.items(),
        key=lambda item: item[1]["projected_profit"],
        reverse=True,
    ):
        comparison_lines.append(
            f"- {name}: projected_profit={values['projected_profit']}, projected_revenue={values['projected_revenue']}, guardrail_trigger_rate_pct={values['guardrail_trigger_rate_pct']}, average_absolute_price_change_pct={values['average_absolute_price_change_pct']}"
        )
    (report_dir / "baseline_comparison_report.md").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")

    return {
        "dataset_path": str(dataset_path),
        "rows": len(featured),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "split_date": split_date,
        "model_dir": str(model_dir),
        "report_dir": str(report_dir),
        "data_quality_report": str(report_dir / "data_quality_report.json"),
        "selected_model": selected_model,
        "epochs": epochs,
        "runtime_estimate_seconds": runtime_estimate_seconds,
        "actual_runtime_seconds": total_runtime_seconds,
    }


if __name__ == "__main__":
    args = _parse_args()
    result = run_real_experiment(
        dataset_path=args.dataset_path,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        epochs=args.epochs,
        base_seed=args.base_seed,
        rf_n_estimators=args.rf_n_estimators,
        gb_learning_rate=args.gb_learning_rate,
        gb_max_depth=args.gb_max_depth,
        gb_n_estimators=args.gb_n_estimators,
        gb_min_samples_leaf=args.gb_min_samples_leaf,
        gb_subsample=args.gb_subsample,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_reg_lambda=args.xgb_reg_lambda,
        cat_iterations=args.cat_iterations,
        cat_depth=args.cat_depth,
        cat_learning_rate=args.cat_learning_rate,
    )
    for key, value in result.items():
        print(f"{key}={value}")
