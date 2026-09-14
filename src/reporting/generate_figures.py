from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MPL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "artifacts" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "static_baseline": "Static baseline",
    "linear_regression": "Linear regression",
    "random_forest": "Random forest",
    "gradient_boosting_tuned": "Tuned gradient boosting",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)


def plot_model_errors(metrics: dict, output_dir: Path) -> None:
    rows = [(MODEL_LABELS.get(name, name), values) for name, values in metrics.items()]
    labels = [row[0] for row in rows]
    mae = [row[1]["mae"] for row in rows]
    rmse = [row[1]["rmse"] for row in rows]
    positions = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar([p - width / 2 for p in positions], mae, width, label="MAE", color="#b75d3a")
    ax.bar([p + width / 2 for p in positions], rmse, width, label="RMSE", color="#315c72")
    ax.set_xticks(list(positions), labels, rotation=22, ha="right")
    ax.set_ylabel("Error (units sold)")
    ax.set_title("Out-of-sample demand prediction error")
    ax.legend(frameon=False, ncols=2)
    _style_axes(ax)
    fig.tight_layout()
    _save_figure(fig, output_dir, "model_error_comparison")


def plot_business_outcomes(policy_metrics: dict, output_dir: Path) -> None:
    labels = ["Rule-based\nbaseline"] + [MODEL_LABELS.get(name, name) for name in policy_metrics]
    revenue = [policy_metrics[next(iter(policy_metrics))]["current_revenue"]]
    projected_revenue = [policy_metrics[next(iter(policy_metrics))]["current_revenue"]]
    profit = [policy_metrics[next(iter(policy_metrics))]["current_profit"]]
    projected_profit = [policy_metrics[next(iter(policy_metrics))]["current_profit"]]
    for values in policy_metrics.values():
        revenue.append(values["current_revenue"])
        projected_revenue.append(values["projected_revenue"])
        profit.append(values["current_profit"])
        projected_profit.append(values["projected_profit"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    positions = range(len(labels))
    width = 0.36
    for ax, current, projected, title in (
        (axes[0], revenue, projected_revenue, "Revenue"),
        (axes[1], profit, projected_profit, "Profit"),
    ):
        ax.bar([p - width / 2 for p in positions], current, width, label="Current", color="#9a9a92")
        ax.bar([p + width / 2 for p in positions], projected, width, label="Projected", color="#b75d3a")
        ax.set_xticks(list(positions), labels, rotation=25, ha="right")
        ax.set_ylabel("Value")
        ax.set_title(title)
        _style_axes(ax)
    axes[0].legend(frameon=False, ncols=2)
    fig.suptitle("Offline pricing-policy business outcomes", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir, "business_outcomes")


def plot_feature_importance(feature_path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(feature_path).sort_values("importance").tail(10)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(frame["feature"], frame["importance"], color="#315c72")
    ax.set_xlabel("Importance")
    ax.set_title("Selected-model feature importance")
    _style_axes(ax)
    fig.tight_layout()
    _save_figure(fig, output_dir, "feature_importance")


def plot_price_stability(price_path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(price_path)
    changes = frame["price_change_pct"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(changes, bins=30, color="#b75d3a", alpha=0.88, edgecolor="white")
    axes[0].axvline(changes.median(), color="#315c72", linestyle="--", label=f"Median: {changes.median():.2f}%")
    axes[0].set_xlabel("Price change (%)")
    axes[0].set_ylabel("Products evaluated")
    axes[0].set_title("Distribution of price changes")
    axes[0].legend(frameon=False)
    axes[1].boxplot(
        changes,
        orientation="vertical",
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#d8e2e6"},
    )
    axes[1].set_ylabel("Price change (%)")
    axes[1].set_title("Guardrail-bounded spread")
    for ax in axes:
        _style_axes(ax)
    fig.suptitle("Pricing stability under candidate selection", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir, "price_change_stability")


def plot_policy_comparison(policy_metrics: dict, guardrails: dict, output_dir: Path) -> None:
    names = list(policy_metrics)
    labels = [MODEL_LABELS.get(name, name) for name in names]
    uplift = [policy_metrics[name]["profit_uplift_pct"] for name in names]
    trigger = [policy_metrics[name]["guardrail_trigger_rate_pct"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(labels, uplift, color=["#315c72" if value >= 0 else "#b75d3a" for value in uplift])
    axes[0].axhline(0, color="#444444", linewidth=0.8)
    axes[0].set_ylabel("Profit uplift (%)")
    axes[0].set_title("Policy profit uplift")
    axes[1].bar(labels, trigger, color="#b88a3b")
    axes[1].axhline(guardrails.get("guardrail_trigger_rate_pct", 0), color="#315c72", linestyle="--", label="Selected policy")
    axes[1].set_ylabel("Guardrail-trigger rate (%)")
    axes[1].set_title("Guardrail activity by policy")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        _style_axes(ax)
    fig.suptitle("Policy-level comparison under shared guardrails", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir, "policy_comparison")


def generate_figures(report_dir: Path, output_dir: Path | None = None) -> list[Path]:
    output_dir = output_dir or report_dir / "figures"
    model_metrics = _load_json(report_dir / "model_metrics.json")
    final_metrics = _load_json(report_dir / "final_metrics.json")
    policy_metrics = _load_json(report_dir / "pricing_policy_comparison.json")
    guardrails = _load_json(report_dir / "guardrail_summary.json")
    plot_model_errors(model_metrics, output_dir)
    plot_business_outcomes(policy_metrics, output_dir)
    plot_feature_importance(report_dir / "feature_importance.csv", output_dir)
    plot_price_stability(report_dir / "price_changes.csv", output_dir)
    plot_policy_comparison(policy_metrics, guardrails, output_dir)
    return sorted(output_dir.glob("*.png"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dissertation figures from experiment artifacts")
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    for figure in generate_figures(args.report_dir, args.output_dir):
        print(figure)
