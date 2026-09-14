from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import pandas as pd

from src.app.db import insert_price_history
from src.pricing.guardrails import evaluate_guardrails
from src.pricing.price_selector import choose_best_price

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable


def _candidate_prices(current_price: float) -> list[float]:
    return [round(current_price * factor, 2) for factor in (0.95, 1.0, 1.05)]


def _estimate_units(row: pd.Series, candidate_price: float) -> float:
    baseline_units = max(float(row.get("prev_units_sold", row.get("units_sold", 1))), 1.0)
    price_gap = candidate_price - float(row["competitor_price"])
    penalty = price_gap * 0.08
    demand_boost = min(float(row.get("page_views", 0)) / 500, 5)
    return max(1.0, baseline_units + demand_boost - penalty)


def _model_feature_row(row: pd.Series, candidate_price: float) -> dict:
    candidate = row.to_dict()
    candidate["current_price"] = candidate_price
    candidate["price_gap"] = candidate_price - float(row["competitor_price"])
    if "base_price" in candidate and float(candidate.get("base_price", 0) or 0) != 0:
        candidate["discount_pct"] = 1 - (candidate_price / float(candidate["base_price"]))
    return candidate


def run_pricing_cycle(
    df: pd.DataFrame,
    min_price: float = 1.0,
    max_price: float = 10_000.0,
    max_change_pct: float = 0.10,
    min_margin_pct: float = 0.05,
    persist_history: bool = True,
    unit_estimator: Callable[[pd.Series, float], float] | None = None,
    unit_estimator_batch: Callable[[pd.DataFrame], pd.Series | list[float]] | None = None,
    show_progress: bool = False,
    progress_desc: str = "pricing-cycle",
) -> pd.DataFrame:
    updated = df.copy()
    price_history_rows: list[tuple[str, float, float, float, str]] = []
    run_timestamp = datetime.now(timezone.utc).isoformat()

    candidate_feature_rows: list[dict] = []
    candidate_metadata: list[dict] = []
    row_iterator = updated.iterrows()
    if show_progress:
        row_iterator = tqdm(row_iterator, total=len(updated), desc=progress_desc)

    for _, row in row_iterator:
        unit_cost = float(row.get("unit_cost", row["current_price"] * 0.7))
        for candidate_price in _candidate_prices(float(row["current_price"])):
            candidate_feature_rows.append(_model_feature_row(row, candidate_price))
            candidate_metadata.append(
                {
                    "row": row,
                    "price": float(candidate_price),
                    "unit_cost": unit_cost,
                }
            )

    if unit_estimator_batch:
        candidate_frame = pd.DataFrame(candidate_feature_rows)
        predicted_units = pd.Series(unit_estimator_batch(candidate_frame), index=candidate_frame.index)
    else:
        predicted_units = pd.Series(
            [
                float(unit_estimator(meta["row"], meta["price"])) if unit_estimator else _estimate_units(meta["row"], meta["price"])
                for meta in candidate_metadata
            ]
        )

    final_rows = []
    for row_index in range(len(updated)):
        row = candidate_metadata[row_index * 3]["row"]
        candidates = []
        for candidate_offset in range(3):
            meta = candidate_metadata[(row_index * 3) + candidate_offset]
            candidates.append(
                {
                    "price": meta["price"],
                    "predicted_units": max(1.0, float(predicted_units.iloc[(row_index * 3) + candidate_offset])),
                    "unit_cost": float(meta["unit_cost"]),
                }
            )
        evaluated_candidates = []
        for candidate in candidates:
            guardrail_result = evaluate_guardrails(
                current_price=float(row["current_price"]),
                suggested_price=float(candidate["price"]),
                min_price=min_price,
                max_price=max_price,
                max_change_pct=max_change_pct,
                unit_cost=float(candidate["unit_cost"]),
                min_margin_pct=min_margin_pct,
            )
            evaluated_candidates.append(
                {
                    **candidate,
                    "guarded_price": float(guardrail_result["final_price"]),
                    "guardrail_feasible": bool(guardrail_result["feasible"]),
                    "guardrail_reasons": "|".join(guardrail_result["reasons"]),
                }
            )

        feasible_candidates = [item for item in evaluated_candidates if item["guardrail_feasible"]]
        candidate_pool = feasible_candidates or evaluated_candidates
        best = choose_best_price(candidate_pool)
        final_price = float(best.get("guarded_price", best["price"]))
        final_units = float(best["predicted_units"])
        final_profit = (final_price - float(best["unit_cost"])) * final_units
        final_revenue = final_price * final_units
        selection_reason = "best_feasible_candidate" if feasible_candidates else "best_adjusted_candidate"
        change_pct = ((final_price - float(row["current_price"])) / float(row["current_price"])) * 100 if float(row["current_price"]) else 0.0

        final_rows.append(
            {
                **row.to_dict(),
                "suggested_price": float(best["price"]),
                "new_price": final_price,
                "predicted_units": final_units,
                "predicted_profit": round(float(final_profit), 4),
                "predicted_revenue": round(float(final_revenue), 4),
                "price_change_pct": round(float(change_pct), 4),
                "feasible_candidate_count": len(feasible_candidates),
                "candidate_count": len(evaluated_candidates),
                "selection_reason": selection_reason,
                "guardrail_feasible": bool(best["guardrail_feasible"]),
                "guardrail_reasons": str(best["guardrail_reasons"]),
                "run_timestamp": run_timestamp,
            }
        )
        price_history_rows.append(
            (
                str(row["product_id"]),
                float(row["current_price"]),
                float(best["price"]),
                final_price,
                run_timestamp,
            )
        )

    if persist_history:
        insert_price_history(price_history_rows)
    return pd.DataFrame(final_rows)
