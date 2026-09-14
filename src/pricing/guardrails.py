from __future__ import annotations


def evaluate_guardrails(
    current_price: float,
    suggested_price: float,
    min_price: float,
    max_price: float,
    max_change_pct: float,
    unit_cost: float | None = None,
    min_margin_pct: float = 0.0,
) -> dict:
    reasons: list[str] = []
    lower_change = current_price * (1 - max_change_pct)
    upper_change = current_price * (1 + max_change_pct)
    bounded = suggested_price

    if suggested_price < min_price:
        bounded = max(bounded, min_price)
        reasons.append("min_price_floor")

    if suggested_price > max_price:
        bounded = min(bounded, max_price)
        reasons.append("max_price_ceiling")

    if unit_cost is not None:
        margin_floor = unit_cost * (1 + min_margin_pct)
        if bounded < margin_floor:
            bounded = margin_floor
            reasons.append("min_margin_floor")

    if bounded < lower_change:
        bounded = lower_change
        reasons.append("max_change_lower_bound")

    if bounded > upper_change:
        bounded = upper_change
        reasons.append("max_change_upper_bound")

    final_price = round(bounded, 2)
    return {
        "final_price": final_price,
        "feasible": not reasons,
        "reasons": reasons,
    }


def apply_guardrails(
    current_price: float,
    suggested_price: float,
    min_price: float,
    max_price: float,
    max_change_pct: float,
    unit_cost: float | None = None,
    min_margin_pct: float = 0.0,
) -> float:
    result = evaluate_guardrails(
        current_price=current_price,
        suggested_price=suggested_price,
        min_price=min_price,
        max_price=max_price,
        max_change_pct=max_change_pct,
        unit_cost=unit_cost,
        min_margin_pct=min_margin_pct,
    )
    return result["final_price"]
