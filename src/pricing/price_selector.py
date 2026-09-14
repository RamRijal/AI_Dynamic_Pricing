def choose_best_price(candidates: list[dict]) -> dict:
    scored = []
    for item in candidates:
        profit = (item["price"] - item["unit_cost"]) * item["predicted_units"]
        revenue = item["price"] * item["predicted_units"]
        scored.append({**item, "predicted_profit": profit, "predicted_revenue": revenue})
    return max(scored, key=lambda item: (item["predicted_profit"], item["predicted_revenue"]))
