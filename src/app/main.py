from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import pickle

import pandas as pd

from fastapi import FastAPI

from src.app.db import fetch_price_history, initialize_database
from src.app.config import MODELS_DIR
from src.app.schemas import DemandPredictionRequest, PricingCycleRequest, TrainRequest
from src.dashboard.routes import router as dashboard_router
from src.data.cleaning import clean_dataset
from src.data.load_dataset import load_dataset
from src.features.build_features import build_features
from src.ml.evaluate_models import compare_model_metrics, extend_metrics_with_advanced_models
from src.ml.train_advanced_models import train_advanced_models
from src.ml.train_baselines import train_baselines
from src.ml.train_main_model import train_main_model
from src.pricing.run_pricing_cycle import run_pricing_cycle

app = FastAPI(title="AI Dynamic Pricing Prototype")
app.include_router(dashboard_router)
initialize_database()


@lru_cache(maxsize=1)
def _load_demand_model():
    model_path = MODELS_DIR / "random_forest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Demand model not found: {model_path}")
    with model_path.open("rb") as file_handle:
        return pickle.load(file_handle)


def _prediction_frame(request: DemandPredictionRequest, prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "current_price": price,
                "competitor_price": request.competitor_price,
                "stock_level": request.stock_level,
                "page_views": request.page_views,
                "price_gap": price - request.competitor_price,
                "prev_price": request.prev_price,
                "prev_units_sold": request.prev_units_sold,
                "rolling_avg_sales": request.rolling_avg_sales,
                "stock_ratio": request.stock_ratio,
                "view_to_sales_ratio": request.view_to_sales_ratio,
            }
            for price in prices
        ]
    )


def _guarded_price(current_price: float, price: float, max_change_pct: float) -> tuple[float, bool]:
    lower = current_price * (1 - max_change_pct)
    upper = current_price * (1 + max_change_pct)
    guarded = min(max(price, lower), upper)
    return round(guarded, 4), lower <= price <= upper


def run_experiment(dataset_path: str) -> dict:
    raw = load_dataset(dataset_path)
    cleaned = clean_dataset(raw)
    featured = build_features(cleaned)
    baseline_models = train_baselines(featured)
    main_model = train_main_model(featured)
    advanced_models = train_advanced_models(featured)
    metrics = compare_model_metrics(featured, baseline_models, main_model)
    metrics = extend_metrics_with_advanced_models(metrics, featured, advanced_models)
    return {
        "dataset_path": str(Path(dataset_path)),
        "row_count": len(featured),
        "feature_columns": list(featured.columns),
        "baselines": baseline_models,
        "main_model": main_model,
        "advanced_models": advanced_models,
        "metrics": metrics,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/demand-prediction")
def demand_prediction(request: DemandPredictionRequest) -> dict:
    model = _load_demand_model()
    lower_price = max(0.01, request.current_price * (1 - request.max_change_pct))
    upper_price = request.current_price * (1 + request.max_change_pct)
    prices = [round(value, 4) for value in pd.Series(range(21)).map(lambda index: lower_price + (upper_price - lower_price) * index / 20)]
    frame = _prediction_frame(request, prices)
    predictions = model.predict(frame).clip(min=0)
    unit_cost = request.current_price * request.unit_cost_pct
    curve = []
    for price, demand in zip(prices, predictions):
        guarded_price, feasible = _guarded_price(request.current_price, price, request.max_change_pct)
        revenue = price * float(demand)
        profit = (price - unit_cost) * float(demand)
        margin = (price - unit_cost) / price if price else 0.0
        curve.append(
            {
                "price": round(price, 4),
                "predicted_units": round(float(demand), 4),
                "revenue": round(revenue, 4),
                "profit": round(profit, 4),
                "margin_pct": round(margin * 100, 4),
                "guarded_price": guarded_price,
                "feasible": feasible and margin >= request.min_margin_pct,
            }
        )
    feasible_curve = [point for point in curve if point["feasible"]]
    recommended = max(feasible_curve or curve, key=lambda point: (point["profit"], point["revenue"]))
    return {
        "model": "random_forest",
        "unit_cost": round(unit_cost, 4),
        "recommended": recommended,
        "current": next(point for point in curve if point["price"] == min(prices, key=lambda price: abs(price - request.current_price))),
        "curve": curve,
    }


@app.post("/train")
def train(request: TrainRequest) -> dict:
    result = run_experiment(request.dataset_path)
    return {
        "dataset_path": result["dataset_path"],
        "row_count": result["row_count"],
        "metrics": result["metrics"],
    }


@app.get("/metrics")
def metrics() -> dict:
    result = run_experiment("data/raw/sample_products.csv")
    return result["metrics"]


@app.post("/pricing-cycle")
def pricing_cycle(request: PricingCycleRequest) -> dict:
    dataset = load_dataset(request.dataset_path)
    cleaned = clean_dataset(dataset)
    featured = build_features(cleaned)
    priced = run_pricing_cycle(
        featured,
        min_price=request.min_price,
        max_price=request.max_price,
        max_change_pct=request.max_change_pct,
        min_margin_pct=request.min_margin_pct,
    )
    return {"rows": priced.to_dict(orient="records")}


@app.get("/price-history")
def price_history() -> list[dict]:
    return fetch_price_history()
