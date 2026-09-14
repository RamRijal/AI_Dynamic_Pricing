from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_path: str = "data/raw/sample_products.csv"


class PricingCycleRequest(BaseModel):
    dataset_path: str = "data/raw/sample_products.csv"
    min_price: float = Field(default=1.0, gt=0)
    max_price: float = Field(default=10_000.0, gt=0)
    max_change_pct: float = Field(default=0.10, ge=0, le=1)
    min_margin_pct: float = Field(default=0.05, ge=0, le=1)


class DemandPredictionRequest(BaseModel):
    current_price: float = Field(default=10.0, gt=0)
    competitor_price: float = Field(default=10.0, gt=0)
    stock_level: float = Field(default=100.0, ge=0)
    page_views: float = Field(default=250.0, ge=0)
    prev_price: float = Field(default=10.0, gt=0)
    prev_units_sold: float = Field(default=10.0, ge=0)
    rolling_avg_sales: float = Field(default=10.0, ge=0)
    stock_ratio: float = Field(default=0.5, ge=0, le=1)
    view_to_sales_ratio: float = Field(default=20.0, ge=0)
    unit_cost_pct: float = Field(default=0.70, ge=0, le=0.99)
    max_change_pct: float = Field(default=0.10, ge=0, le=1)
    min_margin_pct: float = Field(default=0.05, ge=0, le=1)
