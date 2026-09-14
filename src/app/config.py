from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PIPELINE_MANIFEST_PATH = PROCESSED_DATA_DIR / "pipeline_run_manifest.json"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DB_PATH = ROOT_DIR / "pricing.db"
DB_URL = f"sqlite:///{DB_PATH}"

REQUIRED_DATASET_COLUMNS = {
    "product_id",
    "category",
    "date",
    "current_price",
    "units_sold",
    "stock_level",
    "competitor_price",
    "page_views",
}

SOURCE_DATASET_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
}
