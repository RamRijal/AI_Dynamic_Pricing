from pathlib import Path

import pandas as pd

from src.app.config import REQUIRED_DATASET_COLUMNS, SOURCE_DATASET_COLUMNS


def _read_table(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path)
    return pd.read_csv(dataset_path, low_memory=False)


def _validate_columns(dataframe: pd.DataFrame, required_columns: set[str], label: str) -> pd.DataFrame:
    missing = required_columns.difference(dataframe.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{label} is missing required columns: {missing_list}")
    return dataframe


def load_source_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    dataframe = _read_table(dataset_path)
    return _validate_columns(dataframe, SOURCE_DATASET_COLUMNS, "Source dataset")


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    dataframe = _read_table(dataset_path)
    return _validate_columns(dataframe, REQUIRED_DATASET_COLUMNS, "Experiment dataset")
