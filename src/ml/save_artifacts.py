import json
import pickle
from pathlib import Path


def save_json_report(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_csv_report(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return

    header = list(rows[0].keys())
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in header))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_model_artifact(path: str | Path, model) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file_handle:
        pickle.dump(model, file_handle)
