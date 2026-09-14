from pathlib import Path
import html
import json

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

from src.app.config import REPORTS_DIR


router = APIRouter()
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _read_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _load_json_report(name: str) -> dict:
    path = REPORTS_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_markdown_report(name: str) -> str:
    path = REPORTS_DIR / name
    if not path.exists():
        return "<p>Report not available yet.</p>"
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    in_list = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if in_list:
                blocks.append("</ul>")
                in_list = False
            continue
        if line.startswith("# "):
            if in_list:
                blocks.append("</ul>")
                in_list = False
            blocks.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_list:
                blocks.append("</ul>")
                in_list = False
            blocks.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                blocks.append("<ul>")
                in_list = True
            blocks.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            if in_list:
                blocks.append("</ul>")
                in_list = False
            blocks.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        blocks.append("</ul>")
    return "".join(blocks)


@router.get("/", response_class=HTMLResponse)
def index() -> str:
    return _read_template("index.html")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    summary = _load_json_report("experiment_summary.json")
    metrics = _load_json_report("final_metrics.json")
    technical = metrics.get("technical_metrics", {})
    business = metrics.get("business_metrics", {})
    rows = []
    for model_name, values in technical.items():
        rows.append(
            f"<tr><td>{html.escape(model_name)}</td><td>{values.get('mae', 0):.4f}</td><td>{values.get('rmse', 0):.4f}</td></tr>"
        )

    result_template = _read_template("results.html")
    replacements = {
        "__DATASET_PATH__": html.escape(summary.get("dataset_path", "Unavailable")),
        "__ROW_COUNT__": f"{summary.get('rows', 0):,}",
        "__PRODUCT_COUNT__": f"{summary.get('unique_products', 0):,}",
        "__CATEGORY_COUNT__": f"{summary.get('categories', 0):,}",
        "__DATE_MIN__": html.escape(summary.get("date_min", "Unavailable")),
        "__DATE_MAX__": html.escape(summary.get("date_max", "Unavailable")),
        "__SPLIT_DATE__": html.escape(summary.get("split_date", "Unavailable")),
        "__REVENUE_UPLIFT__": f"{business.get('revenue_uplift_pct', 0):.2f}%",
        "__PROFIT_UPLIFT__": f"{business.get('profit_uplift_pct', 0):.2f}%",
        "__CURRENT_REVENUE__": f"{business.get('current_revenue', 0):,.2f}",
        "__PROJECTED_REVENUE__": f"{business.get('projected_revenue', 0):,.2f}",
        "__CURRENT_PROFIT__": f"{business.get('current_profit', 0):,.2f}",
        "__PROJECTED_PROFIT__": f"{business.get('projected_profit', 0):,.2f}",
        "__METRICS_ROWS__": "".join(rows) or "<tr><td colspan='3'>No metrics available</td></tr>",
        "__SUMMARY_HTML__": _load_markdown_report("dissertation_results_summary.md"),
    }
    for key, value in replacements.items():
        result_template = result_template.replace(key, value)
    return result_template


@router.get("/reports/figures/{figure_name}")
def report_figure(figure_name: str) -> FileResponse:
    figures_dir = REPORTS_DIR / "figures"
    requested = (figures_dir / figure_name).resolve()
    if requested.parent != figures_dir.resolve() or requested.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("Unsupported figure path")
    if not requested.exists():
        raise FileNotFoundError(requested)
    return FileResponse(requested)
