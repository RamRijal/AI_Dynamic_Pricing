# AI Dynamic Pricing Prototype

Research prototype for a master's dissertation on AI-driven dynamic pricing for e-commerce.

## Project Goal

Build a reproducible pricing research workflow that:

- prepares a pricing dataset
- engineers demand, inventory, and competitor features
- compares baseline and ML-driven pricing approaches
- applies business guardrails to recommended prices
- exports results that can support dissertation analysis, tables, and figures

## Current Scope

This repository is for an offline research prototype, not a production commerce platform.

In scope:

- batch dataset preparation
- baseline and ML model comparison
- guarded pricing-cycle simulation
- API and dashboard for demonstration
- dissertation-oriented reporting artifacts

Out of scope for the first serious version:

- live store integration
- real-time scraping
- enterprise deployment concerns
- a polished customer-facing storefront

## Repository Structure

```text
src/
  app/        application entrypoints and runtime config
  data/       dataset loading and cleaning
  features/   feature engineering
  ml/         model training, evaluation, and artifact saving
  pricing/    decision logic and guardrails
  dashboard/  minimal demo interface
data/
  raw/        source datasets
  processed/  cleaned and featured outputs
artifacts/
  models/     trained model files
  reports/    metrics, price logs, and dissertation-ready outputs
tests/        smoke and unit tests
docs/         design and implementation planning
outputs/      supporting dissertation documents and progress reports
```

## Dissertation-Grade Standard

The project is not considered complete because files exist or tests pass. A serious milestone should produce:

- a documented dataset suitable for experiments
- reproducible training and evaluation runs
- non-empty model and report artifacts
- meaningful baseline-versus-ML comparisons
- outputs that can be reused in dissertation writing

## Quick Start

Install the working Python dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `catboost`
- `fastapi`
- `pytest`
- `httpx`
- `uvicorn`
- `streamlit`
- `plotly`
- `tqdm`

Or install them from the pinned working list:

```powershell
pip install -r requirements.txt
```

Run the current smoke and unit tests:

```powershell
pytest tests -v
```

Run the prototype API locally:

```powershell
uvicorn src.app.main:app --reload

# Interactive decision UI
streamlit run streamlit_app.py
```

Run the full offline experiment with batched inference, CLI-configurable training parameters, and two epochs by default:

```powershell
python src/ml/run_real_experiment.py --epochs 2 --gb-n-estimators 300 --xgb-n-estimators 300 --cat-iterations 300
```

Useful endpoints:

- `GET /health`
- `POST /train`
- `GET /metrics`
- `POST /pricing-cycle`
- `GET /price-history`

## Current Status

The repository currently has a working bootstrap scaffold. The next work is to replace placeholder-style outputs with dissertation-grade data, trained artifacts, and experiment reports.

See [progress-report.md](C:/Users/laxman/Documents/dynamic-pricing-system/outputs/progress-report.md) for task-by-task status.
