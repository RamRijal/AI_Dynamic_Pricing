from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_searchbox import st_searchbox

from src.app.config import MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR
from src.features.build_features import build_features


FEATURE_COLUMNS = [
    "current_price",
    "competitor_price",
    "stock_level",
    "page_views",
    "price_gap",
    "prev_price",
    "prev_units_sold",
    "rolling_avg_sales",
    "stock_ratio",
    "view_to_sales_ratio",
]
ACTIVE_MODEL_NAME = "Random Forest"
ACTIVE_MODEL_FILE = "random_forest.pkl"
MODEL_OPTIONS = {
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "CatBoost": "catboost.pkl",
}


@st.cache_resource
def load_model(model_file: str):
    with (MODELS_DIR / model_file).open("rb") as handle:
        return pickle.load(handle)


@st.cache_data
def load_report(name: str) -> dict:
    path = REPORTS_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_price_changes() -> pd.Series:
    path = REPORTS_DIR / "price_changes.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    return pd.to_numeric(pd.read_csv(path, usecols=["price_change_pct"])["price_change_pct"], errors="coerce").dropna()


MATRIX_PARAMETERS = [
    "page_views",
    "stock_level",
    "stock_ratio",
]

MATRIX_PARAMETER_LABELS = {
    "page_views": "Customer interest (page views)",
    "stock_level": "Inventory available (stock level)",
    "stock_ratio": "Inventory pressure (stock ratio)",
}


@st.cache_data
def load_price_change_matrix() -> tuple[pd.DataFrame, pd.Series]:
    changes_path = REPORTS_DIR / "price_changes.csv"
    featured_path = RAW_DATA_DIR.parent / "processed" / "featured_pricing_data.csv"
    if not changes_path.exists() or not featured_path.exists():
        return pd.DataFrame(), pd.Series(dtype=int)
    changes = pd.read_csv(changes_path, usecols=["product_id", "date", "price_change_pct"])
    features = pd.read_csv(featured_path, usecols=["product_id", "date", *MATRIX_PARAMETERS], dtype={"product_id": str})
    changes["product_id"] = changes["product_id"].astype(str)
    features["product_id"] = features["product_id"].astype(str)
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    merged = changes.merge(features, on=["product_id", "date"], how="inner")
    bins = [-float("inf"), -0.01, 0.01, 5.5, 9.5, float("inf")]
    labels = ["Decrease price", "Hold price", "Standard increase (0-5.5%)", "Guardrail-limited increase (5.5-9.5%)", "Exception / rounding increase (>9.5%)"]
    merged["price_change_band"] = pd.cut(merged["price_change_pct"], bins=bins, labels=labels, include_lowest=True)
    raw_means = merged.groupby("price_change_band", observed=False)[MATRIX_PARAMETERS].mean()
    counts = merged.groupby("price_change_band", observed=False).size()
    global_mean = merged[MATRIX_PARAMETERS].mean()
    global_std = merged[MATRIX_PARAMETERS].std().replace(0, 1)
    standardized = ((raw_means - global_mean) / global_std).round(2)
    return standardized, counts


@st.cache_data
def load_product_catalog() -> pd.DataFrame:
    """Build a lightweight product selector from the latest observed product state."""
    path = RAW_DATA_DIR / "ecommerce_pricing.csv"
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path, low_memory=False)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["product_id"] = raw["product_id"].astype(str)
    featured = build_features(raw)
    latest = featured.sort_values(["product_id", "date"]).groupby("product_id", as_index=False).tail(1).copy()
    frequencies = raw.groupby("product_id").size().rename("frequency")
    latest = latest.join(frequencies, on="product_id")
    source_path = RAW_DATA_DIR.parent / "processed" / "product_names.csv"
    if source_path.exists():
        names = pd.read_csv(source_path, dtype={"product_id": str})
        latest = latest.merge(names, on="product_id", how="left")
    latest["product_name"] = latest["product_name"].fillna(latest["product_id"])
    latest["product_label"] = latest.apply(
        lambda row: f"{row['product_name']} · {row['category']} · {int(row['frequency']):,} records [{row['product_id']}]", axis=1
    )
    return latest.sort_values(["frequency", "product_id"], ascending=[False, True]).reset_index(drop=True)


def product_defaults(product: pd.Series) -> dict:
    current_price = max(float(product.get("current_price", 10.0)), 0.1)
    previous_price = float(product.get("prev_price", 0.0)) or current_price
    previous_units = float(product.get("prev_units_sold", 0.0)) or float(product.get("units_sold", 10.0))
    rolling_sales = float(product.get("rolling_avg_sales", 0.0)) or previous_units
    return {
        "product_id": str(product["product_id"]),
        "current_price": current_price,
        "competitor_price": max(float(product.get("competitor_price", current_price)), 0.1),
        "stock_level": max(float(product.get("stock_level", 100.0)), 0.0),
        "page_views": max(float(product.get("page_views", 250.0)), 0.0),
        "prev_price": max(previous_price, 0.1),
        "prev_units_sold": max(previous_units, 0.0),
        "rolling_avg_sales": max(rolling_sales, 0.0),
        "stock_ratio": min(max(float(product.get("stock_ratio", 0.5)), 0.0), 1.0),
        "view_to_sales_ratio": min(max(float(product.get("view_to_sales_ratio", 20.0)), 0.0), 100.0),
        "unit_cost_pct": min(max(float(product.get("unit_cost", current_price * 0.7)) / current_price, 0.0), 0.99),
    }


def pricing_policy_signal(values: dict) -> tuple[float, float, float]:
    """Return demand pressure, policy target price, and market-relative gap."""
    view_demand = float(values["page_views"]) / max(float(values["view_to_sales_ratio"]), 1.0)
    history_demand = (float(values["prev_units_sold"]) + float(values["rolling_avg_sales"])) / 2
    demand_signal = 0.2 * view_demand + 0.4 * history_demand + 0.4 * max(float(values["prev_units_sold"]), float(values["rolling_avg_sales"]))
    available_stock = max(float(values["stock_level"]), 1.0)
    pressure = demand_signal / available_stock
    pressure_signal = max(-1.0, min(1.0, math.log(max(pressure, 1e-6))))
    market_gap = (float(values["competitor_price"]) - float(values["current_price"])) / max(float(values["current_price"]), 0.1)
    market_gap = max(-0.5, min(0.5, market_gap))
    stock_signal = 0.5 - float(values["stock_ratio"])
    adjustment = max(-float(values["max_change_pct"]), min(float(values["max_change_pct"]), 0.12 * pressure_signal + 0.25 * market_gap + 0.06 * stock_signal))
    target_price = float(values["current_price"]) * (1 + adjustment)
    return pressure, target_price, market_gap


def predict_curve(values: dict, model_file: str = ACTIVE_MODEL_FILE) -> tuple[pd.DataFrame, pd.Series]:
    current_price = values["current_price"]
    change = values["max_change_pct"]
    demand_pressure, policy_target_price, market_gap = pricing_policy_signal(values)
    prices = pd.Series(
        [current_price * (1 - change) + (current_price * 2 * change) * index / 20 for index in range(21)],
        name="price",
    ).round(4)
    frame = pd.DataFrame(
        {
            "current_price": prices,
            "competitor_price": values["competitor_price"],
            "stock_level": values["stock_level"],
            "page_views": values["page_views"],
            "price_gap": prices - values["competitor_price"],
            "prev_price": values["prev_price"],
            "prev_units_sold": values["prev_units_sold"],
            "rolling_avg_sales": values["rolling_avg_sales"],
            "stock_ratio": values["stock_ratio"],
            "view_to_sales_ratio": values["view_to_sales_ratio"],
        }
    )[FEATURE_COLUMNS]
    model = load_model(model_file)
    demand = pd.Series(model.predict(frame).clip(min=0), index=prices.index, name="predicted_units")
    unit_cost = current_price * values["unit_cost_pct"]
    result = pd.DataFrame({"price": prices, "predicted_units": demand})
    result["revenue"] = result["price"] * result["predicted_units"]
    result["profit"] = (result["price"] - unit_cost) * result["predicted_units"]
    result["margin_pct"] = ((result["price"] - unit_cost) / result["price"] * 100).round(2)
    price_span = max(float(prices.max() - prices.min()), 0.01)
    profit_scale = max(float(result["profit"].abs().max()), 1.0)
    result["policy_score"] = result["profit"] - 0.15 * profit_scale * ((result["price"] - policy_target_price) / price_span) ** 2
    result["demand_pressure"] = demand_pressure
    result["policy_target_price"] = policy_target_price
    result["market_gap"] = market_gap
    result["feasible"] = (result["margin_pct"] >= values["min_margin_pct"] * 100)
    feasible = result[result["feasible"]]
    recommended = (feasible if not feasible.empty else result).sort_values(["policy_score", "profit", "revenue"]).iloc[-1]
    return result, recommended


SENSITIVITY_INPUTS = [
    ("current_price", "Current price", 0.05),
    ("competitor_price", "Competitor price", 0.05),
    ("stock_level", "Stock level", 0.10),
    ("page_views", "Page views", 0.10),
    ("prev_units_sold", "Previous units sold", 0.10),
    ("rolling_avg_sales", "Rolling average sales", 0.10),
    ("stock_ratio", "Stock ratio", 0.10),
    ("view_to_sales_ratio", "Views-to-sales ratio", 0.10),
]


def sensitivity_analysis(values: dict, baseline: pd.Series, model_file: str = ACTIVE_MODEL_FILE) -> tuple[pd.DataFrame, float, float]:
    rows = []
    recommendation_changes = 0
    total_scenarios = 0
    for key, label, perturbation in SENSITIVITY_INPUTS:
        base_value = float(values[key])
        if key == "stock_ratio":
            low_value, high_value = max(0.0, base_value - perturbation), min(1.0, base_value + perturbation)
        else:
            low_value, high_value = max(0.0, base_value * (1 - perturbation)), base_value * (1 + perturbation)
        scenario_results = []
        for direction, scenario_value in (("Low", low_value), ("High", high_value)):
            scenario = values.copy()
            scenario[key] = scenario_value
            _, recommendation = predict_curve(scenario, model_file)
            scenario_results.append(recommendation)
            total_scenarios += 1
            recommendation_changes += int(round(float(recommendation["price"]), 2) != round(float(baseline["price"]), 2))
        rows.append(
            {
                "Driver": label,
                "Low scenario recommendation": scenario_results[0]["price"],
                "High scenario recommendation": scenario_results[1]["price"],
                "Low demand": scenario_results[0]["predicted_units"],
                "High demand": scenario_results[1]["predicted_units"],
                "Demand range": scenario_results[1]["predicted_units"] - scenario_results[0]["predicted_units"],
                "Profit range": scenario_results[1]["profit"] - scenario_results[0]["profit"],
            }
        )
    price_scenarios = values.copy()
    price_change = max(0.01, float(values["current_price"]) * 0.05)
    price_scenarios["current_price"] = float(values["current_price"]) - price_change
    low_curve, _ = predict_curve(price_scenarios, model_file)
    price_scenarios["current_price"] = float(values["current_price"]) + price_change
    high_curve, _ = predict_curve(price_scenarios, model_file)
    low_demand = low_curve.iloc[(low_curve["price"] - values["current_price"]).abs().argmin()]["predicted_units"]
    high_demand = high_curve.iloc[(high_curve["price"] - values["current_price"]).abs().argmin()]["predicted_units"]
    elasticity_proxy = ((high_demand - low_demand) / max(low_demand, 1e-9)) / 0.10
    return pd.DataFrame(rows), recommendation_changes / max(total_scenarios, 1) * 100, elasticity_proxy


def factor_effects(values: dict, reference_price: float, model_file: str = ACTIVE_MODEL_FILE) -> pd.DataFrame:
    """Estimate each driver's local effect while holding the price decision fixed."""
    rows = []
    for key, label, perturbation in SENSITIVITY_INPUTS:
        base_value = float(values[key])
        if key == "stock_ratio":
            low_value, high_value = max(0.0, base_value - perturbation), min(1.0, base_value + perturbation)
        else:
            low_value, high_value = max(0.0, base_value * (1 - perturbation)), base_value * (1 + perturbation)
        scenario_rows = []
        for scenario_value in (low_value, high_value):
            scenario = values.copy()
            scenario[key] = scenario_value
            curve, _ = predict_curve({**scenario, "current_price": reference_price}, model_file)
            scenario_rows.append(curve.iloc[(curve["price"] - reference_price).abs().argmin()])
        rows.append(
            {
                "Factor": label,
                "Demand effect (high - low)": float(scenario_rows[1]["predicted_units"] - scenario_rows[0]["predicted_units"]),
                "Profit effect (£)": float(scenario_rows[1]["profit"] - scenario_rows[0]["profit"]),
            }
        )
    return pd.DataFrame(rows).sort_values("Demand effect (high - low)", key=lambda column: column.abs(), ascending=False)


def format_money(value: float) -> str:
    return f"£{value:,.2f}"


def render_curve_chart(curve: pd.DataFrame, recommended: pd.Series) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=curve["price"], y=curve["predicted_units"], name="Predicted demand", mode="lines+markers", line={"color": "#315c72"}, xaxis="x", yaxis="y"))
    figure.add_trace(go.Scatter(x=curve["price"], y=curve["profit"], name="Expected profit", mode="lines+markers", line={"color": "#b75d3a"}, xaxis="x2", yaxis="y2"))
    figure.add_vline(x=float(recommended["price"]), line_dash="dash", line_color="#8a4b2a", annotation_text=f"Recommended £{recommended['price']:.2f}")
    figure.update_layout(
        height=540,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        title="Price-response decision curves",
        grid={"rows": 2, "columns": 1, "pattern": "independent", "roworder": "top to bottom"},
        xaxis={"title": "Candidate price (£)", "domain": [0, 1]},
        yaxis={"title": "Predicted units", "rangemode": "tozero", "domain": [0.56, 1]},
        xaxis2={"title": "Candidate price (£)", "domain": [0, 1]},
        yaxis2={"title": "Expected profit (£)", "rangemode": "tozero", "domain": [0, 0.44]},
        legend={"orientation": "h", "y": 1.1},
        hovermode="x unified",
    )
    return figure


def render_experiment_evidence() -> None:
    metrics = load_report("final_metrics.json")
    business = metrics.get("business_metrics", {})
    stability = metrics.get("stability_metrics", {})
    changes = load_price_changes()
    if not business or changes.empty:
        st.info("Run the experiment to populate the evidence section.")
        return
    st.subheader("Experiment evidence")
    changes = changes.sort_values().reset_index(drop=True)
    empirical_percent = (pd.Series(range(1, len(changes) + 1)) / len(changes) * 100).round(4)
    mean_change = float(changes.mean())
    median_change = float(changes.median())
    percentile_95 = float(changes.quantile(0.95))
    figure = go.Figure(
        go.Scatter(
            x=changes,
            y=empirical_percent,
            mode="lines",
            name="Observed policy decisions",
            line={"color": "#315c72", "width": 3},
            hovertemplate="Price change: %{x:.2f}%<br>Decisions at or below: %{y:.1f}%<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_dash="dot", line_color="#777777", annotation_text="No change")
    figure.add_vline(x=mean_change, line_dash="dash", line_color="#b75d3a", annotation_text=f"Mean {mean_change:.2f}%")
    figure.add_vline(x=median_change, line_dash="dash", line_color="#b88a3b", annotation_text=f"Median {median_change:.2f}%")
    figure.update_layout(
        title="Observed price-change distribution (ECDF)",
        height=430,
        xaxis_title="Price change from current price (%)",
        yaxis_title="Cumulative share of decisions (%)",
        yaxis={"range": [0, 100]},
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
    )
    stats = st.columns(5)
    stats[0].metric("Decisions", f"{len(changes):,}")
    stats[1].metric("Price increases", f"{stability.get('upward_changes', 0):,}")
    stats[2].metric("Price decreases", f"{stability.get('downward_changes', 0):,}")
    stats[3].metric("Unchanged", f"{stability.get('unchanged_prices', 0):,}")
    stats[4].metric("95th percentile", f"{percentile_95:+.2f}%")
    st.caption(
        f"Median absolute price change: {stability.get('median_absolute_price_change_pct', 0):.2f}%. "
        f"Projected revenue uplift: {business.get('revenue_uplift_pct', 0):+.2f}%; "
        f"projected profit uplift: {business.get('profit_uplift_pct', 0):+.2f}%."
    )
    matrix, band_counts = load_price_change_matrix()
    if not matrix.empty:
        visible_bands = [band for band in matrix.index if int(band_counts.get(band, 0)) > 0]
        matrix = matrix.loc[visible_bands]
        y_labels = [f"{band} (n={int(band_counts.get(band, 0)):,})" for band in visible_bands]
        x_labels = [MATRIX_PARAMETER_LABELS[parameter] for parameter in matrix.columns]
        matrix_figure = go.Figure(
            go.Heatmap(
                z=matrix.to_numpy(),
                x=x_labels,
                y=y_labels,
                zmid=0,
                colorscale="RdBu",
                colorbar={"title": "Standardized level"},
                hovertemplate="Price-change band: %{y}<br>Parameter: %{x}<br>Standardized mean: %{z:.2f}<extra></extra>",
            )
        )
        matrix_figure.update_layout(
            title="Manager decision matrix: dominant drivers by pricing action",
            height=520,
            xaxis={"title": "Pricing parameters", "tickangle": -35},
            yaxis={"title": "Price-change combination"},
            margin={"l": 20, "r": 20, "t": 70, "b": 120},
        )
        st.plotly_chart(matrix_figure, use_container_width=True)
        st.caption("Cells show standardized mean driver levels from the recorded experiment for each business pricing action. Red means above the experiment-wide driver mean; blue means below it. The columns are the experiment's dominant drivers: page views, stock level, and stock ratio. This is an association matrix, not causal attribution and is not recalculated when the UI model changes.")


def main() -> None:
    st.set_page_config(page_title="Dynamic Pricing Decision Lab", page_icon="£", layout="wide")
    st.title("Dynamic Pricing Decision Lab")
    st.caption("Model-backed demand analysis for price, revenue, profit, and guardrail decisions")

    catalog = load_product_catalog()
    with st.sidebar:
        st.header("Model evaluation")
        model_names = list(MODEL_OPTIONS)
        selected_model_name = st.selectbox(
            "Demand model",
            model_names,
            index=model_names.index(ACTIVE_MODEL_NAME),
        )
        selected_model_file = MODEL_OPTIONS[selected_model_name]
        metrics = load_report("model_metrics.json").get(
            selected_model_name.lower().replace(" ", "_"), {}
        )
        if metrics:
            st.caption(f"Recorded test performance for {selected_model_name}")
            metric_cols = st.columns(2)
            metric_cols[0].metric("MAE", f"{metrics['mae']:.3f}")
            metric_cols[1].metric("RMSE", f"{metrics['rmse']:.3f}")
        st.caption("Experiment evidence charts below remain fixed to the recorded offline run.")
        st.header("Product selection")
        if catalog.empty:
            st.error("Product catalog unavailable. Check data/raw/ecommerce_pricing.csv.")
            product_options = ["No products available"]
        else:
            product_options = catalog.head(5)["product_label"].tolist()

        def search_products(query: str) -> list[str]:
            if catalog.empty:
                return []
            query = query.strip()
            matches = catalog if not query else catalog[catalog["product_name"].str.contains(query, case=False, na=False)]
            return matches.head(5)["product_label"].tolist()

        selected_label = st_searchbox(
            search_products,
            label="Search product name",
            placeholder="Click to browse or type a name",
            default=product_options[0] if product_options else None,
            default_options=product_options,
            edit_after_submit="option",
            key="product_search",
        )
        if selected_label and selected_label in set(catalog["product_label"]):
            selected_product = catalog[catalog["product_label"] == selected_label].iloc[0]
            defaults = product_defaults(selected_product)
            st.caption(f"Latest observed date: {selected_product['date'].date()} · {int(selected_product['frequency']):,} records")
        else:
            selected_product = pd.Series({"product_id": "default"})
            defaults = product_defaults(selected_product)
            st.info("No matching products. Showing generic inputs.")

        with st.form("decision_inputs"):
            st.header("Operating context")
            widget_suffix = str(defaults["product_id"])
            current_price = st.slider("Current price (£)", 0.1, max(100.0, defaults["current_price"] * 2), defaults["current_price"], 0.1, key=f"current_price_{widget_suffix}")
            competitor_price = st.slider("Competitor price (£)", 0.1, max(100.0, defaults["competitor_price"] * 2), defaults["competitor_price"], 0.1, key=f"competitor_price_{widget_suffix}")
            stock_level = st.slider("Stock level", 0.0, max(1000.0, defaults["stock_level"] * 1.5), defaults["stock_level"], 1.0, key=f"stock_level_{widget_suffix}")
            page_views = st.slider("Page views", 0.0, max(2000.0, defaults["page_views"] * 1.5), defaults["page_views"], 10.0, key=f"page_views_{widget_suffix}")
            st.divider()
            st.header("Demand history")
            prev_price = st.slider("Previous price (£)", 0.1, max(100.0, defaults["prev_price"] * 2), defaults["prev_price"], 0.1, key=f"prev_price_{widget_suffix}")
            prev_units_sold = st.slider("Previous units sold", 0.0, max(200.0, defaults["prev_units_sold"] * 2), defaults["prev_units_sold"], 1.0, key=f"prev_units_sold_{widget_suffix}")
            rolling_avg_sales = st.slider("Rolling average sales", 0.0, max(200.0, defaults["rolling_avg_sales"] * 2), defaults["rolling_avg_sales"], 1.0, key=f"rolling_avg_sales_{widget_suffix}")
            stock_ratio = st.slider("Stock ratio", 0.0, 1.0, defaults["stock_ratio"], 0.01, key=f"stock_ratio_{widget_suffix}")
            view_to_sales_ratio = st.slider("Views-to-sales ratio", 0.0, 100.0, defaults["view_to_sales_ratio"], 1.0, key=f"view_to_sales_ratio_{widget_suffix}")
            st.divider()
            st.header("Commercial guardrails")
            unit_cost_pct = st.slider("Unit cost (% of current price)", 0.0, 0.99, defaults["unit_cost_pct"], 0.01, key=f"unit_cost_pct_{widget_suffix}")
            max_change_pct = st.slider("Maximum price change", 0.0, 0.50, 0.10, 0.01)
            min_margin_pct = st.slider("Minimum margin", 0.0, 0.90, 0.05, 0.01)
            calculate = st.form_submit_button("Calculate output", type="primary", use_container_width=True)

    input_values = {
        key: value
        for key, value in {
            "product_id": defaults["product_id"],
            "current_price": current_price,
            "competitor_price": competitor_price,
            "stock_level": stock_level,
            "page_views": page_views,
            "prev_price": prev_price,
            "prev_units_sold": prev_units_sold,
            "rolling_avg_sales": rolling_avg_sales,
            "stock_ratio": stock_ratio,
            "view_to_sales_ratio": view_to_sales_ratio,
            "unit_cost_pct": unit_cost_pct,
            "max_change_pct": max_change_pct,
            "min_margin_pct": min_margin_pct,
        }.items()
    }
    if calculate or "calculated_values" not in st.session_state:
        st.session_state["calculated_values"] = input_values
    values = st.session_state["calculated_values"]

    st.caption(f"Active demand model: **{selected_model_name}**")
    curve, recommended = predict_curve(values, selected_model_file)
    current_row = curve.iloc[(curve["price"] - values["current_price"]).abs().argmin()]
    sensitivity, change_probability, elasticity_proxy = sensitivity_analysis(values, recommended, selected_model_file)
    effects = factor_effects(values, float(recommended["price"]), selected_model_file)

    st.subheader("Recommended pricing action")
    if not calculate and "calculated_values" in st.session_state and st.session_state["calculated_values"] != input_values:
        st.info("Inputs changed. Press **Calculate output** to update the model result.")
    kpis = st.columns(5)
    kpis[0].metric("Recommended price", format_money(recommended["price"]))
    kpis[1].metric("Predicted units", f"{recommended['predicted_units']:,.1f}")
    kpis[2].metric("Expected revenue", format_money(recommended["revenue"]))
    kpis[3].metric("Expected profit", format_money(recommended["profit"]))
    kpis[4].metric("Profit change", f"{recommended['profit'] - current_row['profit']:+,.2f}")

    chart, explanation = st.columns([2.4, 1])
    with chart:
        st.plotly_chart(render_curve_chart(curve, recommended), use_container_width=True)
    with explanation:
        st.markdown("#### Manager readout")
        top_effects = effects.head(3)
        effect_text = "; ".join(f"{row['Factor']} changes demand by {row['Demand effect (high - low)']:+,.1f} units and profit by £{row['Profit effect (£)']:+,.2f}" for _, row in top_effects.iterrows())
        upper_guardrail = values["current_price"] * (1 + values["max_change_pct"])
        guardrail_binding = abs(float(recommended["price"]) - upper_guardrail) <= max(values["current_price"] * 0.001, 0.01)
        st.write(f"The model recommends **{format_money(recommended['price'])}**, the highest-profit feasible candidate after applying the margin and price-change guardrails.")
        st.write(f"At the current price, predicted demand is **{current_row['predicted_units']:,.1f} units**; at the recommendation it is **{recommended['predicted_units']:,.1f} units**, with expected profit changing by **£{recommended['profit'] - current_row['profit']:+,.2f}**.")
        st.write(f"The strongest local drivers are: {effect_text}.")
        st.write(f"Demand pressure is **{recommended['demand_pressure']:.2f}x available stock** and the policy target is **{format_money(recommended['policy_target_price'])}** before the guardrail is applied.")
        if guardrail_binding:
            st.warning(f"The recommendation is at the upper price-change ceiling of **{format_money(upper_guardrail)}**. Increase Maximum price change if you want to test whether the demand signal supports a higher price.")
        st.write(f"The search range is **{format_money(values['current_price'] * (1 - values['max_change_pct']))}–{format_money(values['current_price'] * (1 + values['max_change_pct']))}**. If the recommended price is unchanged after a driver moves, that driver changed demand or profit levels without changing the best-ranked candidate in this constrained grid.")
        st.caption("This is an offline model simulation, not a causal guarantee of live sales.")

    st.subheader("Inferred change likelihood and sensitivity")
    probability_col, elasticity_col, pressure_col, target_col = st.columns(4)
    probability_col.metric("Recommendation-change probability proxy", f"{change_probability:.1f}%")
    elasticity_col.metric("Local price-elasticity proxy", f"{elasticity_proxy:.2f}")
    pressure_col.metric("Demand pressure", f"{recommended['demand_pressure']:.2f}x stock")
    target_col.metric("Policy target before guardrail", format_money(recommended["policy_target_price"]))
    st.caption("The change likelihood is inferred from low/high perturbations of the input drivers. It is a local sensitivity measure, not a calibrated probability or confidence interval.")
    st.dataframe(
        sensitivity.style.format(
            {
                "Low scenario recommendation": "£{:,.2f}",
                "High scenario recommendation": "£{:,.2f}",
                "Low demand": "{:,.1f}",
                "High demand": "{:,.1f}",
                "Demand range": "{:+,.1f}",
                "Profit range": "£{:+,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Counterfactual effects hold the recommended price fixed so each factor's demand and profit impact can be read separately.")
    st.dataframe(
        effects.style.format({"Demand effect (high - low)": "{:+,.1f}", "Profit effect (£)": "£{:+,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Candidate price table"):
        st.dataframe(curve.style.format({"price": "£{:,.2f}", "predicted_units": "{:,.1f}", "revenue": "£{:,.2f}", "profit": "£{:,.2f}", "margin_pct": "{:.2f}%"}), use_container_width=True, hide_index=True)

    render_experiment_evidence()


if __name__ == "__main__":
    main()
