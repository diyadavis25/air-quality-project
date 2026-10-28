"""Plotly chart builders (all return go.Figure) with a restrained burgundy/off-white style."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config

FONT = dict(family="Inter, Segoe UI, sans-serif", color="#2B2B2B")


def _style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(family="Georgia, serif", size=18)), height=height,
                      paper_bgcolor="#FAF8F5", plot_bgcolor="#FAF8F5", font=FONT,
                      margin=dict(l=40, r=20, t=60, b=40), legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showline=True, linecolor="#999", gridcolor="#E6E2DB")
    fig.update_yaxes(showline=True, linecolor="#999", gridcolor="#E6E2DB")
    return fig


def aqi_histogram(df):
    fig = px.histogram(df, x="AQI", nbins=60, color_discrete_sequence=[config.ACCENT], marginal="box")
    fig.update_xaxes(title="AQI"); fig.update_yaxes(title="Number of days")
    return _style(fig, "Overall AQI distribution")


def category_bar(cat_df):
    fig = px.bar(cat_df, x="category", y="days", color="category", color_discrete_map=config.AQI_COLORS,
                 text="percent", category_orders={"category": config.AQI_ORDER})
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False)
    return _style(fig, "AQI category distribution (CPCB thresholds)")


def daily_line(daily):
    fig = px.line(daily, x="Date", y="AQI", color_discrete_sequence=[config.ACCENT])
    fig.update_yaxes(title="Mean AQI"); fig.update_traces(line_width=1)
    return _style(fig, "Daily mean AQI (selected cities)")


def monthly_line(monthly):
    fig = px.line(monthly, x="Month", y="AQI", markers=True, color_discrete_sequence=[config.ACCENT])
    fig.update_yaxes(title="Mean AQI")
    return _style(fig, "Monthly mean AQI over time")


def month_of_year_bar(m):
    fig = px.bar(m, x="MonthName", y="AQI", color="AQI", color_continuous_scale=["#F0D9DC", config.ACCENT],
                 category_orders={"MonthName": list(m["MonthName"])})
    fig.update_yaxes(title="Mean AQI"); fig.update_xaxes(title="")
    return _style(fig, "Average AQI by calendar month")


def yearly_bar(y):
    fig = px.bar(y, x="Year", y="mean_aqi", text="mean_aqi", hover_data=["observations", "cities"],
                 color_discrete_sequence=[config.ACCENT])
    fig.update_xaxes(type="category"); fig.update_yaxes(title="Mean AQI")
    return _style(fig, "Yearly mean AQI")


def season_bar(s):
    fig = px.bar(s, x="Season", y="mean", color_discrete_sequence=[config.ACCENT], hover_data=["size"])
    fig.update_yaxes(title="Mean AQI")
    return _style(fig, "Seasonal mean AQI")


def city_bar(ranking, n=None):
    r = ranking if n is None else ranking.head(n)
    r = r.sort_values("mean_aqi")
    fig = px.bar(r, x="mean_aqi", y="City", orientation="h", color="mean_aqi",
                 color_continuous_scale=["#F0D9DC", config.ACCENT], hover_data=["median_aqi", "max_aqi", "observations"])
    fig.update_xaxes(title="Mean AQI"); fig.update_yaxes(title="")
    return _style(fig, "City-wise mean AQI", height=max(380, 24 * len(r) + 120))


def city_month_heatmap(pivot):
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale=["#FAF8F5", "#E5B1B8", config.ACCENT, "#3B0A12"],
                    labels=dict(x="Month", y="City", color="Mean AQI"))
    fig.update_xaxes(tickvals=list(range(12)), ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    return _style(fig, "City-wise pollution heatmap (mean AQI by month)", height=max(400, 22 * len(pivot) + 120))


def correlation_heatmap(corr, method="Pearson"):
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
    return _style(fig, f"{method} correlation matrix", height=520)


def scatter_vs_aqi(df, pollutant, sample=8000):
    d = df[[pollutant, "AQI", "City"]].dropna()
    if len(d) > sample:
        d = d.sample(sample, random_state=config.RANDOM_SEED)
    fig = px.scatter(d, x=pollutant, y="AQI", color="City", opacity=0.5)
    return _style(fig, f"AQI vs {pollutant}")


def pollutant_box(df, pollutants):
    long = df[pollutants].melt(var_name="Pollutant", value_name="Concentration").dropna()
    fig = px.box(long, x="Pollutant", y="Concentration", color_discrete_sequence=[config.ACCENT])
    fig.update_yaxes(title="Concentration (dataset units)", type="log")
    return _style(fig, "Pollutant distributions (log scale)")


def pollutant_hist(df, pollutant):
    fig = px.histogram(df, x=pollutant, nbins=60, color_discrete_sequence=[config.ACCENT])
    return _style(fig, f"Distribution of {pollutant}", height=340)


def pollutant_means_bar(df, pollutants):
    m = df[pollutants].mean().reset_index(); m.columns = ["Pollutant", "Mean"]
    fig = px.bar(m, x="Pollutant", y="Mean", color_discrete_sequence=[config.ACCENT])
    return _style(fig, "Mean concentration per pollutant (units differ: CO is mg/m³, others µg/m³)", height=360)


def actual_vs_predicted(preds):
    d = preds.sample(min(len(preds), 6000), random_state=config.RANDOM_SEED)
    fig = px.scatter(d, x="actual", y="predicted", opacity=0.4, color_discrete_sequence=[config.ACCENT])
    lim = [0, max(d["actual"].max(), d["predicted"].max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#555", dash="dash"), name="Perfect fit"))
    fig.update_xaxes(title="Actual AQI"); fig.update_yaxes(title="Predicted AQI")
    return _style(fig, "Actual vs predicted AQI (test set)")


def residual_plot(preds):
    fig = px.scatter(preds, x="predicted", y="residual", opacity=0.4, color_discrete_sequence=[config.ACCENT])
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    fig.update_yaxes(title="Residual (actual − predicted)")
    return _style(fig, "Residuals vs predicted (test set)")


def residual_hist(preds):
    fig = px.histogram(preds, x="residual", nbins=50, color_discrete_sequence=[config.ACCENT])
    return _style(fig, "Residual distribution", height=340)


def metric_comparison(results: pd.DataFrame):
    long = results.melt(id_vars="model", value_vars=["test_MAE", "test_RMSE"], var_name="metric", value_name="value")
    fig = px.bar(long, x="model", y="value", color="metric", barmode="group",
                 color_discrete_sequence=[config.ACCENT, "#B9A38A"])
    return _style(fig, "Test-set error by model (lower is better)", height=360)


def importance_bar(imp: dict, title="Random Forest feature importance"):
    s = pd.Series(imp).sort_values()
    fig = px.bar(x=s.values, y=s.index, orientation="h", color_discrete_sequence=[config.ACCENT])
    fig.update_xaxes(title="Importance"); fig.update_yaxes(title="")
    return _style(fig, title, height=360)
