"""Streamlit dashboard. Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st
import config
from src import eda, pollution_analysis as pa, visualizations as viz
from src.data_loader import DatasetError, load_csv
from src.model_training import train_and_save
from src.prediction import ModelNotTrainedError, load_artifacts, predict_aqi
from src.preprocessing import clean, data_dictionary, filter_data, flag_outliers_iqr, missing_summary

st.set_page_config(page_title="Air Quality Prediction & Pollution Pattern Analysis", page_icon="🌫️", layout="wide")
st.markdown("""
<style>
h1,h2,h3 {font-family: Georgia, 'Times New Roman', serif !important; color:#2B2B2B;}
.block-container {padding-top: 2rem;}
div[data-testid="stMetric"] {background:#fff; border:1px solid #D9D3CA; border-radius:4px; padding:12px 16px;}
div[data-testid="stMetricValue"] {color:#7B1E2B; font-family: Georgia, serif;}
section[data-testid="stSidebar"] {border-right:1px solid #D9D3CA;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and cleaning data…")
def get_data(uploaded_bytes=None):
    src = pd.io.common.BytesIO(uploaded_bytes) if uploaded_bytes else None
    return clean(load_csv(src))


# ---------- sidebar ----------
st.sidebar.title("Air Quality")
up = st.sidebar.file_uploader("Use a different compatible CSV (optional)", type="csv")
try:
    DF, REPORT = get_data(up.getvalue() if up else None)
except DatasetError as e:
    st.title("Air Quality Prediction & Pollution Pattern Analysis")
    st.error("Dataset problem")
    st.code(str(e))
    st.stop()

POLL = REPORT["pollutants"]
PAGES = ["Home / Overview", "Dataset Explorer", "Exploratory Data Analysis", "Pollution Pattern Analysis",
         "Pollutant Correlation", "AQI Prediction", "Model Performance", "About the Project"]
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
cities = st.sidebar.multiselect("City", sorted(DF["City"].unique()))
dmin, dmax = DF["Date"].min().date(), DF["Date"].max().date()
drange = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
years = st.sidebar.multiselect("Year", sorted(DF["Year"].unique()))
months = st.sidebar.multiselect("Month", list(range(1, 13)), format_func=lambda m: pd.to_datetime(m, format="%m").strftime("%B"))
if len(drange) != 2:
    st.sidebar.warning("Select both a start and an end date.")
    drange = (dmin, dmax)
F = filter_data(DF, cities, drange, years, months)

st.title("Air Quality Prediction & Pollution Pattern Analysis")
st.caption("Data-driven insights into pollution trends and AQI prediction")

if F.empty and page not in ("AQI Prediction", "Model Performance", "About the Project"):
    st.warning("No observations match the current filters. Widen the filters in the sidebar.")
    st.stop()

# ---------- pages ----------
if page == PAGES[0]:
    st.write("Input 1: a historical CSV (Date, City, PM2.5, PM10, NO₂, SO₂, CO, O₃, AQI) for trend analysis and model training. "
             "Input 2: pollutant values you enter to get a predicted AQI and category. Historical analysis and AQI *estimation* are shown separately; this app does not forecast the future.")
    c = st.columns(5)
    c[0].metric("Valid observations", f"{len(F):,}")
    c[1].metric("Cities", F["City"].nunique())
    c[2].metric("Average AQI", f"{F['AQI'].mean():.1f}")
    c[3].metric("Highest AQI", f"{F['AQI'].max():.0f}")
    c[4].metric("Lowest AQI", f"{F['AQI'].min():.0f}")
    st.plotly_chart(viz.category_bar(eda.category_distribution(F)), use_container_width=True)
    st.subheader("Pattern summary")
    for line in pa.pattern_summary(F, pa.city_ranking(F)):
        st.markdown(f"- {line}")

elif page == PAGES[1]:
    st.subheader("Cleaning report")
    st.json(REPORT)
    a, b = st.columns(2)
    a.subheader("Data dictionary"); a.dataframe(data_dictionary(F), use_container_width=True)
    b.subheader("Missing values (pollutants + AQI)"); b.dataframe(missing_summary(F, POLL + ["AQI"]), use_container_width=True)
    st.subheader("Potential outliers (IQR rule, flagged not removed)")
    st.dataframe(flag_outliers_iqr(F, POLL), use_container_width=True)
    st.subheader("Data preview"); st.dataframe(F.head(500), use_container_width=True)

elif page == PAGES[2]:
    st.subheader("AQI distribution and summary")
    st.dataframe(eda.aqi_summary(F).to_frame("AQI").T, use_container_width=True)
    st.plotly_chart(viz.aqi_histogram(F), use_container_width=True)
    st.subheader("Time trends")
    t1, t2, t3, t4 = st.tabs(["Daily", "Monthly", "Yearly", "Seasonal"])
    with t1: st.plotly_chart(viz.daily_line(eda.daily_trend(F)), use_container_width=True)
    with t2:
        st.plotly_chart(viz.monthly_line(eda.monthly_trend(F)), use_container_width=True)
        st.plotly_chart(viz.month_of_year_bar(eda.month_of_year(F)), use_container_width=True)
        st.caption("Months with the highest average AQI:"); st.dataframe(eda.unusual_periods(F, 8), use_container_width=True)
    with t3: st.plotly_chart(viz.yearly_bar(eda.yearly_trend(F)), use_container_width=True)
    with t4: st.plotly_chart(viz.season_bar(eda.seasonal_summary(F)), use_container_width=True)
    st.subheader("Pollutants")
    sel = st.selectbox("Pollutant", POLL)
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.pollutant_hist(F, sel), use_container_width=True)
    c2.plotly_chart(viz.scatter_vs_aqi(F, sel), use_container_width=True)
    st.plotly_chart(viz.pollutant_box(F, POLL), use_container_width=True)
    st.plotly_chart(viz.pollutant_means_bar(F, POLL), use_container_width=True)

elif page == PAGES[3]:
    rank = pa.city_ranking(F)
    st.plotly_chart(viz.city_bar(rank), use_container_width=True)
    top, bot = pa.most_least_polluted(rank)
    a, b = st.columns(2)
    a.subheader("Most polluted (mean AQI)"); a.dataframe(top, use_container_width=True, hide_index=True)
    b.subheader("Least polluted (mean AQI)"); b.dataframe(bot, use_container_width=True, hide_index=True)
    st.caption("Cities with fewer than 30 valid days are excluded from rankings. Observation periods differ by city.")
    st.plotly_chart(viz.city_month_heatmap(pa.city_month_heatmap_data(F)), use_container_width=True)
    st.subheader("Hotspots (mean AQI > 200 or ≥25% of days Poor or worse)")
    st.dataframe(pa.hotspots(rank), use_container_width=True, hide_index=True)
    st.subheader("Dominant pollutant per city")
    st.caption("Pollutant most strongly rank-correlated with AQI. Association, not causation.")
    st.dataframe(pa.dominant_pollutants(F, POLL), use_container_width=True, hide_index=True)

elif page == PAGES[4]:
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    corr = eda.correlation(F, POLL + ["AQI"], method)
    st.plotly_chart(viz.correlation_heatmap(corr, method.title()), use_container_width=True)
    st.subheader("Strongest relationships with AQI")
    s = corr["AQI"].drop("AQI").dropna().sort_values(ascending=False)
    for p, r in s.items():
        st.markdown(f"- **{p}**: r = {r:.2f} ({eda.describe_strength(r)})")
    hp = eda.high_correlation_pairs(corr.drop(index="AQI", columns="AQI"))
    st.subheader("Potential multicollinearity (|r| ≥ 0.8 between pollutants)")
    st.dataframe(hp, use_container_width=True, hide_index=True) if len(hp) else st.write("None at this threshold.")
    st.info("Correlation measures linear/monotonic association only. It does not prove that one variable causes another.")

elif page == PAGES[5]:
    st.subheader("Estimate AQI from pollutant readings")
    try:
        model, meta = load_artifacts()
    except ModelNotTrainedError as e:
        st.warning(str(e))
        if st.button("Train model now from the loaded dataset"):
            with st.spinner("Training…"):
                train_and_save(DF, POLL)
            st.rerun()
        st.stop()
    st.caption(f"Model: **{meta['best_model']}** · trained on the loaded historical dataset. "
               "Enter ALL six pollutants in the same units as that dataset, or fill them from a real record below.")
    feats = meta["features"]
    full = DF.dropna(subset=feats)
    with st.expander("Fill the form from a real record in the dataset (then compare with the actual AQI)"):
        if full.empty:
            st.info("No record in the loaded data has all six pollutants.")
        else:
            c1, c2 = st.columns(2)
            city = c1.selectbox("City", sorted(full["City"].unique()), key="pick_city")
            days = full.loc[full["City"] == city, "Date"].dt.date.tolist()[::-1]
            c2.selectbox("Date", days, key="pick_date")

            def _fill():
                r = full[(full["City"] == st.session_state["pick_city"]) &
                         (full["Date"].dt.date == st.session_state["pick_date"])].iloc[0]
                st.session_state["filled"] = {f: f"{r[f]:g}" for f in feats}
                for f in feats:
                    st.session_state[f"in_{f}"] = st.session_state["filled"][f]
                st.session_state["actual_aqi"] = float(r["AQI"])
            st.button("Fill form with this record", on_click=_fill)
    vals, cols = {}, st.columns(3)
    for i, f in enumerate(feats):
        r = meta["train_ranges"][f]
        vals[f] = cols[i % 3].text_input(f"{f} ({config.POLLUTANT_UNITS.get(f, '')})",
                                         placeholder=f"typical ≈ {r['median']:.1f}", key=f"in_{f}")
    with st.expander("Explain these inputs"):
        for f in feats:
            st.markdown(f"- **{f}** ({config.POLLUTANT_UNITS.get(f, '')}): {config.POLLUTANT_HELP.get(f, '')}")
        st.markdown("Values can come from manual entry, the historical CSV, or a real-time air-quality API / sensor. "
                    "The result is an **estimate of AQI from same-day pollutant values**, valid only if units match the "
                    "training data. It is not tomorrow's forecast.")
    if st.button("Predict AQI", type="primary"):
        res = predict_aqi(vals, model, meta, require_all=True)
        for e in res.get("errors", []): st.error(e)
        for w in res.get("warnings", []): st.warning(w)
        if res["ok"]:
            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated AQI", res["aqi"])
            color = config.AQI_COLORS.get(res["category"], "#555")
            m2.markdown(f"**Category**<br><span style='background:{color};color:#fff;padding:6px 14px;"
                        f"border-radius:4px;font-size:1.1rem'>{res['category']}</span>", unsafe_allow_html=True)
            if st.session_state.get("filled") == {k: v.strip() for k, v in vals.items()}:
                actual = st.session_state["actual_aqi"]
                m3.metric("Actual AQI (dataset)", f"{actual:.0f}", f"{res['aqi'] - actual:+.1f} model error", delta_color="off")
            st.caption(res["note"] + " Categories follow CPCB thresholds and are approximate.")

elif page == PAGES[6]:
    try:
        _, meta = load_artifacts()
    except ModelNotTrainedError as e:
        st.warning(str(e)); st.stop()
    res = pd.DataFrame(meta["results"])
    st.write(f"**Selected model:** {meta['best_model']} — {meta['selection_rule']}.")
    st.write(f"**Split:** {meta['split']['strategy']} · train {meta['split']['train'][:2]} · "
             f"validation {meta['split']['validation'][:2]} · test {meta['split']['test'][:2]}")
    st.dataframe(res.round(3), use_container_width=True, hide_index=True)
    st.plotly_chart(viz.metric_comparison(res), use_container_width=True)
    if config.PREDICTIONS_PATH.exists():
        preds = pd.read_csv(config.PREDICTIONS_PATH)
        a, b = st.columns(2)
        a.plotly_chart(viz.actual_vs_predicted(preds), use_container_width=True)
        b.plotly_chart(viz.residual_plot(preds), use_container_width=True)
        st.plotly_chart(viz.residual_hist(preds), use_container_width=True)
    st.plotly_chart(viz.importance_bar(meta["rf_feature_importance"]), use_container_width=True)

elif page == PAGES[7]:
    st.markdown("""
**Objective A — Pollution pattern analysis:** daily/monthly/yearly/seasonal trends, city comparison, pollutant relationships.
**Objective B — AQI prediction:** Linear Regression vs Random Forest, chronological split, validation-based model selection.

**Limitations.** The model estimates AQI from same-day pollutant values; it is not a time-series forecast.
Pollutant values outside the training range are unreliable. City coverage and time periods differ in the data.
Correlation is not causation. Dataset: [Kaggle – Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).
""")
