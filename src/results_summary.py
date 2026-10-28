"""Generate reports/results_summary.md from the REAL dataset and the trained model.

Use it to fill the Results section of the report / README / slides with actual numbers
(nothing is hard-coded here, nothing is invented).

Run (after `python -m src.model_training`):   python -m src.results_summary
"""
from __future__ import annotations
import pandas as pd
import config
from src import eda, pollution_analysis as pa
from src.data_loader import load_csv
from src.preprocessing import clean, flag_outliers_iqr, missing_summary
from src.prediction import load_artifacts

OUTPUT_PATH = config.REPORTS_DIR / "results_summary.md"


def md_table(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> markdown table (avoids the optional 'tabulate' dependency)."""
    if df.empty:
        return "_(none)_"
    d = df.copy()
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def build_summary(data: pd.DataFrame, report: dict, meta: dict) -> str:
    pollutants = report["pollutants"]
    out: list[str] = ["# Results summary (auto-generated from the real dataset)\n"]

    out.append("## 1. Dataset and cleaning")
    out.append(f"- Raw rows: **{report['rows_raw']}**; invalid dates dropped: {report['invalid_dates']}; "
               f"duplicate (City, Date) removed: {report['duplicates_removed']}")
    out.append(f"- Negative values set to missing: {report['negative_values_set_nan'] or 'none'}")
    out.append(f"- Rows with missing AQI dropped: {report['missing_aqi_dropped']}")
    out.append(f"- **Clean rows: {report['rows_clean']}**, cities: {report['cities']}, "
               f"period: {report['date_min']} to {report['date_max']}")
    out.append("\nMissing values per pollutant (clean data):\n")
    out.append(md_table(missing_summary(data, pollutants).reset_index().rename(columns={"index": "pollutant"})))
    out.append("\nOutliers flagged (IQR rule, not removed):\n")
    out.append(md_table(flag_outliers_iqr(data, pollutants + ["AQI"])))

    out.append("\n## 2. AQI overview")
    out.append(md_table(eda.aqi_summary(data).rename("value").reset_index().rename(columns={"index": "statistic"})))
    out.append("\nAQI category distribution (CPCB bands):\n")
    out.append(md_table(eda.category_distribution(data)))

    out.append("\n## 3. Pollution patterns")
    for line in pa.pattern_summary(data, pa.city_ranking(data)):
        out.append(f"- {line}")
    out.append("\nYearly (note: city coverage differs by year):\n")
    out.append(md_table(eda.yearly_trend(data)))
    out.append("\nSeasonal:\n")
    out.append(md_table(eda.seasonal_summary(data)))
    rank = pa.city_ranking(data)
    top, bottom = pa.most_least_polluted(rank)
    cols = ["rank", "City", "mean_aqi", "median_aqi", "max_aqi", "observations", "pct_days_poor_or_worse"]
    out.append("\nMost polluted cities (mean AQI):\n")
    out.append(md_table(top[cols]))
    out.append("\nLeast polluted cities (mean AQI):\n")
    out.append(md_table(bottom[cols]))
    out.append("\nHotspots (mean AQI > 200 or >=25% of days Poor or worse):\n")
    out.append(md_table(pa.hotspots(rank)[cols]))

    out.append("\n## 4. Correlation with AQI (Pearson)")
    corr = eda.correlation(data, pollutants + ["AQI"])
    s = corr["AQI"].drop("AQI").dropna().sort_values(ascending=False)
    out.append(md_table(pd.DataFrame({"pollutant": s.index, "r": s.round(3).values,
                                      "strength": [eda.describe_strength(r) for r in s.values]})))
    hp = eda.high_correlation_pairs(corr.drop(index="AQI", columns="AQI"))
    out.append("\nPollutant pairs with |r| >= 0.8 (multicollinearity):\n")
    out.append(md_table(hp))

    out.append("\n## 5. Model results")
    sp = meta["split"]
    out.append(f"- Split ({sp['strategy']}): train {sp['train'][0]} to {sp['train'][1]} (n={sp['train'][2]}); "
               f"validation {sp['validation'][0]} to {sp['validation'][1]} (n={sp['validation'][2]}); "
               f"test {sp['test'][0]} to {sp['test'][1]} (n={sp['test'][2]})")
    out.append(f"- Features used: {', '.join(meta['features'])}")
    out.append(f"- **Selected model: {meta['best_model']}** ({meta['selection_rule']})\n")
    out.append(md_table(pd.DataFrame(meta["results"]).round(3)))
    imp = pd.Series(meta["rf_feature_importance"]).sort_values(ascending=False)
    out.append("\nRandom Forest feature importance:\n")
    out.append(md_table(pd.DataFrame({"feature": imp.index, "importance": imp.round(4).values})))
    return "\n".join(out) + "\n"


def main() -> None:
    data, report = clean(load_csv())
    _, meta = load_artifacts()
    config.REPORTS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(build_summary(data, report, meta), encoding="utf-8")
    print("Written:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
