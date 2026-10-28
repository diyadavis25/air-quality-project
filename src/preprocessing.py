"""Cleaning and feature preparation. Documented decisions:
- Duplicate rows (same City+Date) are dropped (first kept).
- Pollutant concentrations must be >= 0; negative readings are sensor errors -> set to NaN.
- Rows with missing/invalid AQI are DROPPED from analysis and modelling (AQI is never invented).
- Pollutant NaNs are kept here; imputation happens inside the ML Pipeline (fit on training data only).
- Outliers are FLAGGED (IQR rule), not removed, because extreme smog episodes are real.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config
from src.data_loader import validate_columns


def aqi_category(value: float) -> str | float:
    """CPCB category for an AQI value (NaN stays NaN)."""
    if pd.isna(value):
        return np.nan
    for upper, label in config.AQI_BINS:
        if value <= upper:
            return label
    return np.nan


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pollutants = validate_columns(df)
    report: dict = {"rows_raw": int(len(df))}
    out = df.copy()

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    report["invalid_dates"] = int(out["Date"].isna().sum())
    out = out.dropna(subset=["Date"])

    before = len(out)
    out = out.drop_duplicates(subset=["City", "Date"], keep="first")
    report["duplicates_removed"] = int(before - len(out))

    for col in pollutants + ["AQI"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    neg = {c: int((out[c] < 0).sum()) for c in pollutants + ["AQI"] if (out[c] < 0).any()}
    report["negative_values_set_nan"] = neg
    for c in neg:
        out.loc[out[c] < 0, c] = np.nan

    report["missing_aqi_dropped"] = int(out["AQI"].isna().sum())
    out = out.dropna(subset=["AQI"])
    out = out[out["AQI"] <= 500 * 2]  # guard against absurd values; CPCB scale tops out at 500
    out = out.sort_values(["Date", "City"]).reset_index(drop=True)

    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["MonthName"] = out["Date"].dt.strftime("%b")
    out["Season"] = out["Month"].map(config.SEASONS)
    # Always derive the category from the numeric AQI so it is consistent with documented thresholds
    out["AQI_Category"] = pd.Categorical(out["AQI"].map(aqi_category), categories=config.AQI_ORDER, ordered=True)

    report.update(rows_clean=int(len(out)), cities=int(out["City"].nunique()),
                  date_min=str(out["Date"].min().date()) if len(out) else None,
                  date_max=str(out["Date"].max().date()) if len(out) else None,
                  pollutants=pollutants)
    return out, report


def missing_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    s = pd.DataFrame({"missing": df[cols].isna().sum(), "missing_%": (df[cols].isna().mean() * 100).round(2)})
    return s.sort_values("missing_%", ascending=False)


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes],
                         "non_null": df.notna().sum().values,
                         "unique": [df[c].nunique() for c in df.columns]})


def flag_outliers_iqr(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    """Count of IQR outliers per column (flag only; nothing removed)."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        n = int(((s < lo) | (s > hi)).sum())
        rows.append({"pollutant": c, "lower_fence": round(lo, 2), "upper_fence": round(hi, 2),
                     "outliers": n, "outlier_%": round(100 * n / len(s), 2)})
    return pd.DataFrame(rows)


def filter_data(df, cities=None, date_range=None, years=None, months=None) -> pd.DataFrame:
    out = df
    if cities:
        out = out[out["City"].isin(cities)]
    if date_range and len(date_range) == 2:
        out = out[(out["Date"] >= pd.Timestamp(date_range[0])) & (out["Date"] <= pd.Timestamp(date_range[1]))]
    if years:
        out = out[out["Year"].isin(years)]
    if months:
        out = out[out["Month"].isin(months)]
    return out
