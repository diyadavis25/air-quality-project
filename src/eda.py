"""Pure-pandas EDA helpers (no plotting)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import config


def aqi_summary(df: pd.DataFrame) -> pd.Series:
    s = df["AQI"]
    return pd.Series({"count": int(s.count()), "mean": s.mean(), "median": s.median(),
                      "min": s.min(), "max": s.max(), "std": s.std()}).round(2)


def category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    c = df["AQI_Category"].value_counts().reindex(config.AQI_ORDER).fillna(0).astype(int)
    return pd.DataFrame({"category": c.index, "days": c.values, "percent": (100 * c.values / max(c.sum(), 1)).round(2)})


def daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Mean AQI per calendar day across the selected cities, sorted by date."""
    return df.groupby("Date", as_index=False)["AQI"].mean().sort_values("Date")


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df["Date"].dt.to_period("M"))["AQI"].mean().reset_index()
    g["Date"] = g["Date"].dt.to_timestamp()
    return g.rename(columns={"Date": "Month"})


def month_of_year(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Month")["AQI"].mean().reindex(range(1, 13)).reset_index()
    g["MonthName"] = pd.to_datetime(g["Month"], format="%m").dt.strftime("%b")
    return g


def yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Year").agg(mean_aqi=("AQI", "mean"), observations=("AQI", "size"),
                               cities=("City", "nunique")).reset_index()
    return g.round(2)


def seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
    g = df.groupby("Season")["AQI"].agg(["mean", "median", "size"]).reindex(order).round(2)
    return g.reset_index()


def unusual_periods(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Months with the highest mean AQI (flag of unusually polluted periods)."""
    m = monthly_trend(df)
    out = m.nlargest(top, "AQI")
    out["AQI"] = out["AQI"].round(2)
    return out


def correlation(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> pd.DataFrame:
    return df[cols].corr(method=method)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Pollutant pairs with |r| >= threshold -> potential multicollinearity."""
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"feature_1": a, "feature_2": b, "r": round(r, 3)})
    return pd.DataFrame(rows, columns=["feature_1", "feature_2", "r"])


def describe_strength(r: float) -> str:
    a = abs(r)
    s = "very strong" if a >= 0.8 else "strong" if a >= 0.6 else "moderate" if a >= 0.4 else "weak" if a >= 0.2 else "very weak"
    return f"{s} {'positive' if r > 0 else 'negative'}"
