"""City-level pollution pattern analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd


def city_ranking(df: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """Rank cities by MEAN AQI (cities with fewer than `min_obs` valid days excluded for fairness)."""
    g = df.groupby("City").agg(mean_aqi=("AQI", "mean"), median_aqi=("AQI", "median"),
                               max_aqi=("AQI", "max"), observations=("AQI", "size"),
                               pct_days_poor_or_worse=("AQI", lambda s: 100 * (s > 200).mean())).reset_index()
    g = g[g["observations"] >= min_obs].sort_values("mean_aqi", ascending=False).reset_index(drop=True)
    g.insert(0, "rank", g.index + 1)
    return g.round(2)


def most_least_polluted(ranking: pd.DataFrame, n: int = 5):
    return ranking.head(n), ranking.tail(n).sort_values("mean_aqi")


def city_month_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Mean AQI, rows = city, columns = month number (1-12)."""
    return df.pivot_table(index="City", columns="Month", values="AQI", aggfunc="mean").reindex(columns=range(1, 13))


def hotspots(ranking: pd.DataFrame, threshold: float = 200) -> pd.DataFrame:
    """Cities whose mean AQI is in the 'Moderate-to-Poor' band or worse (> threshold) or share of Poor+ days is high."""
    return ranking[(ranking["mean_aqi"] > threshold) | (ranking["pct_days_poor_or_worse"] >= 25)]


def dominant_pollutants(df: pd.DataFrame, pollutants: list[str]) -> pd.DataFrame:
    """Per city, the pollutant whose Spearman correlation with AQI is highest.
    NOTE: association only, not proof that the pollutant causes the AQI value."""
    rows = []
    for city, g in df.groupby("City"):
        best, best_r = None, -np.inf
        for p in pollutants:
            sub = g[[p, "AQI"]].dropna()
            if len(sub) >= 30 and sub[p].nunique() > 1:
                r = sub[p].corr(sub["AQI"], method="spearman")
                if pd.notna(r) and r > best_r:
                    best, best_r = p, r
        if best:
            rows.append({"City": city, "dominant_pollutant (by correlation)": best, "spearman_r": round(best_r, 3)})
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)


def pattern_summary(df: pd.DataFrame, ranking: pd.DataFrame) -> list[str]:
    """Plain-language insights computed from the (filtered) data only."""
    out = []
    if df.empty or ranking.empty:
        return ["Not enough data for the current filters."]
    top, bot = ranking.iloc[0], ranking.iloc[-1]
    out.append(f"Highest mean AQI: {top['City']} ({top['mean_aqi']}); lowest: {bot['City']} ({bot['mean_aqi']}).")
    mon = df.groupby("Month")["AQI"].mean()
    if len(mon):
        out.append(f"Most polluted month of year (average): {pd.to_datetime(int(mon.idxmax()), format='%m').strftime('%B')} ({mon.max():.1f}); "
                   f"cleanest: {pd.to_datetime(int(mon.idxmin()), format='%m').strftime('%B')} ({mon.min():.1f}).")
    sea = df.groupby("Season")["AQI"].mean()
    if len(sea):
        out.append(f"Season with highest average AQI: {sea.idxmax()} ({sea.max():.1f}); lowest: {sea.idxmin()} ({sea.min():.1f}).")
    yr = df.groupby("Year")["AQI"].mean()
    if len(yr) > 1:
        out.append(f"Yearly mean AQI moved from {yr.iloc[0]:.1f} ({yr.index[0]}) to {yr.iloc[-1]:.1f} ({yr.index[-1]}). "
                   "Coverage of cities differs by year, so compare with care.")
    return out
