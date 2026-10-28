"""Loading and validating the raw CSV (Kaggle: rohanrao/air-quality-data-in-india, file city_day.csv)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import config

DOWNLOAD_HELP = (
    "Dataset not found at: {path}\n"
    "1. Open https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india\n"
    "2. Download the dataset and extract it.\n"
    "3. Copy 'city_day.csv' into the data/ folder and rename it to 'air_quality.csv'\n"
    "   (or set the AQ_DATA_PATH environment variable to the CSV location)."
)

# Lower-case alias -> canonical name (handles minor naming differences between dataset versions)
ALIASES = {"city": "City", "date": "Date", "datetime": "Date", "aqi": "AQI", "aqi_bucket": "AQI_Bucket",
           "pm2.5": "PM2.5", "pm25": "PM2.5", "pm10": "PM10", "no2": "NO2",
           "so2": "SO2", "co": "CO", "o3": "O3"}


class DatasetError(Exception):
    """Raised for a missing or incompatible dataset (message is user-friendly)."""


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: ALIASES[c.strip().lower()] for c in df.columns if c.strip().lower() in ALIASES}
    return df.rename(columns=rename)


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return the pollutant columns found. Raise DatasetError if essentials are missing."""
    missing = [c for c in ("Date", "AQI") if c not in df.columns]
    if missing:
        raise DatasetError(f"Required column(s) missing: {missing}. Columns found: {list(df.columns)}")
    pollutants = [c for c in config.POLLUTANT_CANDIDATES if c in df.columns]
    if not pollutants:
        raise DatasetError("No known pollutant columns (PM2.5, PM10, NO2, ...) were found.")
    return pollutants


def load_csv(source=None) -> pd.DataFrame:
    """Load from a path (default config.DATA_PATH) or a file-like object (Streamlit upload)."""
    source = source if source is not None else config.DATA_PATH
    if isinstance(source, (str, Path)):
        if not Path(source).exists():
            raise DatasetError(DOWNLOAD_HELP.format(path=source))
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Could not read the CSV file: {exc}") from exc
    if df.empty:
        raise DatasetError("The CSV file is empty.")
    df = standardize_columns(df)
    validate_columns(df)
    if "City" not in df.columns:
        df["City"] = "Unknown"
    return df
