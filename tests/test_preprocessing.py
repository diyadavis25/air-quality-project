"""Unit tests use tiny SYNTHETIC frames purely to check code logic; they are not project data."""
import numpy as np
import pandas as pd
import pytest
from src.data_loader import DatasetError, load_csv, validate_columns, standardize_columns
from src.preprocessing import aqi_category, clean, flag_outliers_iqr, filter_data


def make_df():
    return pd.DataFrame({
        "City": ["A", "A", "A", "B", "B"],
        "Date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-01", "bad-date"],
        "PM2.5": [10, 10, -5, 50, 20], "PM10": ["20", "20", "abc", "80", "30"],
        "AQI": [50, 50, 120, np.nan, 90]})


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(DatasetError, match="kaggle.com"):
        load_csv(tmp_path / "nope.csv")


def test_missing_required_columns():
    with pytest.raises(DatasetError):
        validate_columns(pd.DataFrame({"City": [1], "Date": [1]}))


def test_alias_standardization():
    df = standardize_columns(pd.DataFrame(columns=["pm25", "aqi", "datetime"]))
    assert {"PM2.5", "AQI", "Date"} <= set(df.columns)


def test_clean_rules():
    out, rep = clean(make_df())
    assert rep["invalid_dates"] == 1
    assert rep["duplicates_removed"] == 1
    assert rep["negative_values_set_nan"] == {"PM2.5": 1}
    assert rep["missing_aqi_dropped"] == 1          # AQI never invented
    assert len(out) == 2 and out["AQI"].notna().all()
    assert out["PM10"].isna().sum() == 1             # 'abc' coerced to NaN, row kept
    assert {"Year", "Month", "Season", "AQI_Category"} <= set(out.columns)
    assert out["Date"].is_monotonic_increasing


@pytest.mark.parametrize("v,cat", [(50, "Good"), (51, "Satisfactory"), (150, "Moderate"),
                                    (250, "Poor"), (350, "Very Poor"), (450, "Severe")])
def test_aqi_category(v, cat):
    assert aqi_category(v) == cat


def test_outlier_flagging_does_not_remove():
    df = pd.DataFrame({"x": [1, 2, 2, 3, 2, 100]})
    r = flag_outliers_iqr(df, ["x"])
    assert r.loc[0, "outliers"] == 1 and len(df) == 6


def test_filter_data():
    out, _ = clean(pd.DataFrame({"City": ["A", "B"], "Date": ["2020-01-01", "2021-02-01"], "PM10": [1, 2], "AQI": [10, 20]}))
    assert len(filter_data(out, cities=["A"])) == 1
    assert len(filter_data(out, years=[2021])) == 1
    assert len(filter_data(out, date_range=("2021-01-01", "2021-12-31"))) == 1
