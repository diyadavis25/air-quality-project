import numpy as np
import pandas as pd
import pytest
import config
from src import model_training as mt
from src.prediction import predict_aqi, validate_inputs
from src.preprocessing import clean


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train on a SYNTHETIC frame (test-only) to verify the pipeline mechanics."""
    rng = np.random.default_rng(0)
    n = 600
    pm25, pm10, no2 = rng.gamma(4, 20, n), rng.gamma(4, 35, n), rng.gamma(3, 10, n)
    df = pd.DataFrame({"City": rng.choice(["A", "B"], n), "Date": pd.date_range("2018-01-01", periods=n),
                       "PM2.5": pm25, "PM10": pm10, "NO2": no2,
                       "AQI": 0.8 * pm25 + 0.3 * pm10 + 0.5 * no2 + rng.normal(0, 5, n)})
    df.loc[::7, "NO2"] = np.nan
    data, rep = clean(df)
    d = tmp_path_factory.mktemp("art")
    config.MODEL_PATH, config.METADATA_PATH, config.PREDICTIONS_PATH = d / "m.joblib", d / "m.json", d / "p.csv"
    config.MODELS_DIR = config.REPORTS_DIR = d
    return mt.train_and_save(data, rep["pollutants"])


def test_split_is_chronological(trained):
    s = trained["metadata"]["split"]
    assert s["train"][1] <= s["validation"][0] and s["validation"][1] <= s["test"][0]


def test_all_metrics_present_and_baseline_beaten(trained):
    res = trained["results"].set_index("model")
    for m in ("MAE", "MSE", "RMSE", "R2"):
        assert f"test_{m}" in res.columns
    assert res.loc["Linear Regression", "test_RMSE"] < res.loc["Baseline (mean)", "test_RMSE"]


def test_best_model_chosen_on_validation(trained):
    res = trained["results"].set_index("model").drop("Baseline (mean)")
    assert trained["metadata"]["best_model"] == res["val_RMSE"].idxmin()


def test_leakage_columns_not_features(trained):
    f = trained["metadata"]["features"]
    assert "AQI" not in f and "AQI_Bucket" not in f and "Date" not in f


def test_prediction_output(trained):
    meta = trained["metadata"]
    out = predict_aqi({"PM2.5": "80", "PM10": 150}, meta=meta, model=__import__("joblib").load(config.MODEL_PATH))
    assert out["ok"] and 0 <= out["aqi"] <= 500 and out["category"]
    assert any("NO2" in w for w in out["warnings"])  # missing feature -> median fill warning


def test_input_validation(trained):
    meta = trained["metadata"]
    assert validate_inputs({"PM2.5": "abc"}, meta)[1]
    assert validate_inputs({"PM2.5": -4}, meta)[1]
    assert validate_inputs({}, meta)[1]
    X, err, warn = validate_inputs({"PM2.5": 1e6}, meta)
    assert not err and warn and list(X.columns) == meta["features"]
