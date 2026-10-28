import numpy as np
import pandas as pd
import config
from src import model_training as mt
from src.preprocessing import clean
from src.results_summary import build_summary, md_table


def _synthetic():
    rng = np.random.default_rng(0)
    n = 400
    pm25, pm10 = rng.gamma(4, 20, n), rng.gamma(4, 35, n)
    df = pd.DataFrame({"City": rng.choice(["A", "B", "C"], n), "Date": pd.date_range("2018-01-01", periods=n),
                       "PM2.5": pm25, "PM10": pm10, "AQI": 0.8 * pm25 + 0.3 * pm10 + rng.normal(0, 5, n)})
    return clean(df)


def test_md_table_handles_empty_and_rows():
    assert md_table(pd.DataFrame()) == "_(none)_"
    t = md_table(pd.DataFrame({"a": [1], "b": ["x"]}))
    assert t.splitlines()[0] == "| a | b |" and "| 1 | x |" in t


def test_build_summary_contains_all_sections(tmp_path):
    data, rep = _synthetic()
    config.MODELS_DIR = config.REPORTS_DIR = tmp_path
    config.MODEL_PATH, config.METADATA_PATH = tmp_path / "m.joblib", tmp_path / "m.json"
    config.PREDICTIONS_PATH = tmp_path / "p.csv"
    meta = mt.train_and_save(data, rep["pollutants"])["metadata"]
    text = build_summary(data, rep, meta)
    for heading in ("## 1. Dataset", "## 2. AQI overview", "## 3. Pollution patterns",
                    "## 4. Correlation", "## 5. Model results"):
        assert heading in text
    assert meta["best_model"] in text
