"""Load the trained pipeline and predict AQI from user-entered pollutant values."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import config
from src.preprocessing import aqi_category


class ModelNotTrainedError(Exception):
    pass


def load_artifacts():
    if not (config.MODEL_PATH.exists() and config.METADATA_PATH.exists()):
        raise ModelNotTrainedError("No trained model found. Run:  python -m src.model_training")
    meta = json.loads(config.METADATA_PATH.read_text())
    if not set(meta["features"]) <= set(config.POLLUTANT_CANDIDATES):
        raise ModelNotTrainedError("The saved model was trained on a different pollutant set. "
                                   "Retrain it:  python -m src.model_training")
    return joblib.load(config.MODEL_PATH), meta


def validate_inputs(values: dict, meta: dict, require_all: bool = False) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (1-row DataFrame in training feature order, errors, warnings).
    Blank/None inputs are allowed (median-imputed by the pipeline) but at least one value is required."""
    feats = meta["features"]
    errors, warnings, row = [], [], {}
    provided = 0
    for f in feats:
        v = values.get(f)
        if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and np.isnan(v)):
            row[f] = np.nan
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            errors.append(f"{f}: '{v}' is not a number.")
            continue
        if not np.isfinite(x) or x < 0:
            errors.append(f"{f}: must be a finite, non-negative number.")
            continue
        rng = meta["train_ranges"][f]
        if x > rng["max"] * 1.5:
            warnings.append(f"{f}={x:g} is far above the training maximum ({rng['max']:g}); the estimate is unreliable.")
        row[f] = x
        provided += 1
    if provided == 0 and not errors:
        errors.append("Enter at least one pollutant value.")
    missing = [f for f in feats if np.isnan(row.get(f, np.nan))]
    if require_all and missing and not errors:
        errors.append(f"Enter a value for every pollutant. Missing: {', '.join(missing)}.")
    elif missing and provided:
        warnings.append(f"Not provided (filled with training median): {', '.join(missing)}.")
    return pd.DataFrame([row], columns=feats), errors, warnings


def predict_aqi(values: dict, model=None, meta=None, require_all: bool = False) -> dict:
    if model is None or meta is None:
        model, meta = load_artifacts()
    X, errors, warnings = validate_inputs(values, meta, require_all)
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}
    pred = float(np.clip(model.predict(X)[0], 0, 500))
    return {"ok": True, "aqi": round(pred, 1), "category": aqi_category(pred), "warnings": warnings,
            "note": "Model estimate of AQI from same-day pollutant readings. Not an official AQI and not a forecast."}
