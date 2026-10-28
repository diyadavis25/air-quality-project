"""AQI regression: Linear Regression vs Random Forest (plus a mean-baseline).

Design decisions
- Target: the dataset's AQI column. Features: pollutant concentrations only (AQI_Bucket is excluded: it is
  derived from AQI -> target leakage). Date/City are not used, so predictions are for pollutant->AQI estimation,
  NOT future forecasting.
- Split: CHRONOLOGICAL 70/15/15 (train/validation/test) because rows are time-ordered.
  Model selection uses VALIDATION RMSE; the TEST set is touched once for the final report.
- Preprocessing (median imputation, scaling for LR) lives inside a sklearn Pipeline, so it is fitted
  on training data only.
Run:  python -m src.model_training
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import config


def select_features(df: pd.DataFrame, pollutants: list[str]) -> list[str]:
    feats = [p for p in pollutants if df[p].notna().mean() >= config.MIN_FEATURE_COVERAGE]
    if not feats:
        raise ValueError("No pollutant has enough non-missing data to be used as a model feature.")
    return feats


def chronological_split(df: pd.DataFrame):
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    if n < 100:
        raise ValueError(f"Only {n} rows available; at least 100 are needed to train and evaluate a model.")
    i1, i2 = int(n * config.TRAIN_FRAC), int(n * (config.TRAIN_FRAC + config.VAL_FRAC))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def build_models() -> dict[str, Pipeline]:
    imp = lambda: SimpleImputer(strategy="median", keep_empty_features=True)  # noqa: E731
    return {
        "Baseline (mean)": Pipeline([("imputer", imp()), ("model", DummyRegressor(strategy="mean"))]),
        "Linear Regression": Pipeline([("imputer", imp()), ("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": Pipeline([("imputer", imp()),
                                   ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                                   n_jobs=-1, random_state=config.RANDOM_SEED))]),
    }


def metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)), "R2": float(r2_score(y_true, y_pred))}


def train_and_save(df: pd.DataFrame, pollutants: list[str], save: bool = True) -> dict:
    feats = select_features(df, pollutants)
    train, val, test = chronological_split(df)
    Xtr, ytr = train[feats], train["AQI"]
    Xva, yva = val[feats], val["AQI"]
    Xte, yte = test[feats], test["AQI"]

    results, fitted = [], {}
    for name, pipe in build_models().items():
        pipe.fit(Xtr, ytr)
        fitted[name] = pipe
        v, t = metrics(yva, pipe.predict(Xva)), metrics(yte, pipe.predict(Xte))
        results.append({"model": name, **{f"val_{k}": x for k, x in v.items()}, **{f"test_{k}": x for k, x in t.items()}})
    res = pd.DataFrame(results)

    candidates = res[res["model"] != "Baseline (mean)"]
    best_name = candidates.sort_values("val_RMSE").iloc[0]["model"]  # chosen on VALIDATION only
    best = fitted[best_name]

    rf = fitted["Random Forest"].named_steps["model"]
    lr = fitted["Linear Regression"].named_steps["model"]
    meta = {
        "features": feats,
        "best_model": best_name,
        "selection_rule": "lowest validation RMSE among Linear Regression and Random Forest",
        "split": {"strategy": "chronological 70/15/15",
                  "train": [str(train["Date"].min().date()), str(train["Date"].max().date()), len(train)],
                  "validation": [str(val["Date"].min().date()), str(val["Date"].max().date()), len(val)],
                  "test": [str(test["Date"].min().date()), str(test["Date"].max().date()), len(test)]},
        "results": res.round(4).to_dict(orient="records"),
        "rf_feature_importance": dict(zip(feats, map(float, rf.feature_importances_))),
        "lr_standardized_coefficients": dict(zip(feats, map(float, lr.coef_))),
        "train_ranges": {f: {"min": float(Xtr[f].min()), "p99": float(Xtr[f].quantile(0.99)), "max": float(Xtr[f].max()),
                             "median": float(Xtr[f].median())} for f in feats},
        "aqi_train_range": [float(ytr.min()), float(ytr.max())],
        "seed": config.RANDOM_SEED,
    }
    preds = test[["Date", "City"]].copy()
    preds["actual"] = yte.values
    preds["predicted"] = best.predict(Xte)
    preds["residual"] = preds["actual"] - preds["predicted"]

    if save:
        config.MODELS_DIR.mkdir(exist_ok=True)
        config.REPORTS_DIR.mkdir(exist_ok=True)
        joblib.dump(best, config.MODEL_PATH)
        config.METADATA_PATH.write_text(json.dumps(meta, indent=2))
        preds.to_csv(config.PREDICTIONS_PATH, index=False)
    return {"metadata": meta, "results": res, "predictions": preds}


if __name__ == "__main__":
    from src.data_loader import load_csv
    from src.preprocessing import clean
    data, rep = clean(load_csv())
    out = train_and_save(data, rep["pollutants"])
    print(out["results"].round(3).to_string(index=False))
    print("Best model (by validation RMSE):", out["metadata"]["best_model"])
    print("Saved to", config.MODEL_PATH)
