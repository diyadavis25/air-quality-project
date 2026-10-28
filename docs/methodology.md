# Methodology

Every step below matches what the code in this repository actually does (file names in brackets).

## 1. Problem definition
| | |
|---|---|
| **Objective A** | Analyse pollution patterns: trends over time, seasonal behaviour, city comparison, hotspots, pollutant relationships. |
| **Objective B** | Estimate a day's AQI from that day's pollutant readings using regression. |
| **Not in scope** | Forecasting future AQI. No time-series model is implemented; the model uses same-day pollutants only. |

## 2. Data
- **Source:** Kaggle, *Air Quality Data in India* (rohanrao), file `city_day.csv`, saved as `data/air_quality.csv`.
- **Columns:** City, Date, PM2.5, PM10, NO2, SO2, CO, O3, AQI (other columns in the file are ignored).
- **Loading** (`src/data_loader.py`): column names are matched case-insensitively (aliases such as `pm25` → `PM2.5`); missing file, empty file or missing `Date`/`AQI` columns give a readable error instead of a crash.

## 3. Cleaning (`src/preprocessing.py → clean`)
| Step | Rule | Reason |
|---|---|---|
| 1 | Unparseable dates are dropped | A row without a date cannot be placed in any trend. |
| 2 | Duplicate (City, Date) rows removed, first kept | One observation per city-day. |
| 3 | Negative pollutant/AQI values set to missing | Concentrations cannot be negative; these are sensor errors. |
| 4 | Rows with missing AQI are **dropped** | AQI is the target; inventing it would contaminate both analysis and model. |
| 5 | AQI > 1000 dropped | Guard against absurd entries (CPCB scale ends at 500). |
| 6 | Outliers **flagged** with the IQR rule (1.5×IQR), **not removed** | Extreme smog days are real events; deleting them would hide the problem being studied. |
| 7 | Pollutant missing values **kept** at this stage | Imputation happens later inside the ML pipeline so it is learnt from training data only. |

Derived columns: Year, Month, MonthName, Season, AQI_Category.

**Seasons (India):** Winter Dec–Feb, Summer Mar–May, Monsoon Jun–Sep, Post-monsoon Oct–Nov.

**AQI categories (CPCB):** 0–50 Good, 51–100 Satisfactory, 101–200 Moderate, 201–300 Poor, 301–400 Very Poor, 401+ Severe. The category is always recomputed from the numeric AQI, so it is consistent with these thresholds (the dataset's own `AQI_Bucket` column is not used).

## 4. Exploratory analysis (`src/eda.py`, `src/pollution_analysis.py`)
- AQI summary statistics and category distribution.
- Daily, monthly, month-of-year, yearly and seasonal mean AQI.
- Months with the highest mean AQI (unusual periods).
- City ranking by **mean AQI**; cities with fewer than 30 valid days are excluded from the ranking for fairness.
- **Hotspot rule:** city mean AQI > 200, **or** at least 25 % of its days with AQI > 200 (Poor or worse).
- **Dominant pollutant per city:** pollutant with the highest Spearman correlation with AQI in that city (≥ 30 paired observations). This is association, not proof of cause.
- Pearson and Spearman correlation matrices; pollutant pairs with |r| ≥ 0.8 are reported as potential multicollinearity.

## 5. Model building (`src/model_training.py`)
**Target:** AQI. **Features:** pollutants present in at least 50 % of valid rows.
**Excluded on purpose:** `AQI_Bucket` (it is computed from AQI, so using it would be *target leakage*); `City` and `Date` (the aim is pollutant → AQI estimation).

**Models compared**
1. Baseline — always predicts the training mean (a model must beat this to be useful).
2. Linear Regression — median imputation, standardisation, linear fit.
3. Random Forest — median imputation, 200 trees, `min_samples_leaf=2`, `random_state=42`.

**Splitting:** chronological 70 % / 15 % / 15 % (train / validation / test). The data are time-ordered, and a random split would let the model train on days after the ones it is tested on.

**Model selection:** lowest **validation RMSE** between Linear Regression and Random Forest. The **test set is used once** for the final report, so the reported test score is not tuned on.

**Leakage control:** imputation and scaling live inside a scikit-learn `Pipeline`, so they are fitted on training rows only.

**Metrics:** MAE (average error in AQI points), MSE, RMSE (penalises large errors), R² (share of variance explained; can be negative if worse than the mean baseline).

**Saved artefacts:** `models/aqi_model.joblib`, `models/model_metadata.json` (features, split dates, metrics, importances, training ranges), `reports/test_predictions.csv`.

## 6. Prediction (`src/prediction.py`)
- Inputs must be numeric and non-negative; at least one pollutant is required.
- Blank pollutants are filled with the training median (the user is told which).
- A warning is shown if a value is more than 1.5× the training maximum.
- Output is clipped to 0–500 and labelled with a CPCB category, with a note that it is an estimate, not an official AQI or forecast.

## 7. Dashboard (`app.py`)
Streamlit app with eight pages (Overview, Dataset Explorer, EDA, Pollution Patterns, Correlation, AQI Prediction, Model Performance, About) and sidebar filters for city, date range, year and month.

## 8. Testing (`tests/`)
Automated checks for: loader errors and aliases, cleaning rules, AQI categories, outlier flagging (nothing removed), filters, chronological split, all four metrics present, baseline beaten, validation-based selection, no leakage column among features, prediction output and input validation. Model tests use a **synthetic** frame only to verify the code mechanics; they say nothing about real results.

## 9. Known limitations
1. Same-day estimation only, not forecasting.
2. AQI in the dataset is itself computed by formula from pollutants, so a high R² partly reflects that known relationship; the model should not be described as "discovering" the AQI rule.
3. Cities have different observation periods, so yearly/overall comparisons are not like-for-like.
4. Median imputation is simple; heavy missingness in a pollutant weakens the features that depend on it.
5. Random Forests cannot predict beyond the range seen in training; very high inputs are unreliable.
6. Correlation does not show causation.
7. No hyper-parameter tuning or cross-validation beyond the single chronological validation block.
