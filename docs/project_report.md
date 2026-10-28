# Air Quality Prediction & Pollution Pattern Analysis

**Micro-project report — Data Analytics**

| | |
|---|---|
| Student name(s) | ⟨fill in⟩ |
| Roll no. / Class | ⟨fill in⟩ |
| Guide / Faculty | ⟨fill in⟩ |
| Institution | ⟨fill in⟩ |
| Academic year | ⟨fill in⟩ |

> **How to use this report.** Text is complete except items marked **⟨RESULT: …⟩**. Run
> `python -m src.model_training` and then `python -m src.results_summary`; copy the matching numbers and
> tables from `reports/results_summary.md` and the charts from the dashboard. Do not write any figure that
> did not come from your own run. Delete this box before submission.

---

## Abstract
Air pollution is a serious public-health issue in Indian cities. This project analyses daily air-quality data
from Indian cities to (A) describe pollution patterns across time, seasons and cities, and (B) estimate the
Air Quality Index (AQI) from same-day pollutant concentrations using Linear Regression and Random Forest.
The data were cleaned with documented rules, models were evaluated with a chronological train/validation/test
split, and the results are presented in an interactive Streamlit dashboard.
⟨RESULT: one sentence on the main finding, e.g. which city/season was worst and which model won, with its test RMSE and R².⟩

## 1. Introduction
The Air Quality Index converts concentrations of several pollutants into one number that the public can
understand. In India the Central Pollution Control Board (CPCB) defines six bands from *Good* to *Severe*.
Understanding when and where AQI is high helps in health advisories and policy, and a model that estimates AQI
from pollutant readings shows how strongly each pollutant relates to the index.

## 2. Problem statement and objectives
**Problem statement:** Analyse historical city-level air-quality data to identify pollution patterns and build
a regression model that estimates AQI from pollutant measurements.

**Objectives**
1. Clean and validate the dataset with transparent, documented rules.
2. Analyse daily, monthly, yearly and seasonal AQI trends.
3. Compare cities, identify pollution hotspots and the pollutants most associated with AQI.
4. Build and compare Linear Regression and Random Forest models against a mean baseline.
5. Present the analysis and a prediction tool in an interactive dashboard.

**Scope.** The model *estimates* AQI from same-day pollutant values. It does not forecast future AQI.

## 3. Dataset
Source: Kaggle, *Air Quality Data in India* (file `city_day.csv`).
Columns: City, Date, PM2.5, PM10, NO2, SO2, CO, O3, AQI (other columns in the file are ignored).

| Item | Value |
|---|---|
| Raw rows | ⟨RESULT: Dataset Explorer / results_summary §1⟩ |
| Rows after cleaning | ⟨RESULT⟩ |
| Number of cities | ⟨RESULT⟩ |
| Period covered | ⟨RESULT⟩ |
| Pollutants with ≥ 50 % coverage (used as features) | ⟨RESULT: `features` in results_summary §5⟩ |

**Missing data.** ⟨RESULT: paste the missing-value table and name the pollutants with the most missing data.⟩

## 4. Methodology
Full detail is in `docs/methodology.md`. In brief:

- **Cleaning:** invalid dates and duplicate (City, Date) rows removed; negative concentrations set to missing;
  rows with missing AQI dropped (AQI is never imputed or recomputed); outliers flagged with the IQR rule but
  kept, because extreme smog days are genuine.
- **Features:** pollutants present in at least 50 % of rows. `AQI_Bucket` is excluded because it is derived from
  the target (data leakage). City and Date are not used.
- **Models:** mean baseline, Linear Regression (median imputation + standardisation), Random Forest
  (200 trees, seed 42).
- **Evaluation:** chronological 70/15/15 split; model chosen on validation RMSE; test set used once; metrics MAE,
  MSE, RMSE, R². Imputation and scaling are inside a `Pipeline` so they learn from training data only.

## 5. System design and implementation
```
data/air_quality.csv
      │  data_loader.py  (load, validate, standardise column names)
      ▼
preprocessing.py  (clean, derive Year/Month/Season/AQI category, flag outliers)
      ├──► eda.py, pollution_analysis.py ──► visualizations.py ──► Streamlit pages
      └──► model_training.py ──► models/aqi_model.joblib + model_metadata.json
                                   │
                                   ▼
                            prediction.py ──► AQI Prediction page
```
Technology: Python, pandas, NumPy, scikit-learn, Plotly, Streamlit, joblib, pytest.
The dashboard has eight pages: Overview, Dataset Explorer, Exploratory Data Analysis, Pollution Pattern Analysis,
Pollutant Correlation, AQI Prediction, Model Performance and About, with sidebar filters (city, date range, year, month).

## 6. Results and analysis

### 6.1 AQI overview
⟨RESULT: AQI summary table (count, mean, median, min, max, std) and the category distribution; add the
histogram and category bar chart from the *Exploratory Data Analysis* page.⟩

### 6.2 Time and seasonal patterns
⟨RESULT: state the month and season with the highest and lowest average AQI, and how the yearly mean changed.
Include the monthly trend and month-of-year charts.⟩
*Caution to state in the report:* cities enter the data in different years, so yearly changes may reflect which
cities were measured, not only real change in air quality.

### 6.3 City comparison and hotspots
⟨RESULT: top-5 and bottom-5 cities by mean AQI with their observation counts; the hotspot table; the
city × month heatmap.⟩
Hotspot rule used: mean AQI above 200, or at least 25 % of days with AQI above 200.

### 6.4 Pollutant relationships
⟨RESULT: pollutants ranked by correlation with AQI, the dominant pollutant for the most polluted cities, and any
pairs with |r| ≥ 0.8.⟩
These are associations; they do not prove that a pollutant causes a given AQI.

### 6.5 Model performance
⟨RESULT: the full results table (validation and test: MAE, MSE, RMSE, R²) for Baseline, Linear Regression and
Random Forest; the selected model; actual-vs-predicted and residual plots; feature importance.⟩

Suggested wording once you have numbers (edit to match them):
- State whether each model beats the baseline.
- State which model had the lower **validation** RMSE (that is why it was selected) and report its **test** scores.
- Express RMSE/MAE in AQI points and relate to the category width (e.g. ~100 AQI points in the Moderate band).
- Comment on where residuals are largest (usually very high-AQI days).

## 7. Discussion
- The AQI in the dataset is itself calculated from pollutant concentrations by a formula, so a strong fit between
  pollutants and AQI is expected. The model learns that relationship approximately; it is not a discovery of new
  science.
- Linear Regression is simple and interpretable; Random Forest can capture non-linear effects but cannot predict
  beyond the training range. ⟨RESULT: say which performed better here and offer one reason.⟩
- Chronological splitting gives a more honest estimate than a random split because the model is tested on later
  dates than it was trained on.

## 8. Limitations
1. Same-day estimation only; not forecasting.
2. Different cities have different observation periods; overall and yearly comparisons are not like-for-like.
3. Median imputation is simple; pollutants with heavy missingness weaken the model.
4. Inputs far outside the training range give unreliable estimates.
5. Correlation does not imply causation.
6. The output is a model estimate, not an official AQI.

## 9. Conclusion
⟨RESULT: 4–6 sentences — (i) which cities/seasons are worst, (ii) which pollutants relate most strongly to AQI,
(iii) which model was selected and its test RMSE/R², (iv) one limitation.⟩
The project delivers a reproducible pipeline, an evaluated model, automated tests and an interactive dashboard.

## 10. Future scope
- Time-series forecasting (SARIMA / Prophet / gradient-boosted lag models) with rolling-origin validation.
- Weather features (temperature, humidity, wind) and festival/crop-burning indicators.
- City-grouped cross-validation to test generalisation to unseen cities.
- Hyper-parameter tuning and additional models (Gradient Boosting, XGBoost).
- Persistent storage (SQLite) and deployment on Streamlit Community Cloud.

## References
1. Rao, R. *Air Quality Data in India (2015–2020)*. Kaggle. https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
2. Central Pollution Control Board (CPCB), India. *National Air Quality Index* (AQI categories and sub-index method).
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
5. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proc. 9th Python in Science Conf.*
6. Streamlit documentation. https://docs.streamlit.io
7. Plotly Python documentation. https://plotly.com/python/

## Appendix A — How to run
See `README.md` (Setup).

## Appendix B — Project structure
See `README.md` (Folder structure).
