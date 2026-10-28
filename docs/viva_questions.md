# Viva Questions & Answers

Answers are based on what this project actually does. Where an answer needs your own number, it says
**[your result]**: look it up in `reports/results_summary.md` and quote it.

---

## A. Project basics

**1. What is the aim of your project?**
To analyse pollution patterns in Indian cities (trends, seasons, city comparison) and to build a model that
estimates AQI from pollutant readings.

**2. What is AQI?**
The Air Quality Index turns concentrations of pollutants (PM2.5, PM10, NO2, SO2, CO, O3, etc.) into a single
number. India's CPCB bands: 0–50 Good, 51–100 Satisfactory, 101–200 Moderate, 201–300 Poor, 301–400 Very Poor, 401+ Severe.

**3. Is your model a forecasting model?**
No. It estimates AQI from the *same day's* pollutant values. Forecasting needs a time-series approach and is listed
as future scope.

**4. Which dataset did you use and how big is it?**
Kaggle "Air Quality Data in India", `city_day.csv`. **[your result: rows, cities, period]**.

**5. Which tools did you use and why?**
pandas/NumPy for data handling, scikit-learn for modelling, Plotly for interactive charts, Streamlit for the
dashboard, joblib to save the model, pytest for tests.

---

## B. Data cleaning

**6. How did you handle missing values?**
Rows with missing AQI were dropped because AQI is the target and must not be invented. Missing pollutant values
were kept and filled with the training median *inside the model pipeline*, so no information from validation/test
data leaks in.

**7. Why not just fill every missing value with the mean at the start?**
Filling before splitting lets test-set information influence the training data (data leakage). It also hides how
much data is missing.

**8. What did you do with outliers?**
Flagged them with the IQR rule (below Q1 − 1.5·IQR or above Q3 + 1.5·IQR) but did not remove them. Very high
pollution days (smog episodes) are real and are exactly what the study is about.

**9. Why were negative values set to missing?**
A concentration cannot be negative; it is a sensor/recording error.

**10. Why did you drop duplicate (City, Date) rows?**
Each city should have one reading per day; duplicates would double-count that day.

---

## C. Analysis

**11. How did you define a "hotspot"?**
A city whose mean AQI is above 200, or where at least 25 % of days have AQI above 200 (Poor or worse).

**12. How did you rank cities?**
By mean AQI. Cities with fewer than 30 valid days are excluded so a handful of readings cannot distort the ranking.

**13. Which city/season is most polluted?**
**[your result]**. (Typically winter and post-monsoon are worse in northern India, but quote only what your data show.)

**14. What does correlation tell you? Does it show causation?**
Correlation measures how strongly two variables move together. It does not prove one causes the other.

**15. Pearson vs Spearman?**
Pearson measures linear relationships and is sensitive to outliers. Spearman uses ranks, so it captures any
monotonic relationship and is more robust to outliers. The dashboard offers both.

**16. What is multicollinearity and did you find it?**
When two input variables are highly correlated (|r| ≥ 0.8 in this project), so their separate effects are hard to
separate. It affects the interpretation of Linear Regression coefficients more than Random Forest predictions.
**[your result: pairs listed, e.g. PM2.5 and PM10, if present]**.

**17. Why is yearly comparison risky here?**
Cities were added at different times, so the yearly average may change because of *which* cities were measured, not
because the air changed.

---

## D. Modelling

**18. Which models did you use?**
A mean baseline, Linear Regression and Random Forest.

**19. Why include a baseline?**
It is the simplest guess (always predict the average AQI). A model is only useful if it beats this.

**20. Why did you use a chronological split instead of a random split?**
The data are a time series. With a random split the model could train on days after the ones it is tested on,
which gives over-optimistic results. Chronological: first 70 % train, next 15 % validation, last 15 % test.

**21. Why three sets (train / validation / test)?**
Train fits the models; validation picks the better model; test is used once at the end for an unbiased score. If
you choose a model on the test set, the test score is no longer honest.

**22. What is data leakage? Where did you avoid it?**
Using information in training that would not be available at prediction time or that reveals the answer.
Avoided by: excluding `AQI_Bucket` (derived from AQI), fitting imputers/scalers on training data only, and
choosing the model on validation data.

**23. Explain Linear Regression.**
Fits AQI = b0 + b1·PM2.5 + b2·PM10 + … by minimising squared errors. Simple and interpretable, but assumes a linear
relationship.

**24. Explain Random Forest.**
An ensemble of many decision trees trained on random samples of data and features; predictions are averaged. It
handles non-linear patterns and reduces overfitting compared with one tree, but it cannot predict outside the range
of AQI it has seen.

**25. Which model performed better?**
**[your result: selected model, validation RMSE, test RMSE, test R²]**. The selection was on validation RMSE.

**26. Explain MAE, MSE, RMSE and R².**
MAE: average absolute error in AQI points. MSE: average squared error. RMSE: square root of MSE, same unit as AQI,
penalises big errors more. R²: proportion of variance explained (1 is perfect; 0 equals the mean baseline;
negative is worse than the baseline).

**27. Your R² is high. Is that suspicious?**
Not necessarily: the dataset's AQI is itself calculated from the pollutant concentrations, so a strong relationship
is expected. I made sure there is no leakage from `AQI_Bucket` and that the test set was only used once. The model
approximates a known formula; it does not discover something new.

**28. What does the residual plot show?**
Residual = actual − predicted. Random scatter around zero is good; a pattern or a funnel shape means systematic
error (often larger errors on very high-AQI days).

**29. What is feature importance?**
In Random Forest, how much each pollutant reduces prediction error across the trees. It shows association with
the prediction, not causation. **[your result: top features]**.

**30. Why did you scale features for Linear Regression but not for Random Forest?**
Scaling puts features on a comparable scale (useful for coefficient comparison). Trees split on thresholds, so
scaling does not change their result.

**31. What is a Pipeline and why use one?**
It chains imputation → scaling → model as one object. Fitting it on training data guarantees every step learns
only from training data, and the same steps are applied automatically at prediction time.

**32. How do you set the random seed and why?**
`random_state=42` for reproducibility: another person running the code gets the same model.

**33. Did you tune hyper-parameters?**
No; fixed reasonable values (200 trees, `min_samples_leaf=2`). Tuning with cross-validation is future scope.

**34. What happens if a user leaves a pollutant blank in the dashboard?**
It is filled with the training median and the app tells the user which ones were filled.

**35. What if the user enters a very large value?**
If a value exceeds 1.5× the training maximum, the app warns that the estimate is unreliable; the output is also
clipped to 0–500.

---

## E. Limitations and extensions

**36. What are the limitations?**
Same-day estimation only; city coverage differs; simple imputation; unreliable outside training range;
correlation ≠ causation; not an official AQI.

**37. How would you forecast AQI for tomorrow?**
Use a time-series approach (SARIMA, Prophet) or lag features (yesterday's AQI/pollutants) with gradient boosting,
validated with rolling-origin splits; add weather data.

**38. How would you test generalisation to a new city?**
Leave-one-city-out (group) cross-validation.

**39. How did you test your code?**
pytest tests for loading, cleaning rules, AQI categories, outlier flagging, filters, chronological split, metrics,
leakage, prediction and input validation. The model tests use a small synthetic dataset only to check the code works.

**40. How can this help in real life?**
Identify high-risk cities/seasons for health advisories and policy focus, and estimate AQI where a monitoring
station records pollutants but not the index. (Tools like this support, but do not replace, official monitoring.)

---

## Quick "gotcha" questions
- *Why not remove all outliers?* They are real pollution events.
- *Why not use AQI_Bucket as a feature?* It is computed from AQI, so it leaks the answer.
- *Why is the mean-baseline R² about 0 or negative?* It predicts a constant, so it explains no variance.
- *Can your model predict next week's AQI?* No; it needs that day's pollutant readings.
- *Does high PM2.5 cause high AQI?* AQI is calculated from PM2.5 (among others), so they are linked by definition; the statistics show association.
