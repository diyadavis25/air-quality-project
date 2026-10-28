# Presentation Outline (12 slides, about 10–12 minutes)

Slides marked **⟨RESULT⟩** need numbers or charts from your own run (`reports/results_summary.md` and the dashboard).
Keep each slide to a title, one visual and at most 3–4 short bullets.

| # | Slide | Content | Time |
|---|---|---|---|
| 1 | **Title** | Project title, names, guide, institution, date | 0:20 |
| 2 | **Problem & motivation** | Why air quality matters; AQI turns many pollutants into one number; CPCB bands (use the colour scale from the dashboard) | 1:00 |
| 3 | **Objectives & scope** | A: pollution pattern analysis. B: AQI estimation with ML. **Not** forecasting | 0:45 |
| 4 | **Dataset** | Kaggle *Air Quality Data in India*; columns; **⟨RESULT⟩** rows, cities, period (from *Dataset Explorer*) | 0:45 |
| 5 | **Method overview** | Flow diagram: Load → Clean → EDA → Model → Dashboard (the diagram in the report §5) | 0:45 |
| 6 | **Data cleaning** | 5 rules: dates, duplicates, negatives→NaN, missing AQI dropped (never imputed), outliers flagged not removed. **⟨RESULT⟩** counts affected | 1:00 |
| 7 | **Pollution patterns** | **⟨RESULT⟩** monthly trend + seasonal bar; one sentence: worst/best month and season. Note: city coverage differs by year | 1:30 |
| 8 | **City comparison & hotspots** | **⟨RESULT⟩** city ranking bar or heatmap; hotspot rule (mean >200 or ≥25 % days >200) | 1:30 |
| 9 | **Pollutant relationships** | **⟨RESULT⟩** correlation heatmap; strongest links to AQI; "association ≠ causation" | 1:00 |
| 10 | **Modelling & results** | Baseline vs Linear Regression vs Random Forest; chronological 70/15/15 split; chosen on validation RMSE; **⟨RESULT⟩** metrics table + actual-vs-predicted plot | 1:45 |
| 11 | **Live demo** | Dashboard: filter a city → Pattern page → AQI Prediction with sample values → Model Performance | 1:30 |
| 12 | **Limitations, conclusion, future scope** | Same-day estimate only; uneven coverage; unreliable outside training range. Conclusion in 2–3 lines. Future: time-series forecasting, weather features, city-grouped validation | 1:00 |

## Speaker notes (key sentences)
- **Slide 3:** "Our model estimates AQI from the day's pollutant readings. It is not a forecast."
- **Slide 6:** "We never invent AQI values, and we keep outliers because extreme smog days are real events."
- **Slide 10:** "We chose the model on the validation set and looked at the test set only once, so the test score is an honest estimate. The AQI is calculated from pollutants by a formula, so a high R² is expected; we checked for leakage by excluding the AQI bucket."
- **Slide 11:** Start the app *before* the presentation (`streamlit run app.py`) and train the model beforehand (`python -m src.model_training`). Have a screenshot backup in case of no laptop/Internet problems.

## Demo checklist
1. `data/air_quality.csv` present; model trained; app opens without errors.
2. Pick 2–3 cities in the sidebar filter to show the pages react.
3. Prepare one example input (e.g., typical values shown as placeholders in the prediction form) and read out the estimated AQI and category.
4. Keep the viva file (`docs/viva_questions.md`) open for revision before presenting.

## Style tips
- One message per slide; a chart beats a table.
- Use the same colours as the dashboard (maroon accent, CPCB category colours).
- Label units (AQI points, µg/m³) and always say which years/cities a chart covers.
- Do not quote numbers you did not generate from your own run.
