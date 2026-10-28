# Dataset folder

Place the dataset here as `air_quality.csv`.

1. Go to https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
2. Download and extract the archive.
3. Copy **city_day.csv** into this folder and rename it **air_quality.csv**.

(Alternatively set the `AQ_DATA_PATH` environment variable to any compatible CSV.)
Columns used: Date, City, PM2.5, PM10, NO2, SO2, CO, O3, AQI. `Date` and `AQI` are required (AQI is the training target);
any other columns (NO, NOx, NH3, Benzene, AQI_Bucket, …) are ignored.
The CSV is git-ignored. No sample/fake data is included.
