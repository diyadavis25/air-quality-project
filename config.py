"""Central configuration. No machine-specific paths: everything is relative to this file.
Override the dataset location with the AQ_DATA_PATH environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("AQ_DATA_PATH", BASE_DIR / "data" / "air_quality.csv"))
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_PATH = MODELS_DIR / "aqi_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"

RANDOM_SEED = 42
# Chronological split: first 70% train, next 15% validation (model selection), last 15% test (final report)
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
MIN_FEATURE_COVERAGE = 0.50  # a pollutant must be non-missing in >=50% of valid rows to be a model feature

# Pollutants used for analysis AND as model inputs (the only pollutant columns the project reads)
POLLUTANT_CANDIDATES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
POLLUTANT_UNITS = {"PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³", "SO2": "µg/m³",
                   "CO": "dataset units (CPCB data: mg/m³)", "O3": "µg/m³"}
POLLUTANT_HELP = {
    "PM2.5": "Fine particles (≤2.5 µm). Usually the strongest driver of AQI.",
    "PM10": "Coarse particles (≤10 µm), e.g. dust. Strongly related to PM2.5.",
    "NO2": "Nitrogen dioxide, mainly from vehicle and industrial combustion.",
    "SO2": "Sulphur dioxide, mainly from coal burning and industry.",
    "CO": "Carbon monoxide. Its unit is dataset-specific: use the same unit as your training CSV.",
    "O3": "Ground-level ozone, formed by sunlight acting on other pollutants.",
}

# CPCB (India) AQI categories: (upper bound inclusive, label)
AQI_BINS = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
            (300, "Poor"), (400, "Very Poor"), (float("inf"), "Severe")]
AQI_ORDER = [b[1] for b in AQI_BINS]
AQI_COLORS = {"Good": "#2E7D32", "Satisfactory": "#7CB342", "Moderate": "#F9A825",
              "Poor": "#EF6C00", "Very Poor": "#C62828", "Severe": "#6A1B1B"}

# Indian seasons by month
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
           6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"}

ACCENT = "#7B1E2B"
