from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODELS_DIR = BASE_DIR / "models"

TARGET_COLUMN = "disease"

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "smoker",
]

