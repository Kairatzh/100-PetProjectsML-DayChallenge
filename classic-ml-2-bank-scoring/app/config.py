from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODELS_DIR = BASE_DIR / "models"

TARGET_COLUMN = "default"

NUMERIC_FEATURES = [
    "age",
    "income",
    "loan_amount",
    "loan_term_months",
    "has_property",
    "has_previous_loans",
]

