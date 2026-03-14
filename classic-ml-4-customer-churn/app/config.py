from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODELS_DIR = BASE_DIR / "models"

TARGET_COLUMN = "churn"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "is_active",
    "has_support_tickets",
]

