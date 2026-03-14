from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODELS_DIR = BASE_DIR / "models"

TARGET_COLUMN = "price"

NUMERIC_FEATURES = [
    "area_sq_m",
    "num_rooms",
    "num_bathrooms",
    "distance_to_center_km",
    "building_age_years",
    "floor",
]

