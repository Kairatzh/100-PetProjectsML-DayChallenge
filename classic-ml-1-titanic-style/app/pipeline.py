import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Title"] = data["Name"].str.extract(r", ([A-Za-z]+)\.")
    data["Title"] = data["Title"].replace(
        {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        }
    )
    rare_titles = [
        "Dr",
        "Rev",
        "Col",
        "Major",
        "Countess",
        "Capt",
        "Sir",
        "Lady",
        "Jonkheer",
        "Don",
        "Dona",
    ]
    data["Title"] = data["Title"].replace(rare_titles, "Rare")
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    return data


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def build_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()
    model = LogisticRegression(max_iter=500, random_state=42)
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return clf

