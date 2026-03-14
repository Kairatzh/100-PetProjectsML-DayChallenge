from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ClientFeatures(BaseModel):
    age: int = Field(..., ge=18, le=90)
    income: float = Field(..., ge=0)
    loan_amount: float = Field(..., ge=0)
    loan_term_months: int = Field(..., ge=1, le=360)
    has_property: int = Field(..., ge=0, le=1)
    has_previous_loans: int = Field(..., ge=0, le=1)


class ScoringResponse(BaseModel):
    default_proba: float
    default_class: int


class BankScoringModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        )

    def fit(self) -> None:
        X, y = make_classification(
            n_samples=2000,
            n_features=6,
            n_informative=5,
            n_redundant=0,
            n_classes=2,
            weights=[0.7, 0.3],
            class_sep=1.5,
            random_state=42,
        )
        X_train, X_val, y_train, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict_proba(self, features: np.ndarray) -> float:
        X_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(X_scaled)[:, 1][0]
        return float(proba)

    def predict_class(self, features: np.ndarray) -> int:
        X_scaled = self.scaler.transform(features)
        pred = int(self.model.predict(X_scaled)[0])
        return pred


@lru_cache(maxsize=1)
def get_model() -> BankScoringModel:
    model = BankScoringModel()
    model.fit()
    return model


app = FastAPI(
    title="Classic ML 2 — Bank Scoring",
    description="Классический скоринговый сервис банка на RandomForestClassifier.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=ScoringResponse)
def predict(features: ClientFeatures) -> ScoringResponse:
    model = get_model()
    data = np.array(
        [
            [
                features.age,
                features.income,
                features.loan_amount,
                features.loan_term_months,
                features.has_property,
                features.has_previous_loans,
            ]
        ]
    )
    proba = model.predict_proba(data)
    default_class = model.predict_class(data)
    return ScoringResponse(default_proba=proba, default_class=default_class)

