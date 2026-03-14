from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=0, le=240)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_products: int = Field(..., ge=1, le=10)
    is_active: int = Field(..., ge=0, le=1)
    has_support_tickets: int = Field(..., ge=0, le=1)


class ChurnResponse(BaseModel):
    churn_proba: float
    churn_class: int


class ChurnModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=7, random_state=42
        )

    def fit(self) -> None:
        X, y = make_classification(
            n_samples=2500,
            n_features=6,
            n_informative=5,
            n_redundant=0,
            n_classes=2,
            weights=[0.8, 0.2],
            class_sep=1.2,
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
def get_model() -> ChurnModel:
    model = ChurnModel()
    model.fit()
    return model


app = FastAPI(
    title="Classic ML 4 — Customer Churn Prediction",
    description="FastAPI сервис по предсказанию оттока клиентов (churn) на RandomForest.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(features: CustomerFeatures) -> ChurnResponse:
    model = get_model()
    data = np.array(
        [
            [
                features.tenure_months,
                features.monthly_charges,
                features.total_charges,
                features.num_products,
                features.is_active,
                features.has_support_tickets,
            ]
        ]
    )
    proba = model.predict_proba(data)
    cls = model.predict_class(data)
    return ChurnResponse(churn_proba=proba, churn_class=cls)

