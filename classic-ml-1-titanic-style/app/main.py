from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PassengerFeatures(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Класс билета (1, 2, 3)")
    sex_male: int = Field(..., ge=0, le=1, description="Пол: 1 — мужчина, 0 — женщина")
    age: float = Field(..., ge=0, le=100, description="Возраст")
    fare: float = Field(..., ge=0, description="Стоимость билета")
    family_size: int = Field(..., ge=1, le=10, description="Размер семьи (с собой)")


class PredictionResponse(BaseModel):
    survived_proba: float
    survived_class: int


class TitanicStyleModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=500, random_state=42)

    def fit(self) -> None:
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=4,
            n_redundant=0,
            n_classes=2,
            weights=[0.62, 0.38],
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
def get_model() -> TitanicStyleModel:
    model = TitanicStyleModel()
    model.fit()
    return model


app = FastAPI(
    title="Classic ML 1 — Titanic-style Tabular Classification",
    description="Простой FastAPI сервис с классической бинарной классификацией в стиле Titanic.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PassengerFeatures) -> PredictionResponse:
    model = get_model()
    data = np.array(
        [
            [
                features.pclass,
                features.sex_male,
                features.age,
                features.fare,
                features.family_size,
            ]
        ]
    )
    proba = model.predict_proba(data)
    pred_class = model.predict_class(data)
    return PredictionResponse(survived_proba=proba, survived_class=pred_class)

