from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PatientFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100)
    bmi: float = Field(..., ge=10, le=60)
    systolic_bp: float = Field(..., ge=80, le=220)
    diastolic_bp: float = Field(..., ge=40, le=140)
    cholesterol: float = Field(..., ge=2, le=10, description="Отношение total/HDL")
    smoker: int = Field(..., ge=0, le=1)


class DiseaseResponse(BaseModel):
    disease_proba: float
    disease_class: int


class DiseaseModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=500, random_state=42)

    def fit(self) -> None:
        X, y = make_classification(
            n_samples=1500,
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
def get_model() -> DiseaseModel:
    model = DiseaseModel()
    model.fit()
    return model


app = FastAPI(
    title="Classic ML 3 — Disease Risk Prediction",
    description="FastAPI сервис для предсказания риска заболевания на базе LogisticRegression.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=DiseaseResponse)
def predict(features: PatientFeatures) -> DiseaseResponse:
    model = get_model()
    data = np.array(
        [
            [
                features.age,
                features.bmi,
                features.systolic_bp,
                features.diastolic_bp,
                features.cholesterol,
                features.smoker,
            ]
        ]
    )
    proba = model.predict_proba(data)
    cls = model.predict_class(data)
    return DiseaseResponse(disease_proba=proba, disease_class=cls)

