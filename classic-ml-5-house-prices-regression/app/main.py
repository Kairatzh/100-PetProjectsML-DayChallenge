from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HouseFeatures(BaseModel):
    area_sq_m: float = Field(..., ge=10, le=1000)
    num_rooms: int = Field(..., ge=1, le=20)
    num_bathrooms: int = Field(..., ge=1, le=10)
    distance_to_center_km: float = Field(..., ge=0, le=100)
    building_age_years: int = Field(..., ge=0, le=150)
    floor: int = Field(..., ge=1, le=50)


class PriceResponse(BaseModel):
    price: float


class HousePriceModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=42
        )

    def fit(self) -> None:
        X, y = make_regression(
            n_samples=3000,
            n_features=6,
            n_informative=6,
            noise=15.0,
            random_state=42,
        )
        X_train, X_val, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict_price(self, features: np.ndarray) -> float:
        X_scaled = self.scaler.transform(features)
        price = float(self.model.predict(X_scaled)[0])
        return price


@lru_cache(maxsize=1)
def get_model() -> HousePriceModel:
    model = HousePriceModel()
    model.fit()
    return model


app = FastAPI(
    title="Classic ML 5 — House Prices Regression",
    description="FastAPI сервис по предсказанию цены жилья на основе RandomForestRegressor.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PriceResponse)
def predict(features: HouseFeatures) -> PriceResponse:
    model = get_model()
    data = np.array(
        [
            [
                features.area_sq_m,
                features.num_rooms,
                features.num_bathrooms,
                features.distance_to_center_km,
                features.building_age_years,
                features.floor,
            ]
        ]
    )
    price = model.predict_price(data)
    return PriceResponse(price=price)

