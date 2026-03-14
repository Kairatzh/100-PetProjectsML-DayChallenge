## Classic ML 5 — House Prices Regression

Классический пет-проект по регрессии: предсказание цены жилья по табличным признакам.

Модель: `RandomForestRegressor`.  
Сервис: FastAPI.

### Установка

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Запуск

```bash
uvicorn app.main:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`

### Эндпоинты

- `GET /health` — статус.
- `POST /predict` — предсказывает цену объекта.

