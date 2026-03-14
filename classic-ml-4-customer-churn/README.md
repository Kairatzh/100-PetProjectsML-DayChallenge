## Classic ML 4 — Customer Churn Prediction

Пет-проект по предсказанию оттока клиентов (churn) для телеком/сервиса: бинарная классификация, классический ML.

Модель: `RandomForestClassifier`.  
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
- `POST /predict` — вероятность оттока и предсказанный класс.

