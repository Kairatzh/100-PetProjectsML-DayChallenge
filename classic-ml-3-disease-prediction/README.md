## Classic ML 3 — Disease Risk Prediction

Классический пет-проект по предсказанию риска заболевания по клиническим признакам (табличные данные, бинарная классификация).

Модель: `LogisticRegression` из scikit-learn.  
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

- `GET /health` — проверка сервиса.
- `POST /predict` — предсказывает вероятность заболевания и класс.

