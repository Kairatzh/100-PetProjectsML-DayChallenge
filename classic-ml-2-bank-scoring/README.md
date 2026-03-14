## Classic ML 2 — Bank Scoring (Credit Risk)

Классический пет-проект по скорингу клиентов банка: по табличным признакам предсказываем риск невозврата.

Проект использует:
- scikit-learn (RandomForestClassifier)
- FastAPI (обёртка в виде сервиса)

### Установка

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Запуск сервиса

```bash
uvicorn app.main:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`

### Эндпоинты

- `GET /health` — проверка статуса.
- `POST /predict` — предсказание вероятности дефолта клиента.

