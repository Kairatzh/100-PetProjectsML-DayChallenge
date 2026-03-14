## Classic ML 1 — Tabular Classification (Titanic-style)

Небольшой пет-проект с классической задачей бинарной классификации в стиле Titanic: по табличным признакам предсказываем вероятность события.

Проект использует:
- scikit-learn (логистическая регрессия)
- FastAPI (обёртка в виде сервиса)

### Установка

Рекомендуется отдельное виртуальное окружение (пример для Windows PowerShell):

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Запуск сервиса

```bash
uvicorn app.main:app --reload
```

Документация будет доступна по адресу:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Эндпоинты

- `GET /health` — проверка, что сервис жив.
- `POST /predict` — принимает признаки одного "пассажира" и возвращает вероятность положительного класса и предсказанный класс.

