# Інструкція з тестування DogFACS Dataset Generator

Цей документ містить інструкції для тестування всіх компонентів проекту.

---

## Spis treści

1. [Швидкий старт (bez pytest)](#1-швидкий-старт-без-pytest)
2. [Повний набір тестів (pytest)](#2-повний-набір-тестів-pytest)
3. [Тестування FastAPI Backend](#3-тестування-fastapi-backend)
4. [Тестування React Frontend](#4-тестування-react-frontend)
5. [Перевірка валідації COCO](#5-перевірка-валідації-coco)
6. [Troubleshooting](#6-troubleshooting)
7. [Статус тестового покриття](#7-статус-тестового-покриття)

---

## 1. Швидкий старт (без pytest)

Якщо у вас немає pytest або виникають проблеми з установкою, використовуйте простий test runner:

```bash
python test_runner.py
```

Цей скрипт виконує 6 основних тестів:
1. ✅ Перевірка імпортів (всі модулі завантажуються)
2. ✅ Валідація EMOTION_RULES (6 правил)
3. ✅ Валідація ACTION_UNIT_NAMES (12 офіційних DogFACS кодів)
4. ✅ Тест DogFACSRuleEngine (класифікація емоцій)
5. ✅ Тест NeutralFrameDetector (автодетекція нейтрального фрейму)
6. ✅ Тест DeltaActionUnitsExtractor (обчислення delta AU)

**Очікуваний результат:**
```
=== Запуск базових тестів DogFACS ===

[Test 1/6] Перевірка імпортів...
✅ Всі імпорти успішні

[Test 2/6] Перевірка EMOTION_RULES...
✅ EMOTION_RULES: 6 правил

[Test 3/6] Перевірка ACTION_UNIT_NAMES...
✅ ACTION_UNIT_NAMES: 12 AU

[Test 4/6] Тест DogFACSRuleEngine...
✅ DogFACSRuleEngine працює коректно

[Test 5/6] Тест NeutralFrameDetector...
✅ NeutralFrameDetector працює коректно

[Test 6/6] Тест DeltaActionUnitsExtractor...
✅ DeltaActionUnitsExtractor працює коректно

=== Результат ===
✅ Всі тести пройдені: 6/6
```

---

## 2. Повний набір тестів (pytest)

### Установка pytest

```bash
pip install pytest pytest-cov
```

### Запуск всіх тестів

```bash
# Всі тести
pytest

# З детальним виводом
pytest -v

# З покриттям коду
pytest --cov=packages --cov-report=html

# Тільки тести для models
pytest tests/test_models/

# Тільки тести для pipeline
pytest tests/test_pipeline/
```

### Очікувані результати

```bash
$ pytest -v

tests/test_models/test_emotion_rules.py::test_emotion_rules_exist PASSED
tests/test_models/test_emotion_rules.py::test_happy_rule PASSED
tests/test_models/test_emotion_rules.py::test_angry_rule PASSED
tests/test_models/test_delta_action_units.py::test_delta_au_no_change PASSED
tests/test_models/test_delta_action_units.py::test_delta_au_increase PASSED
tests/test_pipeline/test_neutral_frame.py::test_auto_detect PASSED
tests/test_pipeline/test_peak_selector.py::test_select_peaks PASSED
tests/test_data/test_coco.py::test_dogfacs_extensions PASSED

======================== 12 passed in 0.45s ========================
```

---

## 3. Тестування FastAPI Backend

### 1. Health Check

```bash
# Запустіть backend
cd apps/webapp/backend
python main.py
```

У іншому терміналі:

```bash
# Перевірка здоров'я API
curl http://localhost:8000/api/health
```

**Очікувана відповідь:**
```json
{
  "status": "ok",
  "pipeline_loaded": true
}
```

### 2. Тест обробки відео

```bash
curl -X POST http://localhost:8000/api/process_video \
  -F "file=@test_video.mp4" \
  -F "num_peaks=10" \
  -F "min_separation_frames=30"
```

**Очікувана відповідь:**
```json
{
  "session_id": "abc123",
  "video_filename": "test_video.mp4",
  "neutral_frame_idx": 42,
  "neutral_frame_url": "/static/abc123/frame_0042.jpg",
  "peak_frames": [
    {
      "frame_idx": 100,
      "image_url": "/static/abc123/frame_0100.jpg",
      "aus": {
        "AU101": {
          "ratio": 1.15,
          "delta": 0.15,
          "is_active": true,
          "confidence": 0.9
        }
      },
      "emotion": "happy",
      "emotion_confidence": 0.85,
      "tfm_score": 2.345
    }
  ],
  "total_frames": 600
}
```

### 3. Тест експорту COCO

```bash
curl -X POST http://localhost:8000/api/export_coco \
  -H "Content-Type: application/json" \
  -d @export_request.json \
  --output dataset.json
```

---

## 4. Тестування React Frontend

### 1. Запуск dev сервера

```bash
cd apps/webapp/frontend
npm install
npm run dev
```

Відкрийте браузер: `http://localhost:5173`

### 2. E2E тестування (ручне)

#### Крок 1: Upload Video
- ✅ Натисніть "Choose Video" або перетягніть MP4 файл
- ✅ Перевірте, що з'явилася назва файлу
- ✅ Кнопка "Process Video" активна

#### Крок 2: Process Video
- ✅ Натисніть "Process Video"
- ✅ З'явився loader "Processing..."
- ✅ Після обробки (30-60 сек) з'являються:
  - Neutral frame (зелений маркер)
  - 10 peak frames (червоні маркери)
  - Grid з картками фреймів

#### Крок 3: Review Peak Frames
Для кожної картки перевірте:
- ✅ Зображення фрейму відображається
- ✅ 12 чекбоксів AU (AU101, AU12, EAD102, ...)
- ✅ Активні AU позначені галочкою
- ✅ Показані значення ratio (наприклад, 1.15)
- ✅ Емоція відображається з emoji
- ✅ Confidence та TFM score показані

#### Крок 4: Toggle AU
- ✅ Натисніть чекбокс AU12 (Lip Corner Puller)
- ✅ Емоція змінилася в реальному часі
- ✅ Confidence оновилося

#### Крок 5: Export Dataset
- ✅ Натисніть "Export COCO Dataset"
- ✅ Файл JSON завантажився

#### Крок 6: Reset
- ✅ Натисніть "Reset"
- ✅ Всі дані очистилися
- ✅ Можна завантажити нове відео

---

## 5. Перевірка валідації COCO

```bash
# Валідація експортованого датасету
python scripts/annotation/validate_coco.py exported_dataset.json

# Strict mode (всі попередження = помилки)
python scripts/annotation/validate_coco.py exported_dataset.json --strict
```

**Очікуваний результат:**
```
=== COCO Annotation Validation ===
File: exported_dataset.json

✅ Schema validation passed
✅ Image IDs are unique
✅ Annotation IDs are unique
✅ All annotations reference valid images
✅ Bounding boxes are valid
✅ Keypoints are valid
✅ DogFACS extensions present: 10 annotations

Statistics:
  Total images: 10
  Total annotations: 10
  Annotations with AU analysis: 10
  Unique emotion rules: 4

Emotion distribution:
  happy: 4
  angry: 2
  fearful: 2
  relaxed: 2

✅ Validation completed successfully
```

---

## 6. Troubleshooting

### NumPy compatibility error

**Помилка:** `AttributeError: _ARRAY_API not found` або `numpy.core.multiarray failed to import`

**Причина:** OpenCV скомпільований з NumPy 1.x, потрібен downgrade

**Рішення:**
```bash
pip install "numpy<2" --force-reinstall
pip install opencv-python --force-reinstall
```

### Backend не стартує

**Помилка:** `ModuleNotFoundError: No module named 'packages'`

**Рішення:**
```bash
# Встановіть пакет в editable mode
pip install -e .
```

### Frontend не підключається до backend

**Помилка:** `Network Error` або `ERR_CONNECTION_REFUSED`

**Рішення:**
```bash
# Перевірте, що backend запущений
curl http://localhost:8000/api/health

# Перевірте Vite proxy в vite.config.ts
```

### CORS errors

**Помилка:** `Access-Control-Allow-Origin` error

**Рішення:** Перевірте CORS налаштування в `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Модель не завантажується

**Помилка:** `FileNotFoundError: models/yolov8m.pt`

**Рішення:**
```bash
python scripts/download/download_models.py
```

### Тести падають з ImportError

**Помилка:** `ImportError: cannot import name 'EmotionConfig'`

**Рішення:** Переконайтесь, що ви на актуальній версії коду:
```bash
git pull origin sprint-7
```

---

## 7. Статус тестового покриття

| Модуль | Тести | Статус |
|--------|-------|--------|
| models/emotion.py | ✅ | Rule-based імплементація |
| models/delta_action_units.py | ✅ | Delta AU calculation |
| pipeline/neutral_frame.py | ✅ | Auto-detection |
| pipeline/peak_selector.py | ✅ | Peak selection |
| data/coco.py | ✅ | DogFACS extensions |
| webapp/backend/main.py | ⚠️ | Потрібні API тести |
| webapp/frontend/* | ❌ | Потрібні unit тести |

**Легенда:**
- ✅ = Добре покриття
- ⚠️ = Часткове покриття
- ❌ = Немає тестів

---

## Контакти

- Linear: https://linear.app/team/DOG
- GitHub: https://github.com/pg-weti/dog-facs

---

*Документ оновлено: Січень 2026 (Sprint 7 - DogFACS Dataset Generator)*
