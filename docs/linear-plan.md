# DogFACS — Linear Plan (Реструктуризация)

Проект: **Dogs-ai (DOG)**
Команда: https://linear.app/team/DOG

---

## SPRINT 8 — Научная основа и исправление ошибок
**Цель:** Привести AU коды и keypoints в соответствие с DogFACS

| ID | Задача | Приоритет | Labels |
|----|--------|-----------|--------|
| DOG-S8-1 | Расширить keypoints с 20 до 46 (DogFLW) в schemas.py | Urgent | model, data |
| DOG-S8-2 | Исправить AU коды в delta_action_units.py (21 официальных DogFACS) | Urgent | model |
| DOG-S8-3 | Добавить 3 эмоции (surprise, pain, submission) в emotion.py | High | model |
| DOG-S8-4 | Исправить emotion rules под правильные AU коды | High | model |
| DOG-S8-5 | Починить координатный баг в inference.py (keypoints offset) | High | pipeline |
| DOG-S8-6 | Убрать дубликат HeadPoseEstimator (neutral_frame.py vs head_pose.py) | Medium | pipeline |
| DOG-S8-7 | Обновить peak_selector.py под правильные AU весa | Medium | pipeline |
| DOG-S8-8 | Обновить все тесты под 46 keypoints и 21 AU | Medium | pipeline |

**Описания задач:**

### DOG-S8-1: Расширить keypoints с 20 до 46
Файл: `packages/data/schemas.py`
- Обновить `NUM_KEYPOINTS = 46`
- Добавить все 46 названий по DogFLW схеме (глаза×8, брови×6, уши×8, нос×4, рот×8, морда×4, прочее×8)
- Обновить `SKELETON_CONNECTIONS` для 46 точек
- Маппинг теперь 1:1 (убрать урезание с 46 до 20)
- Обновить `get_keypoint_color()` для 46 точек

### DOG-S8-2: Исправить AU коды
Файл: `packages/models/delta_action_units.py`
- УДАЛИТЬ: AU102, AU115, AU117, AU121 (не существуют в DogFACS)
- ИСПРАВИТЬ: AU116 → Lower Lip Depressor (не веко!), EAD102 → Ears Adductor (не Forward), AD37 → Lip Wipe, AD137 → Nose Lick
- ДОБАВИТЬ: AU143, AU145, AU109, AU110, AU118, AU25, AU27, AD33, AD35, EAD101, EAD104, EAD105
- Переписать геометрические вычисления под 46 keypoints

### DOG-S8-3/4: Эмоции
Файл: `packages/models/emotion.py`
- `EMOTION_CLASSES = ['happy','sad','angry','fearful','relaxed','neutral','surprise','pain','submission']`
- Добавить EmotionRule для каждой из 3 новых эмоций (из Mota-Rojas 2021)
- Исправить все AU коды в существующих правилах

---

## SPRINT 9 — Backend Annotation API
**Цель:** REST API для сохранения правок аннотаций

| ID | Задача | Приоритет | Labels |
|----|--------|-----------|--------|
| DOG-S9-1 | Реализовать SessionStore (хранение аннотаций в памяти + JSON) | Urgent | demo |
| DOG-S9-2 | PATCH /api/sessions/{id}/frames/{idx}/keypoints | Urgent | demo |
| DOG-S9-3 | PATCH /api/sessions/{id}/frames/{idx}/aus | Urgent | demo |
| DOG-S9-4 | PATCH /api/sessions/{id}/frames/{idx}/emotion | Urgent | demo |
| DOG-S9-5 | PATCH /api/sessions/{id}/frames/{idx}/breed | High | demo |
| DOG-S9-6 | POST /api/sessions/{id}/frames/{idx}/recompute_aus | High | demo |
| DOG-S9-7 | POST /api/sessions/{id}/frames/{idx}/recompute_emotion | High | demo |
| DOG-S9-8 | GET /api/sessions/{id}/frames — список кадров с аннотациями | High | demo |
| DOG-S9-9 | Реализовать полный COCO export (сейчас заглушка) | High | data |
| DOG-S9-10 | POST /api/sessions/{id}/add_frame — добавить кадр вручную | Medium | demo |

**Детали:**

### SessionStore
```
apps/webapp/backend/sessions/{session_id}/
    metadata.json     # видео, настройки, timestamp
    annotations.json  # все аннотации в расширенном COCO формате
    frames/           # JPEG кадры
```
COCO аннотация включает: bbox, 46 keypoints, 21 AU (intensity 0-5), breed, emotion (9 классов),
`annotation_status` (auto/reviewed/verified), `source` (ai/manual), `neutral_frame_id`

---

## SPRINT 10 — Annotation Editor Frontend
**Цель:** Полноценный редактор аннотаций в React

| ID | Задача | Приоритет | Labels |
|----|--------|-----------|--------|
| DOG-S10-1 | Обновить TypeScript типы (46 keypoints, 21 AU, 9 эмоций) | Urgent | demo |
| DOG-S10-2 | KeypointEditor — canvas drag&drop редактор 46 точек | Urgent | demo |
| DOG-S10-3 | AUPanel — слайдеры 0-5 для 21 AU с группировкой | Urgent | demo |
| DOG-S10-4 | EmotionSelector — выбор из 9 эмоций с подсказками AU | High | demo |
| DOG-S10-5 | BreedPicker — поиск породы из 120+ вариантов | High | demo |
| DOG-S10-6 | DogSelector — переключение между собаками в кадре | High | demo |
| DOG-S10-7 | AnnotationStatusBadge — авто/проверено/верифицировано | Medium | demo |
| DOG-S10-8 | Обновить PeakFrameCard — вкладки [Keypoints][AU][Emotion][Breed] | High | demo |
| DOG-S10-9 | Обновить Zustand store — синхронизация с backend API | High | demo |

**Детали KeypointEditor:**
- `<canvas>` поверх изображения собаки
- 46 точек цветными кружками по анатомическим группам
- Drag&drop для перемещения (mousedown + mousemove)
- ПКМ — toggle visibility точки
- Skeleton connections между точками
- Кнопка "Пересчитать AU" → POST /recompute_aus

**Детали AUPanel:**
- Группы: Верхнее лицо (AU101, AU143, AU145) / Нижнее лицо (8 AU) / Дескрипторы (5 AD) / Уши (5 EAD)
- Каждый AU: название на русском, код, слайдер 0-5, активность
- Кнопка "Пересчитать эмоцию" → POST /recompute_emotion

---

## SPRINT 11 — Multi-dog + Pipeline fixes
**Цель:** Исправить pipeline для всех собак в кадре

| ID | Задача | Приоритет | Labels |
|----|--------|-----------|--------|
| DOG-S11-1 | Проверить и исправить multi-dog в inference.py | Urgent | pipeline |
| DOG-S11-2 | Добавить dog_index ко всем аннотациям | High | pipeline |
| DOG-S11-3 | Фильтр по min_bbox_area (убрать мелкие детекции) | Medium | pipeline |
| DOG-S11-4 | E2E тест: видео с несколькими собаками | High | pipeline |
| DOG-S11-5 | Обновить COCO export для multi-dog | High | data |

---

## SPRINT 12 — Качество Keypoints (Оценка)
**Цель:** Оценить и улучшить качество определения ключевых точек

| ID | Задача | Приоритет | Labels |
|----|--------|-----------|--------|
| DOG-S12-1 | Собрать 50-100 тестовых кадров через annotation editor | High | data |
| DOG-S12-2 | Оценить качество текущей SimpleBaseline модели | High | model |
| DOG-S12-3 | Сравнение: SimpleBaseline vs ViTPose-Small | Medium | model, research |
| DOG-S12-4 | Если плохо — настроить ViTPose для 46 dog keypoints | Medium | model |
| DOG-S12-5 | Скрипт для тренировки на своих данных | Medium | model |

---

## SPRINT 13 — Сбор данных и разметка AU
**Цель:** Создать обучающую выборку для AU neural network

| ID | Задача | Приоритет | Labels |
|----|--------|-----------|--------|
| DOG-S13-1 | Скачать 50+ видео с YouTube (разные породы, эмоции) | Urgent | data |
| DOG-S13-2 | Прогнать через pipeline → auto-аннотации | High | data |
| DOG-S13-3 | Вручную разметить AU для 200+ кадров через editor | High | data |
| DOG-S13-4 | Контроль качества: inter-annotator agreement | Medium | data |
| DOG-S13-5 | Экспорт финального датасета в COCO формате | High | data |

---

## SPRINT 14 — AU Neural Network
**Цель:** Заменить геометрическую аппроксимацию нейросетью

| ID | Задача | Приоритет | Labels |
|----|--------|-----------|--------|
| DOG-S14-1 | Подготовить датасет keypoints→AU для обучения | Urgent | model |
| DOG-S14-2 | Реализовать MLP: 138 вход → 21 AU выход | Urgent | model |
| DOG-S14-3 | Обучить и оценить baseline MLP | High | model |
| DOG-S14-4 | Интегрировать в pipeline (заменить геометрию) | High | pipeline, model |
| DOG-S14-5 | Опционально: гибрид image+keypoints (Approach C) | Low | model, research |

---

## Порядок спринтов

```
Sprint 8 (сейчас) → Sprint 9 → Sprint 10
                                    ↓
                              Sprint 11 (multi-dog)
                                    ↓
                              Sprint 12 (keypoints quality)
                                    ↓
                              Sprint 13 (data collection)
                                    ↓
                              Sprint 14 (AU neural net)
```

**Sprint 8, 9, 10 можно делать параллельно** (разные люди):
- Sprint 8: backend developer
- Sprint 9: backend developer
- Sprint 10: frontend developer
