# 🎯 ПЛАН РЕАЛИЗАЦИИ: DogFACS Action Units → Emotions

**Проект:** Dog FACS Dataset  
**Задача:** Реализовать определение эмоций собак (6 класс) на основе DogFACS Action Units  
**Научная база:** Mota-Rojas et al. 2021  
**Дата:** 24 января 2026

---

## ФАЗА 1: Head Pose Estimation

**Файл:** `packages/models/head_pose.py` (НОВЫЙ)

```python
# Структура:
# 1. @dataclass HeadPose(yaw, pitch, roll, is_frontal, confidence)
# 2. def estimate_head_pose(keypoints: list[Keypoint]) -> HeadPose
#    - Вычислить yaw, pitch, roll из характерных точек
#    - Установить is_frontal = True если |yaw|<30 AND |pitch|<30 AND |roll|<30
#    - confidence = mean(visibility всех keypoints)
# 3. def validate_head_pose(pose: HeadPose) -> bool
```

**Алгоритм:**
- YAW: atan2(nose.x - eye_center.x, eye_width)
- PITCH: atan2(nose.y - ear_center.y, ear_span)
- ROLL: atan2(right_ear.y - left_ear.y, ear_width)

---

## ФАЗА 2: Action Units Computation

**Файл:** `packages/models/dogfacs.py` (НОВЫЙ)

```python
# Структура:
# 1. @dataclass ActionUnits(au101, au145, au25, au26, au301, au401, ead102, ead103, ead104, ad137)
# 2. def compute_action_units(keypoints: list[Keypoint]) -> ActionUnits

# 10 AU (каждый ∈ [0, 1]):
# - AU101: Inner Brow Raiser = (forehead.y - brow.y) / baseline
# - AU145: Blink = 1.0 - eye_opening_ratio
# - AU25: Lips Part = mouth_opening / baseline
# - AU26: Jaw Drop = max(0, (jaw_dist / baseline - 1.2))
# - AU301: Nose Wrinkler = 1.0 - (nose_width / nose_depth)
# - AU401: Upper Lip Raiser = upper_lift / baseline
# - EAD102: Ears Forward = avg(ear_forward_vector)
# - EAD103: Ears Flattener = 1.0 - (ear_height / baseline)
# - EAD104: Ears Rotator = ear_angle_diff / pi
# - AD137: Nose Lick = 1.0 - (nose_to_mouth_dist / mouth_width)

# Нормализировать все к [0, 1]
```

---

## ФАЗА 3: Temporal Aggregation

**Файл:** `packages/pipeline/temporal_processor.py` (НОВЫЙ)

```python
# Структуры:
# 1. class TemporalAUBuffer(window_size=30)
#    - au_history: deque(maxlen=30)
#    - add_frame(aus, confidence)
#    - get_aggregated_au() → ActionUnits (mean за 30 кадров)
#    - get_au_variance() → dict (variance каждого AU)
#    - is_stable(threshold=0.15) → bool

# 2. class TemporalProcessor(window_size=30, head_pose_threshold=30)
#    - process_frame(keypoints, head_pose) → Optional[dict]
#      * ФИЛЬТР: head_pose.is_frontal == False → skip
#      * ФИЛЬТР: mean_visibility < 0.5 → skip
#      * ФИЛЬТР: critical keypoints visibility < 0.3 → skip
#      * Вычислить AU, добавить в буфер
#      * Вернуть aggregated AU если stable
#    - process_video_sequence(keypoints_list, pose_list) → ActionUnits
```

---

## ФАЗА 4: Emotion Classification

**Файл:** `packages/models/emotion.py` (ПЕРЕПИСАТЬ)

```python
# Изменения:
# 1. Заменить EMOTION_CLASSES с 4 на 6 класс:
#    ['sad', 'angry', 'relaxed', 'happy', 'fearful', 'neutral']
#
# 2. @dataclass EmotionPrediction(emotion, emotion_id, confidence, probabilities, au_scores)
#
# 3. def classify_emotion_from_au(aus: ActionUnits) -> EmotionPrediction
#    Scoring (каждый ∈ [0, 1]):
#
#    happy_score = (
#        au['AU25'] * 0.35 +           # Рот открыт
#        au['EAD102'] * 0.25 +          # Уши вперед
#        au['AU101'] * 0.15 +           # Брови подняты
#        (1 - au['EAD103']) * 0.15 +   # Уши не плоские
#        (1 - au['AD137']) * 0.10      # Нет стресса
#    )
#
#    sad_score = (
#        au['EAD103'] * 0.40 +          # Уши плоские
#        au['AU145'] * 0.15 +           # Моргание
#        (1 - au['AU101']) * 0.15 +    # Брови не подняты
#        (1 - au['AU25']) * 0.15 +     # Рот закрыт
#        au['AD137'] * 0.15             # Лизание
#    )
#
#    angry_score = (
#        ((au['AU25'] + au['AU26']) / 2) * 0.30 +  # Рот открыт+челюсть
#        au['AU401'] * 0.25 +            # Оскал зубов
#        au['AU301'] * 0.15 +            # Нос сморщен
#        ((au['EAD103'] + au['EAD104']) / 2) * 0.15 +  # Уши в позиции
#        au['AU145'] * 0.15              # Тенсе моргание
#    )
#
#    fearful_score = (
#        au['EAD103'] * 0.30 +           # Уши плоские
#        au['AD137'] * 0.25 +            # Облизывание (ГЛАВНЫЙ индикатор стресса)
#        au['AU145'] * 0.20 +            # Частое моргание
#        au['AU101'] * 0.12 +            # Брови подняты
#        (1 - au['AU25']) * 0.13         # Рот закрыт
#    )
#
#    relaxed_score = (
#        (1 - sum(au.values())/len(au)) * 0.50 +  # Минимум активации
#        (1 - au['EAD103']) * 0.15 +    # Уши не плоские
#        (1 - au['EAD102']) * 0.15 +    # Уши не вперед
#        (1 - au['AD137']) * 0.10 +     # Нет стресса
#        (1 - au['AU301']) * 0.10       # Нос не напружен
#    )
#
#    neutral_score = (
#        (1 - sum(au.values())/len(au)) * 0.70 +
#        (1 - (au['AU25'] + au['AU26']) / 2) * 0.15 +
#        (1 - (au['EAD103'] + au['EAD102']) / 2) * 0.15
#    )
#
# 4. Нормализировать scores → probabilities (softmax-like)
# 5. Выбрать best_emotion = max(scores)
# 6. Вернуть EmotionPrediction с probabilities для всех 6
```

---

## ФАЗА 5: Pipeline Integration

**Файл:** `packages/pipeline/inference.py` (МОДИФИКАЦИЯ)

```python
# Добавить импорты:
from packages.models.head_pose import estimate_head_pose
from packages.models.dogfacs import compute_action_units
from packages.pipeline.temporal_processor import TemporalProcessor
from packages.models.emotion import classify_emotion_from_au

# В класс InferencePipeline добавить:
self.temporal_processor = TemporalProcessor(window_size=30, head_pose_threshold=30)

# Добавить метод:
def process_video_sequence(self, video_frames: list, stride: int = 1):
    """
    Обработить видео последовательность.
    
    Процесс:
    1. Для каждого кадра (с stride):
       - BBox detection
       - Crop dog region
       - Keypoints detection
       - Head pose estimation
       - Добавить в temporal_processor
    2. Вернуть final emotion prediction
    """
```

---

## РЕАЛИЗАЦИЯ: Step-by-Step

### 1. Head Pose (`packages/models/head_pose.py`)
- [ ] Создать файл
- [ ] Написать `HeadPose` dataclass
- [ ] Написать `estimate_head_pose(keypoints)` - вычислить yaw, pitch, roll
- [ ] Написать `validate_head_pose(pose)`

### 2. Action Units (`packages/models/dogfacs.py`)
- [ ] Создать файл
- [ ] Написать `ActionUnits` dataclass (10 AU)
- [ ] Написать `compute_action_units(keypoints)` - вычислить все 10 AU

### 3. Temporal Processor (`packages/pipeline/temporal_processor.py`)
- [ ] Создать файл
- [ ] Написать `TemporalAUBuffer` - история + агрегация
- [ ] Написать `TemporalProcessor` - фильтрование + обработка

### 4. Emotion Classification (`packages/models/emotion.py`)
- [ ] Заменить EMOTION_CLASSES на 6 класс
- [ ] Переписать `classify_emotion_from_au()` с новыми правилами
- [ ] Тестировать каждую эмоцию

### 5. Integration (`packages/pipeline/inference.py`)
- [ ] Добавить импорты
- [ ] Добавить `self.temporal_processor`
- [ ] Добавить метод `process_video_sequence()`

---

## КЛЮЧЕВЫЕ ДЕТАЛИ

**Head Pose Filtering:**
- Пропускать кадры если |yaw| > 30° или |pitch| > 30° или |roll| > 30°
- Также пропускать если mean_visibility < 0.5
- Пропускать если критические keypoints не видны (visibility < 0.3)

**Temporal Aggregation:**
- Буфер 30 кадров = 1 секунда @ 30 FPS
- Усреднять AU каждый кадр
- Проверять stability: variance < 0.15
- Только стабильные AU возвращать

**Emotion Scoring:**
- Каждая эмоция = weighted sum of AU
- Нормализировать к [0, 1]
- Использовать softmax для probabilities

**6 Классов (новое):**
- happy: AU25 + EAD102
- sad: EAD103 + низкая активация
- angry: AU25+AU26 + AU401
- fearful: EAD103 + AD137 (облизывание - главный стресс индикатор)
- relaxed: минимум активации
- neutral: baseline

---

## NOTES

1. **20 keypoints** уже используются в коде (DogFLW subset)
2. **Научная база:** Mota-Rojas et al. 2021 - Таблица 2-3
3. **Timeline:** ~11 часов работы
4. **Testing:** unit tests для каждого модуля
5. **Compatibility:** все новое встраивается в существующий pipeline

---

Все готово! Claude может сразу начинать реализацию! 🚀
