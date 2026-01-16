# COCO Format Specification - Dog FACS Dataset

**Data:** 2025-01-16
**Autor:** Danylo Lohachov (U1), z pomocą Claude AI
**Sprint:** 1 - Project Setup
**Story:** 1.2 - DogFACS & COCO Research

---

## 1. Wprowadzenie do COCO Format

### 1.1 Czym jest COCO?

**COCO (Common Objects in Context)** to standardowy format JSON używany w machine learning do anotacji obrazów. Został wprowadzony przez Microsoft w 2015 roku i stał się de facto standardem dla:
- Object Detection (bounding boxes)
- Keypoint Detection
- Segmentation
- Image Captioning

### 1.2 Dlaczego COCO dla Dog FACS Dataset?

- Szeroko wspierany przez biblioteki ML (PyTorch, TensorFlow)
- Kompatybilny z narzędziami ewaluacji (pycocotools)
- Umożliwia przechowywanie wielu typów anotacji w jednym pliku
- Łatwy do rozszerzenia o custom kategorie

---

## 2. Struktura JSON - Przegląd

### 2.1 Top-Level Structure

```json
{
  "info": { ... },
  "licenses": [ ... ],
  "images": [ ... ],
  "annotations": [ ... ],
  "categories": [ ... ]
}
```

### 2.2 Szczegółowy opis sekcji

#### Info Section
```json
"info": {
  "description": "Dog FACS Dataset - Automatyczna anotacja emocji psów",
  "url": "https://github.com/eternaki/group-project",
  "version": "1.0",
  "year": 2025,
  "contributor": "Politechnika Gdańska - WETI",
  "date_created": "2025-01-16"
}
```

#### Licenses Section
```json
"licenses": [
  {
    "id": 1,
    "name": "CC BY-NC 4.0",
    "url": "https://creativecommons.org/licenses/by-nc/4.0/"
  }
]
```

#### Images Section
```json
"images": [
  {
    "id": 1,
    "file_name": "video001_frame_0001.jpg",
    "width": 1920,
    "height": 1080,
    "date_captured": "2025-01-16T12:00:00",
    "license": 1,
    "source_video": "youtube_abc123",
    "frame_number": 1
  }
]
```

#### Categories Section
```json
"categories": [
  {
    "id": 1,
    "name": "dog",
    "supercategory": "animal",
    "keypoints": [
      "left_eye_inner", "left_eye_outer",
      "right_eye_inner", "right_eye_outer",
      "nose_tip", "nose_left", "nose_right",
      "left_ear_base", "left_ear_tip",
      "right_ear_base", "right_ear_tip",
      "mouth_left", "mouth_right",
      "upper_lip", "lower_lip", "chin",
      "left_brow", "right_brow",
      "forehead", "muzzle_center"
    ],
    "skeleton": [
      [0, 1], [2, 3], [0, 2],
      [4, 5], [4, 6],
      [7, 8], [9, 10],
      [11, 13], [12, 13], [11, 14], [12, 14], [14, 15],
      [16, 18], [17, 18],
      [4, 19], [19, 13]
    ]
  }
]
```

---

## 3. Annotations - Szczegółowa Specyfikacja

### 3.1 Annotation Structure

Każda anotacja zawiera wszystkie informacje o jednym wykrytym psie:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [100, 50, 400, 300],
  "area": 120000,
  "iscrowd": 0,
  "keypoints": [
    150, 120, 2,  // left_eye_inner (x, y, visibility)
    180, 120, 2,  // left_eye_outer
    250, 120, 2,  // right_eye_inner
    280, 120, 2,  // right_eye_outer
    215, 180, 2,  // nose_tip
    200, 175, 2,  // nose_left
    230, 175, 2,  // nose_right
    120, 80, 2,   // left_ear_base
    100, 40, 2,   // left_ear_tip
    310, 80, 2,   // right_ear_base
    330, 40, 2,   // right_ear_tip
    170, 220, 2,  // mouth_left
    260, 220, 2,  // mouth_right
    215, 200, 2,  // upper_lip
    215, 240, 2,  // lower_lip
    215, 270, 2,  // chin
    160, 100, 2,  // left_brow
    270, 100, 2,  // right_brow
    215, 90, 2,   // forehead
    215, 190, 2   // muzzle_center
  ],
  "num_keypoints": 20,
  "breed": "labrador_retriever",
  "breed_confidence": 0.95,
  "emotion": "happy",
  "emotion_id": 0,
  "emotion_confidence": 0.87
}
```

### 3.2 Bounding Box Format

Format: `[x, y, width, height]`
- `x`, `y` - współrzędne lewego górnego rogu
- `width`, `height` - wymiary bounding box

```json
"bbox": [100, 50, 400, 300]
// x=100, y=50, width=400, height=300
```

### 3.3 Keypoints Format

Format: `[x1, y1, v1, x2, y2, v2, ...]`

Każdy keypoint to 3 wartości:
- `x` - współrzędna X
- `y` - współrzędna Y
- `v` - visibility flag:
  - **0** = nie oznaczony
  - **1** = oznaczony ale niewidoczny (zasłonięty)
  - **2** = oznaczony i widoczny

### 3.4 Custom Fields (rozszerzenia Dog FACS)

Dodatkowe pola specyficzne dla naszego datasetu:

| Pole | Typ | Opis |
|------|-----|------|
| `breed` | string | Nazwa rasy psa |
| `breed_confidence` | float | Pewność klasyfikacji rasy (0-1) |
| `emotion` | string | Kategoria emocji |
| `emotion_id` | int | ID emocji (0-5) |
| `emotion_confidence` | float | Pewność klasyfikacji emocji (0-1) |

---

## 4. Emotion Categories

### 4.1 Definicja kategorii emocji

```json
"emotion_categories": [
  {"id": 0, "name": "happy", "name_pl": "szczęśliwy"},
  {"id": 1, "name": "sad", "name_pl": "smutny"},
  {"id": 2, "name": "angry", "name_pl": "zły"},
  {"id": 3, "name": "fearful", "name_pl": "przestraszony"},
  {"id": 4, "name": "relaxed", "name_pl": "zrelaksowany"},
  {"id": 5, "name": "neutral", "name_pl": "neutralny"}
]
```

---

## 5. Kompletny Przykład JSON

```json
{
  "info": {
    "description": "Dog FACS Dataset - Automatyczna anotacja emocji psów",
    "url": "https://github.com/eternaki/group-project",
    "version": "1.0",
    "year": 2025,
    "contributor": "Politechnika Gdańska - WETI",
    "date_created": "2025-01-16"
  },
  "licenses": [
    {
      "id": 1,
      "name": "CC BY-NC 4.0",
      "url": "https://creativecommons.org/licenses/by-nc/4.0/"
    }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "video001_frame_0001.jpg",
      "width": 1920,
      "height": 1080,
      "date_captured": "2025-01-16T12:00:00",
      "license": 1,
      "source_video": "youtube_abc123",
      "frame_number": 1
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "dog",
      "supercategory": "animal",
      "keypoints": [
        "left_eye_inner", "left_eye_outer",
        "right_eye_inner", "right_eye_outer",
        "nose_tip", "nose_left", "nose_right",
        "left_ear_base", "left_ear_tip",
        "right_ear_base", "right_ear_tip",
        "mouth_left", "mouth_right",
        "upper_lip", "lower_lip", "chin",
        "left_brow", "right_brow",
        "forehead", "muzzle_center"
      ],
      "skeleton": [
        [0, 1], [2, 3], [0, 2],
        [4, 5], [4, 6],
        [7, 8], [9, 10],
        [11, 13], [12, 13], [11, 14], [12, 14], [14, 15],
        [16, 18], [17, 18],
        [4, 19], [19, 13]
      ]
    }
  ],
  "emotion_categories": [
    {"id": 0, "name": "happy", "name_pl": "szczęśliwy"},
    {"id": 1, "name": "sad", "name_pl": "smutny"},
    {"id": 2, "name": "angry", "name_pl": "zły"},
    {"id": 3, "name": "fearful", "name_pl": "przestraszony"},
    {"id": 4, "name": "relaxed", "name_pl": "zrelaksowany"},
    {"id": 5, "name": "neutral", "name_pl": "neutralny"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 50, 400, 300],
      "area": 120000,
      "iscrowd": 0,
      "keypoints": [
        150, 120, 2, 180, 120, 2, 250, 120, 2, 280, 120, 2,
        215, 180, 2, 200, 175, 2, 230, 175, 2,
        120, 80, 2, 100, 40, 2, 310, 80, 2, 330, 40, 2,
        170, 220, 2, 260, 220, 2, 215, 200, 2, 215, 240, 2, 215, 270, 2,
        160, 100, 2, 270, 100, 2, 215, 90, 2, 215, 190, 2
      ],
      "num_keypoints": 20,
      "breed": "labrador_retriever",
      "breed_confidence": 0.95,
      "emotion": "happy",
      "emotion_id": 0,
      "emotion_confidence": 0.87
    }
  ]
}
```

---

## 6. Walidacja z pycocotools

### 6.1 Instalacja

```bash
pip install pycocotools
```

### 6.2 Przykład walidacji

```python
from pycocotools.coco import COCO

# Wczytanie datasetu
coco = COCO('annotations.json')

# Sprawdzenie obrazów
print(f"Liczba obrazów: {len(coco.imgs)}")

# Sprawdzenie anotacji
print(f"Liczba anotacji: {len(coco.anns)}")

# Pobranie anotacji dla obrazu
img_id = 1
ann_ids = coco.getAnnIds(imgIds=[img_id])
anns = coco.loadAnns(ann_ids)

for ann in anns:
    print(f"BBox: {ann['bbox']}")
    print(f"Keypoints: {ann['num_keypoints']}")
    print(f"Emotion: {ann.get('emotion', 'N/A')}")
```

### 6.3 Wizualizacja

```python
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import cv2

coco = COCO('annotations.json')
img_info = coco.loadImgs([1])[0]
img = cv2.imread(img_info['file_name'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
ann_ids = coco.getAnnIds(imgIds=[1])
anns = coco.loadAnns(ann_ids)
coco.showAnns(anns)
plt.show()
```

---

## 7. Organizacja Plików

### 7.1 Struktura katalogów

```
data/
├── raw/                          # Pobrane filmy YouTube
│   └── video_001.mp4
├── frames/                       # Wyekstrahowane klatki
│   └── video_001/
│       ├── frame_0001.jpg
│       ├── frame_0002.jpg
│       └── ...
└── annotations/                  # Pliki COCO JSON
    ├── train.json               # Anotacje treningowe
    ├── val.json                 # Anotacje walidacyjne
    ├── test.json                # Anotacje testowe
    └── full_dataset.json        # Wszystkie anotacje
```

### 7.2 Podział datasetu

| Split | Procent | Cel |
|-------|---------|-----|
| Train | 70% | Trening modeli |
| Val | 15% | Walidacja podczas treningu |
| Test | 15% | Finalna ewaluacja |

---

## 8. Źródła

1. [COCO Dataset Guide - V7 Labs](https://www.v7labs.com/blog/coco-dataset-guide) - Kompletny przewodnik
2. [COCO JSON Format - Roboflow](https://roboflow.com/formats/coco-json) - Specyfikacja formatu
3. [COCO Official - cocodataset.org](https://cocodataset.org/#format-data) - Oficjalna dokumentacja
4. [pycocotools - GitHub](https://github.com/cocodataset/cocoapi) - Biblioteka do pracy z COCO

---

*Dokument wygenerowany w ramach projektu Dog FACS Dataset dla Politechniki Gdańskiej (WETI).*
