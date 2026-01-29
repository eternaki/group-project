# DTP - Dokumentacja Techniczna Projektu

> Skopiuj zawartość do szablonu: `template/PG_WETI_DTP_wer. 1.00.doc`

---

## Informacje podstawowe

| Pole | Wartość |
|------|---------|
| **Tytuł projektu** | Dog FACS Dataset - Pipeline do automatycznej anotacji emocji psów |
| **Katedra** | Katedra Systemów Decyzyjnych i Robotyki |
| **Opiekun projektu** | dr hab. inż. Michał Czubenko |
| **Klient** | dr hab. inż. Michał Czubenko (klient wewnętrzny PG) |
| **Rodzaj projektu** | Projekt politechniczny - klient wewnętrzny z PG |
| **Rok akademicki** | 2025/2026 |
| **Semestr** | I |
| **Gałąź repozytorium** | `feature/dogfacs-rule-based` |

### Zespół projektowy

| Imię i nazwisko | Nr albumu | Kierunek | Rola w projekcie |
|-----------------|-----------|----------|------------------|
| Danylo Lohachov | 196610 | Informatyka | Kierownik projektu / Dokumentacja / QA / Frontend |
| Anton Shkrebela | 196637 | Informatyka | AI/ML Specialist (Keypoints & DogFACS) |
| Danylo Zherzdiev | 196765 | Informatyka | Backend (BBox & Breed models, Pipeline, COCO) |
| Mariia Volkova | 196660 | Informatyka | Data Engineer (Data collection & Manual verification) |

---

## 1. Wprowadzenie - o dokumencie

### 1.1 Cel dokumentu

Niniejszy dokument stanowi Dokumentację Techniczną Projektu (DTP) dla systemu **Dog FACS Dataset**. Celem dokumentu jest przedstawienie szczegółowych informacji technicznych dotyczących produktu, jego architektury, funkcjonalności, parametrów technicznych oraz wyników działania.

### 1.2 Zakres dokumentu

Dokument obejmuje:
- Opis architektury systemu
- Specyfikację funkcjonalną
- Parametry techniczne modeli AI
- Schematy blokowe pipeline'u
- Opis oprogramowania i algorytmów
- Specyfikację systemu DogFACS Rule Engine
- Instrukcję użytkowania

### 1.3 Odbiorcy

Dokument jest przeznaczony dla:
- Opiekuna projektu
- Komisji oceniającej projekt grupowy
- Osób zainteresowanych rozwojem lub użytkowaniem systemu
- Przyszłych zespołów kontynuujących projekt

### 1.4 Terminologia

| Termin | Definicja |
|--------|-----------|
| **DogFACS** | Dog Facial Action Coding System - naukowy system kodowania mimiki psów |
| **COCO** | Common Objects in Context - format anotacji obrazów |
| **Keypoints** | Punkty kluczowe - charakterystyczne punkty anatomiczne (20 punktów) |
| **BBox** | Bounding Box - prostokąt obejmujący obiekt na obrazie |
| **Action Unit (AU)** | Jednostka akcji mimicznej - obiektywna miara ruchu mięśni twarzy |
| **Delta AU** | Różnica AU względem neutralnej klatki bazowej |
| **Neutral Frame** | Klatka referencyjna z neutralnym wyrazem twarzy psa |
| **Peak Frame** | Klatka z maksymalną ekspresją emocji (wysoki TFM) |
| **TFM** | Total Facial Movement - suma wszystkich aktywacji AU |
| **Rule-based** | Klasyfikacja oparta na regułach naukowych, bez ML |
| **Poselet** | Kombinacja AU charakterystyczna dla emocji |

---

## 2. Dokumentacja techniczna projektu

### 2.1 Opis produktu

System **Dog FACS Dataset** to pipeline do automatycznej anotacji emocji psów wykorzystujący:
- **3 modele deep learning** (detekcja, klasyfikacja ras, keypoints)
- **Rule-based DogFACS engine** do klasyfikacji emocji (BEZ ML)

System przetwarza wideo i generuje anotacje w formacie COCO zawierające:
- **Bounding boxes** - prostokąty obejmujące wykryte psy
- **Klasyfikację ras** - identyfikacja rasy psa (50+ ras)
- **Punkty kluczowe twarzy** - 20 punktów anatomicznych
- **Action Units** - 12 jednostek akcji mimicznej DogFACS
- **Etykiety emocji** - 6 klas (happy, sad, angry, fearful, relaxed, neutral)

### 2.2 Architektura systemu

#### 2.2.1 Schemat blokowy systemu

```
┌─────────────────────────────────────────────────────────────────┐
│                   Dog FACS Dataset Generator                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │  Video   │───▶│ Frame        │───▶│   Neutral Frame       │ │
│  │  Input   │    │ Extraction   │    │   Detection           │ │
│  └──────────┘    └──────────────┘    └───────────────────────┘ │
│                                                │                │
│                                                ▼                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    AI Pipeline                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌───────────────────┐   │  │
│  │  │  BBox      │─▶│  Breed     │  │  Keypoints        │   │  │
│  │  │  (YOLOv8m) │  │(EffNetB4)  │  │  (SimpleBaseline) │   │  │
│  │  └────────────┘  └────────────┘  └─────────┬─────────┘   │  │
│  │                                            │              │  │
│  │  ┌─────────────────────────────────────────▼───────────┐ │  │
│  │  │           Delta Action Units Extractor              │ │  │
│  │  │  (neutral vs target frame comparison)               │ │  │
│  │  └─────────────────────────────────────────┬───────────┘ │  │
│  │                                            │              │  │
│  │  ┌─────────────────────────────────────────▼───────────┐ │  │
│  │  │           DogFACS Rule Engine (NO ML)               │ │  │
│  │  │  AU → Poselet Matching → Emotion (6 classes)        │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                │                │
│                       ┌────────────────────────┘                │
│                       ▼                                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Peak Frame Selector (TFM-based)              │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                │                │
│                                                ▼                │
│  ┌───────────────────────┐    ┌──────────────────────────────┐ │
│  │   COCO Exporter       │───▶│   Dataset (annotations/)     │ │
│  └───────────────────────┘    └──────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │     Applications: Streamlit Demo | React+FastAPI Webapp   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2.2 Pipeline przetwarzania wideo (Dataset Generation)

```
┌─────────────────────┐
│ Video Input         │
│ (MP4, AVI, MOV)     │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────┐
│ [1/6] Keypoints Detection│
│ dla wszystkich klatek    │
│ (SimpleBaseline ResNet34)│
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ [2/6] Head Pose          │
│ Estimation               │
│ (filtrowanie nie-front.) │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ [3/6] Neutral Frame      │
│ Auto-Detection           │
│ (min AU activation)      │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ [4/6] Delta AU           │
│ Extraction               │
│ (target vs neutral)      │
│ 12 official DogFACS codes│
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ [5/6] Peak Frame         │
│ Selection (TFM metric)   │
│ + temporal separation    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ [6/6] Emotion            │
│ Classification           │
│ (Rule-based DogFACS)     │
│ Priority poselet match   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ COCO JSON Export         │
│ + Peak Frame Images      │
└──────────────────────────┘
```

#### 2.2.3 Architektura warstw

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Streamlit    │  │ React        │  │ Verification Tool      │ │
│  │ Demo App     │  │ + FastAPI    │  │ (apps/verification/)   │ │
│  │(apps/demo/)  │  │(apps/webapp/)│  │                        │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     PIPELINE LAYER                               │
│                  (packages/pipeline/)                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ inference.py │ │ neutral_     │ │ peak_selector.py         │ │
│  │ (main pipe)  │ │ frame.py     │ │ (TFM-based selection)    │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
│  ┌──────────────┐ ┌──────────────┐                              │
│  │ video.py     │ │ temporal_    │                              │
│  │ (extraction) │ │ processor.py │                              │
│  └──────────────┘ └──────────────┘                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      MODELS LAYER                                │
│                   (packages/models/)                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌───────────────┐ │
│  │ bbox.py    │ │ breed.py   │ │keypoints.py│ │ emotion.py    │ │
│  │ (YOLOv8m)  │ │(EffNetB4)  │ │(SBaseline) │ │ (Rule-based)  │ │
│  └────────────┘ └────────────┘ └────────────┘ └───────────────┘ │
│  ┌────────────────────┐ ┌──────────────────────────────────────┐│
│  │ action_units.py    │ │ delta_action_units.py                ││
│  │ (absolute AU)      │ │ (delta AU vs neutral)                ││
│  └────────────────────┘ └──────────────────────────────────────┘│
│  ┌────────────────────┐                                         │
│  │ head_pose.py       │                                         │
│  │ (pose estimation)  │                                         │
│  └────────────────────┘                                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                       DATA LAYER                                 │
│                    (packages/data/)                              │
│  ┌──────────────┐ ┌──────────────┐                              │
│  │ coco.py      │ │ schemas.py   │                              │
│  │(COCO format) │ │(data classes)│                              │
│  └──────────────┘ └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Specyfikacja modeli AI

#### 2.3.1 Model detekcji psów (BBox)

| Parametr | Wartość |
|----------|---------|
| **Plik wag** | `models/yolov8m.pt` |
| **Rozmiar pliku** | 52.1 MB |
| **Architektura** | YOLOv8m |
| **Rozmiar wejściowy** | 640×640 px |
| **Wyjście** | Lista bounding boxes z confidence score |
| **Klasy** | Filtrowanie tylko klasy "dog" |
| **Confidence threshold** | 0.3 (domyślnie) |
| **Max detections** | 10 psów per obraz |

#### 2.3.2 Model klasyfikacji ras (Breed)

| Parametr | Wartość |
|----------|---------|
| **Plik wag** | `models/breed.pt` |
| **Rozmiar pliku** | 71.8 MB |
| **Architektura** | EfficientNet-B4 |
| **Rozmiar wejściowy** | 224×224 px |
| **Wyjście** | Top-5 ras z prawdopodobieństwami |
| **Liczba klas** | 50+ ras |
| **Baza wag** | ImageNet pretrained |
| **Dane treningowe** | Stanford Dogs Dataset |

#### 2.3.3 Model detekcji keypoints

| Parametr | Wartość |
|----------|---------|
| **Plik wag** | `models/keypoints_dogflw.pt` |
| **Rozmiar pliku** | 102.1 MB |
| **Architektura** | SimpleBaseline (ResNet34 + Deconv Head) |
| **Rozmiar wejściowy** | 256×256 px |
| **Rozmiar heatmap** | 64×64 px |
| **Keypoints DogFLW** | 46 punktów (natywnie) |
| **Keypoints projektu** | 20 punktów (po mapowaniu) |
| **TTA** | Test-Time Augmentation (flip + average) |
| **Dane treningowe** | DogFLW Dataset (Kaggle) |

**Lista 20 keypoints projektu:**

| ID | Nazwa | Opis |
|----|-------|------|
| 0 | left_eye | Środek lewego oka |
| 1 | right_eye | Środek prawego oka |
| 2 | nose | Czubek nosa |
| 3 | left_ear_base | Podstawa lewego ucha |
| 4 | right_ear_base | Podstawa prawego ucha |
| 5 | left_ear_tip | Czubek lewego ucha |
| 6 | right_ear_tip | Czubek prawego ucha |
| 7 | left_mouth_corner | Lewy kącik ust |
| 8 | right_mouth_corner | Prawy kącik ust |
| 9 | upper_lip | Środek górnej wargi |
| 10 | lower_lip | Środek dolnej wargi |
| 11 | chin | Podbródek |
| 12 | left_cheek | Lewy policzek |
| 13 | right_cheek | Prawy policzek |
| 14 | forehead | Środek czoła |
| 15 | left_eyebrow | Lewa brew |
| 16 | right_eyebrow | Prawa brew |
| 17 | muzzle_top | Góra pyska |
| 18 | muzzle_left | Lewa strona pyska |
| 19 | muzzle_right | Prawa strona pyska |

#### 2.3.4 Klasyfikacja emocji (DogFACS Rule Engine)

| Parametr | Wartość |
|----------|---------|
| **Typ** | **Rule-based (BEZ ML)** |
| **Plik wag** | BRAK - nie wymaga treningu |
| **Wejście** | Delta Action Units (12 wartości) |
| **Wyjście** | 6 klas emocji + confidence |
| **Metoda** | Priority poselet matching |
| **Baza naukowa** | Mota-Rojas et al. 2021, animalfacs.com |

**Klasy emocji:**

| ID | Emocja | Priorytet reguły |
|----|--------|------------------|
| 0 | happy | 100 |
| 1 | sad | 85 |
| 2 | angry | 95 |
| 3 | fearful | 90 |
| 4 | relaxed | 70 |
| 5 | neutral | 50 (fallback) |

### 2.4 System DogFACS Action Units

#### 2.4.1 Lista oficjalnych Action Units

System implementuje 12 oficjalnych kodów DogFACS:

| Kod AU | Nazwa | Opis |
|--------|-------|------|
| AU101 | Inner Brow Raiser | Podniesienie wewnętrznej brwi |
| AU102 | Outer Brow Raiser | Podniesienie zewnętrznej brwi |
| AU12 | Lip Corner Puller | Pociągnięcie kącików ust (uśmiech) |
| AU115 | Upper Eyelid Raiser | Podniesienie górnej powieki |
| AU116 | Lower Eyelid Raiser | Zmrużenie dolnej powieki |
| AU117 | Closure of Eyelids | Zamknięcie oczu (mruganie) |
| AU121 | Eye Widener | Rozszerzenie oczu |
| EAD102 | Ears Forward | Uszy do przodu |
| EAD103 | Ears Flattener | Uszy spłaszczone/do tyłu |
| AD19 | Tongue Show | Pokazanie języka |
| AD37 | Nose Lick | Lizanie nosa |
| AU26 | Jaw Drop | Opadnięcie szczęki |

#### 2.4.2 Delta AU Extraction

**Zasada działania:**
```
Delta_AU = (distance_target / distance_neutral) - 1.0
```

Gdzie:
- `distance_target` - pomiar geometryczny na klatce docelowej
- `distance_neutral` - pomiar na neutralnej klatce bazowej
- Wynik > 0 oznacza aktywację AU, < 0 oznacza deaktywację

#### 2.4.3 Reguły klasyfikacji emocji (Poselets)

**HAPPY (priorytet 100):**
```
Wymagane: AU12 ≥ 1.20, EAD102 ≥ 1.10
Inhibitory: EAD103 < 1.10, AU26 < 1.25
Opcjonalne: AU101 ≥ 1.10 (bonus)
```

**ANGRY (priorytet 95):**
```
Wymagane: AU26 ≥ 1.25, AU12 ≥ 1.15
Inhibitory: EAD102 < 1.10
Opcjonalne: AU101, EAD103
```

**FEARFUL (priorytet 90):**
```
Wymagane: EAD103 ≥ 1.15, AU101 ≥ 1.10
Inhibitory: AU26 < 1.20
Opcjonalne: AD37, AU117
```

**SAD (priorytet 85):**
```
Wymagane: EAD103 ≥ 1.10
Inhibitory: AU26 < 1.15, AU12 < 1.10
```

**RELAXED (priorytet 70):**
```
Wymagane: brak silnych aktywacji
Inhibitory: AU26, EAD103, EAD102, AU101 wszystkie < 1.10
```

**NEUTRAL (priorytet 50):**
```
Fallback - zawsze dopasowuje jeśli żadna inna reguła nie pasuje
```

### 2.5 Stack technologiczny

| Kategoria | Technologia | Wersja |
|-----------|-------------|--------|
| **Runtime** | Python | 3.10+ |
| **ML Framework** | PyTorch | 2.0+ |
| **Detection** | Ultralytics (YOLOv8) | 8.0+ |
| **Classification** | timm (EfficientNet) | 0.9+ |
| **Keypoints** | SimpleBaseline (custom) | - |
| **Frontend Demo** | Streamlit | 1.28+ |
| **Frontend Webapp** | React + Vite | 18.0+ |
| **Backend API** | FastAPI | 0.100+ |
| **Linting** | Ruff | 0.1+ |
| **Type checking** | MyPy | - |
| **Wersjonowanie** | Git, GitHub | - |
| **Model storage** | Git LFS | - |

### 2.6 Struktura projektu

```
dog-facs/
├── apps/
│   ├── demo/                   # Streamlit demo application
│   │   └── app.py              # Main entry point
│   ├── webapp/                 # React + FastAPI application
│   │   ├── frontend/           # React frontend (Vite)
│   │   └── backend/            # FastAPI backend
│   │       └── main.py         # API endpoints
│   └── verification/           # Manual verification tool
│       └── app.py
├── packages/
│   ├── models/                 # AI Models
│   │   ├── bbox.py             # YOLOv8 dog detector
│   │   ├── breed.py            # EfficientNet breed classifier
│   │   ├── keypoints.py        # SimpleBaseline keypoints
│   │   ├── emotion.py          # DogFACS Rule Engine (NO ML)
│   │   ├── action_units.py     # Absolute AU extractor
│   │   ├── delta_action_units.py # Delta AU extractor
│   │   └── head_pose.py        # Head pose estimation
│   ├── pipeline/               # Processing pipeline
│   │   ├── inference.py        # Main inference pipeline
│   │   ├── video.py            # Video frame extraction
│   │   ├── neutral_frame.py    # Neutral frame detection
│   │   ├── peak_selector.py    # TFM-based peak selection
│   │   └── temporal_processor.py
│   └── data/                   # Data utilities
│       ├── coco.py             # COCO format handler
│       └── schemas.py          # Data classes (20 keypoints)
├── models/                     # Model weights (Git LFS)
│   ├── yolov8m.pt              # 52 MB - Detection
│   ├── breed.pt                # 72 MB - Breed classification
│   └── keypoints_dogflw.pt     # 102 MB - Keypoints
├── scripts/
│   ├── annotation/             # Batch annotation scripts
│   ├── download/               # Video download scripts
│   ├── training/               # Model training scripts
│   ├── verification/           # Verification scripts
│   └── docs/                   # Documentation generation
├── tests/                      # Unit tests
│   └── test_models/
│       ├── test_bbox.py
│       ├── test_breed.py
│       ├── test_delta_action_units.py
│       └── test_emotion_rules.py
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Documentation
└── data/                       # Local data (gitignored)
```

### 2.7 Interfejs programistyczny (API)

#### 2.7.1 Pipeline podstawowy (single frame)

```python
from packages.pipeline import InferencePipeline, PipelineConfig

# Konfiguracja
config = PipelineConfig(
    device="cuda",
    confidence_threshold=0.3,
    use_rule_based_emotion=True  # DogFACS rules
)

# Inicjalizacja i ładowanie modeli
pipeline = InferencePipeline(config)
pipeline.load()

# Przetwarzanie pojedynczego obrazu
result = pipeline.process_frame(image)

for ann in result.annotations:
    print(f"Dog {ann.dog_id}:")
    print(f"  Breed: {ann.breed.class_name}")
    print(f"  Emotion: {ann.emotion.emotion}")
    print(f"  Rule: {ann.emotion.rule_applied}")
```

#### 2.7.2 Pipeline do generowania datasetu (video)

```python
# Przetwarzanie wideo do datasetu
result = pipeline.process_video_for_dataset(
    frames_list=frames,          # Lista klatek np.ndarray
    num_peaks=10,                # Liczba peak frames do wybrania
    neutral_idx=None,            # Auto-detect neutral frame
    min_separation_frames=30     # Minimalna separacja czasowa
)

# Wyniki
print(f"Neutral frame: {result['neutral_frame_idx']}")
for peak in result['peak_frames']:
    print(f"  Peak {peak['frame_idx']}: {peak['emotion'].emotion}")
    print(f"    TFM score: {peak['tfm_score']:.3f}")
```

#### 2.7.3 REST API (FastAPI)

| Endpoint | Metoda | Opis |
|----------|--------|------|
| `/api/health` | GET | Health check |
| `/api/process_video` | POST | Przetwarza wideo, zwraca peak frames |
| `/api/export_coco` | POST | Eksportuje dataset do formatu COCO |
| `/static/frames/*` | GET | Dostęp do zapisanych klatek |

### 2.8 Format wyjściowy COCO (rozszerzony)

```json
{
  "info": {
    "description": "Dog FACS Dataset",
    "version": "1.0",
    "year": 2025,
    "contributor": "Gdańsk University of Technology"
  },
  "images": [
    {
      "id": 1,
      "file_name": "video001_frame_00150.jpg",
      "width": 1920,
      "height": 1080,
      "source_video": "video001.mp4",
      "frame_number": 150,
      "is_neutral_frame": false,
      "tfm_score": 0.342
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 300, 400],
      "area": 120000,
      "iscrowd": 0,
      "keypoints": [120.5, 180.3, 2, ...],
      "num_keypoints": 20,
      "breed_id": 15,
      "breed": "golden_retriever",
      "emotion_id": 0,
      "emotion": "happy",
      "emotion_rule_applied": "happy_priority_100",
      "action_units": {
        "AU101": 1.15,
        "AU12": 1.32,
        "EAD102": 1.18
      },
      "confidence": {
        "bbox": 0.95,
        "breed": 0.87,
        "keypoints": 0.82,
        "emotion": 0.78
      }
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "dog",
      "supercategory": "animal",
      "keypoints": ["left_eye", "right_eye", ...],
      "skeleton": [[0, 1], [0, 2], ...]
    }
  ]
}
```

### 2.9 Wymagania systemowe

#### 2.9.1 Minimalne wymagania

| Komponent | Wymaganie |
|-----------|-----------|
| System operacyjny | Windows 10/11, Linux, macOS |
| Python | 3.10+ |
| RAM | 8 GB |
| Dysk | 15 GB wolnego miejsca (wagi modeli ~230 MB) |
| FFmpeg | Wymagany dla przetwarzania wideo |

#### 2.9.2 Zalecane wymagania

| Komponent | Wymaganie |
|-----------|-----------|
| RAM | 16 GB |
| GPU | NVIDIA z CUDA 11.8+ |
| VRAM | 6 GB+ |

### 2.10 Instalacja

```bash
# Instalacja Git LFS (wymagane dla wag modeli)
git lfs install

# Klonowanie repozytorium
git clone https://github.com/eternaki/group-project.git
cd group-project

# Checkout na gałąź feature
git checkout feature/dogfacs-rule-based

# Pobranie wag modeli (Git LFS)
git lfs pull

# Tworzenie środowiska wirtualnego
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# lub: .venv\Scripts\activate  # Windows

# Instalacja zależności
pip install -e .
pip install -e ".[dev]"

# Opcjonalnie: wsparcie GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2.11 Uruchomienie aplikacji

#### Streamlit Demo:
```bash
streamlit run apps/demo/app.py
# Dostępne pod: http://localhost:8501
```

#### React + FastAPI Webapp:
```bash
# Terminal 1: Backend
cd apps/webapp/backend
uvicorn main:app --reload

# Terminal 2: Frontend
cd apps/webapp/frontend
npm install && npm run dev
# Dostępne pod: http://localhost:5173
```

### 2.12 Testy

```bash
# Uruchomienie wszystkich testów
pytest

# Testy modeli
pytest tests/test_models/

# Testy z coverage
pytest --cov=packages
```

---

## 3. Wyniki i metryki

### 3.1 Rozmiary modeli

| Model | Plik | Rozmiar |
|-------|------|---------|
| BBox (YOLOv8m) | yolov8m.pt | 52.1 MB |
| Breed (EffNetB4) | breed.pt | 71.8 MB |
| Keypoints | keypoints_dogflw.pt | 102.1 MB |
| **Razem** | | **226 MB** |

### 3.2 Metryki modeli

| Model | Metryka | Wartość |
|-------|---------|---------|
| BBox (YOLOv8m) | mAP@0.5 | Używa pretrained COCO |
| Breed (EffNetB4) | Top-5 Accuracy | ~90% (Stanford Dogs) |
| Keypoints | PCK@0.2 | Trenowany na DogFLW |
| Emotion | Rule-based | 100% deterministic |

---

## 4. Repozytorium

| Pole | Wartość |
|------|---------|
| **URL** | https://github.com/eternaki/group-project |
| **Gałąź główna** | `main` |
| **Gałąź feature** | `feature/dogfacs-rule-based` |
| **Struktura gałęzi** | main → develop → sprint-X |
| **CI/CD** | GitHub Actions |
| **Model storage** | Git LFS |

---

## 5. Licencja

Projekt udostępniony na licencji MIT.

---

## Historia dokumentu

| Wersja | Data | Autor | Opis zmian |
|--------|------|-------|------------|
| 1.0 | 2025-01-28 | Zespół projektowy | Wersja początkowa |
| 2.0 | 2025-01-28 | Zespół projektowy | Aktualizacja na podstawie gałęzi feature/dogfacs-rule-based |

---

*Dokumentacja Techniczna Projektu - Dog FACS Dataset*
*Politechnika Gdańska, WETI, Katedra Systemów Decyzyjnych i Robotyki*
*Rok akademicki 2025/2026*
