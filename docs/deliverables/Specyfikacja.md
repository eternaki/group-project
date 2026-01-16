# Specyfikacja Oprogramowania

> Skopiuj zawartość do szablonu: `template/PG_WETI_DTP_wer. 1.00.doc`

---

## 1. Charakterystyka funkcjonalna

### 1.1 Opis systemu

System **Dog FACS Dataset** to pipeline do automatycznej anotacji emocji psów wykorzystujący modele głębokiego uczenia. System przetwarza obrazy lub klatki wideo i generuje anotacje w formacie COCO.

### 1.2 Główne funkcje

#### F1: Detekcja psów
- Wejście: Obraz RGB
- Wyjście: Lista bounding boxes z confidence score
- Model: YOLOv8 (fine-tuned)

#### F2: Klasyfikacja ras
- Wejście: Wycięty obraz psa (crop)
- Wyjście: Top-5 ras z prawdopodobieństwami
- Model: ViT lub EfficientNet (fine-tuned)

#### F3: Detekcja keypoints
- Wejście: Wycięty obraz psa
- Wyjście: 20+ punktów kluczowych twarzy (x, y, visibility)
- Model: HRNet

#### F4: Klasyfikacja emocji
- Wejście: Wektor keypoints
- Wyjście: Etykiety emocji DogFACS
- Model: Klasyfikator (MLP/SVM)

#### F5: Pipeline inference
- Orkiestracja wszystkich modeli
- Przetwarzanie batch
- Eksport COCO JSON

#### F6: Aplikacja demo
- Upload obrazów/wideo
- Wizualizacja wyników
- Eksport anotacji

---

## 2. Opis interfejsu

### 2.1 Interfejs programistyczny (API)

```python
from dogfacs.pipeline import Pipeline

# Inicjalizacja
pipeline = Pipeline(
    device="cuda",
    confidence_threshold=0.5
)

# Inference na obrazie
result = pipeline.process_image("image.jpg")

# Inference na wideo
results = pipeline.process_video("video.mp4", frame_skip=5)

# Eksport COCO
pipeline.export_coco(results, "output.json")
```

### 2.2 Struktura wyniku

```python
@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x, y, w, h
    confidence: float
    breed: str
    breed_confidence: float
    keypoints: list[tuple[float, float, float]]  # x, y, visibility
    emotions: dict[str, float]  # emotion -> confidence
```

### 2.3 Interfejs użytkownika (Streamlit)

| Ekran | Opis |
|-------|------|
| Upload | Wybór pliku obrazu lub wideo |
| Processing | Pasek postępu, podgląd |
| Results | Wizualizacja bbox, keypoints, emocji |
| Export | Pobranie COCO JSON |

---

## 3. Opis oprogramowania

### 3.1 Architektura

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit App                     │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                  Pipeline Layer                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Detector│ │ Breed   │ │Keypoints│ │ Emotion │   │
│  │ (YOLO)  │ │ (ViT)   │ │ (HRNet) │ │ (MLP)   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                   Data Layer                         │
│  ┌─────────────┐ ┌─────────────┐                    │
│  │ COCO Reader │ │ COCO Writer │                    │
│  └─────────────┘ └─────────────┘                    │
└─────────────────────────────────────────────────────┘
```

### 3.2 Struktura katalogów

```
group-project/
├── apps/
│   └── demo/              # Aplikacja Streamlit
├── packages/
│   ├── models/            # Modele AI
│   │   ├── detector/      # YOLOv8
│   │   ├── breed/         # ViT/EfficientNet
│   │   ├── keypoints/     # HRNet
│   │   └── emotion/       # Klasyfikator emocji
│   ├── pipeline/          # Orkiestracja
│   └── data/              # COCO utilities
├── scripts/               # Skrypty pomocnicze
├── notebooks/             # Analiza danych
└── docs/                  # Dokumentacja
```

### 3.3 Algorytmy

#### Detekcja psów (YOLOv8)
1. Preprocessing obrazu (resize, normalize)
2. Forward pass przez YOLOv8
3. Non-Maximum Suppression
4. Filtrowanie po klasie "dog" i confidence

#### Klasyfikacja ras
1. Crop obrazu według bbox
2. Resize do rozmiaru wejściowego modelu
3. Forward pass przez ViT/EfficientNet
4. Softmax, wybór Top-5

#### Keypoints (HRNet)
1. Crop i resize
2. Forward pass przez HRNet
3. Heatmap decoding
4. Konwersja do współrzędnych

#### Emocje (DogFACS)
1. Normalizacja keypoints
2. Forward pass przez MLP
3. Multi-label classification

---

## 4. Wyniki

### 4.1 Metryki modeli

| Model | Metryka | Wartość docelowa | Wartość osiągnięta |
|-------|---------|------------------|-------------------|
| Detector | mAP@0.5 | > 0.85 | TODO |
| Breed | Top-5 Accuracy | > 0.90 | TODO |
| Keypoints | PCK@0.2 | > 0.85 | TODO |
| Emotion | F1-score | > 0.70 | TODO |

### 4.2 Wydajność

| Operacja | GPU (RTX 3060) | CPU |
|----------|----------------|-----|
| Detection (1 img) | TODO ms | TODO ms |
| Full pipeline (1 img) | TODO ms | TODO ms |
| Video (30 fps) | TODO fps | TODO fps |

### 4.3 Dataset końcowy

| Statystyka | Wartość |
|------------|---------|
| Liczba obrazów | TODO |
| Liczba anotacji psów | TODO |
| Liczba unikalnych ras | TODO |
| Średnia keypoints per pies | TODO |

### 4.4 Przykłady działania

TODO: Dodać screenshoty z aplikacji demo

---

*Dokument wygenerowany przez AI. Wymaga weryfikacji i uzupełnienia przez zespół.*
