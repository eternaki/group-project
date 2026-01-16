# Model Architecture Decisions - Dog FACS Dataset

**Data:** 2025-01-16
**Autorzy:** Anton Shkrebela (U2), Danylo Zherzdiev (U3), z pomocą Claude AI
**Sprint:** 1 - Project Setup
**Story:** 1.4 - Model Architecture Research

---

## 1. Podsumowanie Decyzji

| Model | Zadanie | Architektura | Uzasadnienie |
|-------|---------|--------------|--------------|
| **BBox** | Detekcja psów | YOLOv8m | Balans dokładności i szybkości, dojrzały ekosystem |
| **Breed** | Klasyfikacja ras | EfficientNet-B4 | Lepsza wydajność na małych datasetach |
| **Keypoints** | Punkty twarzy | HRNet-W32 | Wysoka dokładność dla keypoints |
| **Emotion** | Klasyfikacja emocji | MLP (3 warstwy) | Prosty, szybki, na features z keypoints |

---

## 2. Dog Detection (Bounding Box)

### 2.1 Kandydaci

| Model | mAP (COCO) | Inference (ms) | Params | Zalety | Wady |
|-------|------------|----------------|--------|--------|------|
| **YOLOv8n** | 37.3% | ~6ms | 3.2M | Ultra szybki | Niska dokładność |
| **YOLOv8s** | 44.9% | ~10ms | 11.2M | Szybki | Średnia dokładność |
| **YOLOv8m** | 50.2% | ~23ms | 25.9M | Balans | - |
| **YOLOv8l** | 52.9% | ~40ms | 43.7M | Wysoka dokładność | Wolniejszy |
| YOLOv9c | 52.5% | ~25ms | 25.3M | PGI, GELAN | Nowszy, mniej testowany |
| YOLOv10m | 51.3% | ~19ms | 15.4M | NMS-free | Inkonsystencje detekcji |
| YOLO11m | 51.5% | ~21ms | 20.1M | Najnowszy | Bardzo nowy |

### 2.2 Decyzja: **YOLOv8m**

**Uzasadnienie:**
1. **Dojrzały ekosystem** - Ultralytics ma świetną dokumentację i wsparcie
2. **Balans mAP/speed** - 50.2% mAP przy ~23ms inference
3. **Łatwy fine-tuning** - sprawdzony workflow dla custom datasets
4. **Stabilność** - brak inkonsystencji jak w YOLOv10
5. **Multi-task** - obsługuje detection, segmentation, pose

**Alternatywa:** YOLO11m dla nowszych projektów (po stabilizacji)

### 2.3 Specyfikacja techniczna

```yaml
Model: YOLOv8m
Input: 640x640 RGB
Output: List[BBox] z confidence scores
Format wag: bbox.pt
Framework: Ultralytics
Fine-tuning dataset: Stanford Dogs + Open Images (dogs)
Target mAP: >85% na psach
```

---

## 3. Breed Classification

### 3.1 Kandydaci

| Model | Top-1 Acc | Top-5 Acc | Params | Zalety | Wady |
|-------|-----------|-----------|--------|--------|------|
| **EfficientNet-B4** | ~83% | ~96% | 19M | Efektywny, szybki | - |
| EfficientNet-B7 | ~84% | ~97% | 66M | Wyższa dokładność | Duży |
| ViT-B/16 | ~85% | ~97% | 86M | SOTA | Wymaga dużo danych |
| ResNet-50 | ~79% | ~94% | 25M | Prosty | Niższa dokładność |
| ConvNeXt-B | ~84% | ~97% | 89M | Nowoczesny CNN | Duży |

### 3.2 Decyzja: **EfficientNet-B4**

**Uzasadnienie:**
1. **Optymalne dla małych datasetów** - Stanford Dogs ma ~20k obrazów
2. **Efficient compound scaling** - balans głębokości/szerokości/rozdzielczości
3. **Transfer learning friendly** - pretrained na ImageNet-1K/21K
4. **Szybki inference** - ważne dla pipeline
5. **ViT wymaga 14M+ obrazów** dla pełnej skuteczności

**Cytując badania:**
> "If the ViT model is trained on huge datasets that are over 14M images, it can outperform the CNNs. If not, the best option is to stick to EfficientNet."

### 3.3 Specyfikacja techniczna

```yaml
Model: EfficientNet-B4
Input: 380x380 RGB (lub 224x224 dla szybkości)
Output: Softmax 120 klas (Stanford Dogs)
Format wag: breed.pt
Framework: PyTorch + timm
Fine-tuning: Stanford Dogs Dataset
Target Top-5: >90%
```

---

## 4. Keypoint Detection

### 4.1 Kandydaci

| Model | PCK@0.1 | Inference (ms) | Params | Zalety | Wady |
|-------|---------|----------------|--------|--------|------|
| **HRNet-W32** | ~75% | ~50ms | 28.5M | Wysoka dokładność | Średnia prędkość |
| HRNet-W18 | ~70% | ~30ms | 9.3M | Szybszy | Niższa dokładność |
| HRNet-W48 | ~78% | ~80ms | 63.6M | Najwyższa | Wolny |
| RTMPose-m | ~72% | ~12ms | 13.6M | Bardzo szybki | Mniej testowany na zwierzętach |
| RTMPose-l | ~74% | ~18ms | 27.7M | Szybki, dokładny | Nowszy |

### 4.2 Decyzja: **HRNet-W32**

**Uzasadnienie:**
1. **High-resolution representations** - zachowuje szczegóły przez całą sieć
2. **Sprawdzony na DogFLW** - benchmark w publikacji naukowej
3. **MMPose support** - gotowe implementacje dla animal keypoints
4. **46 keypoints** - wystarczająca dokładność dla twarzy psa
5. **Trade-off** - dobry balans między W18 (za mało dokładny) a W48 (za wolny)

### 4.3 Specyfikacja techniczna

```yaml
Model: HRNet-W32
Input: 256x256 RGB (cropped dog face)
Output: 20 keypoints × (x, y, visibility)
Format wag: keypoints.pt
Framework: MMPose / PyTorch
Training dataset: DogFLW (4,335 obrazów, 46 punktów)
Adaptacja: Redukcja do 20 kluczowych punktów
Target PCK@0.1: >75%
```

### 4.4 Keypoints Schema (20 punktów)

```python
KEYPOINTS = [
    "left_eye_inner", "left_eye_outer",      # 0-1
    "right_eye_inner", "right_eye_outer",    # 2-3
    "nose_tip", "nose_left", "nose_right",   # 4-6
    "left_ear_base", "left_ear_tip",         # 7-8
    "right_ear_base", "right_ear_tip",       # 9-10
    "mouth_left", "mouth_right",             # 11-12
    "upper_lip", "lower_lip", "chin",        # 13-15
    "left_brow", "right_brow",               # 16-17
    "forehead", "muzzle_center"              # 18-19
]
```

---

## 5. Emotion Classification

### 5.1 Kandydaci

| Model | Architektura | Input | Zalety | Wady |
|-------|--------------|-------|--------|------|
| **MLP (3-layer)** | FC layers | Keypoints (60 features) | Prosty, szybki, interpretowalny | Wymaga dobrych keypoints |
| Small CNN | Conv2D | Cropped face (64x64) | Może uchwycić tekstury | Większy, wolniejszy |
| Hybrid | CNN + keypoints | Both | Może być najlepszy | Skomplikowany |

### 5.2 Decyzja: **MLP (3 warstwy)**

**Uzasadnienie:**
1. **Keypoints zawierają informację o emocjach** - pozycje uszu, pyska, oczu
2. **Szybki inference** - <1ms na CPU
3. **Interpretowalność** - można analizować feature importance
4. **Badania pokazują 71-89% accuracy** na DogFACS z keypoints
5. **Prostota** - łatwy do trenowania i debugowania

### 5.3 Specyfikacja techniczna

```yaml
Model: MLP
Input: 60 features (20 keypoints × 3 wartości: x, y, visibility)
Hidden layers:
  - Linear(60, 128) + ReLU + Dropout(0.3)
  - Linear(128, 64) + ReLU + Dropout(0.3)
  - Linear(64, 6) + Softmax
Output: 6 klas emocji
Format wag: emotion.pt
Framework: PyTorch
Training dataset: Dog Emotions 5C + augmentacja
Target accuracy: >70%
```

### 5.4 Emotion Classes

| ID | Klasa | Opis |
|----|-------|------|
| 0 | happy | Szczęśliwy, radosny |
| 1 | sad | Smutny, przygnębiony |
| 2 | angry | Zły, agresywny |
| 3 | fearful | Przestraszony |
| 4 | relaxed | Zrelaksowany |
| 5 | neutral | Neutralny |

---

## 6. Pipeline Architecture

### 6.1 Przepływ danych

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Input     │     │   YOLOv8m   │     │   Crop      │
│   Image     │────▶│   BBox      │────▶│   Dog Face  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
            ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
            │ EfficientNet│            │  HRNet-W32  │            │             │
            │    Breed    │            │  Keypoints  │────────────│   MLP       │
            └─────────────┘            └─────────────┘            │  Emotion    │
                    │                          │                  └─────────────┘
                    │                          │                          │
                    ▼                          ▼                          ▼
            ┌─────────────────────────────────────────────────────────────┐
            │                      COCO JSON Output                       │
            │  - bbox, breed, breed_confidence                            │
            │  - keypoints (20 points)                                    │
            │  - emotion, emotion_confidence                              │
            └─────────────────────────────────────────────────────────────┘
```

### 6.2 Inference Times (szacowane)

| Model | Input | GPU (ms) | CPU (ms) |
|-------|-------|----------|----------|
| YOLOv8m | 640x640 | ~7 | ~100 |
| EfficientNet-B4 | 380x380 | ~5 | ~50 |
| HRNet-W32 | 256x256 | ~15 | ~150 |
| MLP Emotion | 60 features | <1 | <1 |
| **Total (1 dog)** | - | **~28ms** | **~300ms** |

**Wniosek:** Cały pipeline ~35 FPS na GPU, ~3 FPS na CPU

---

## 7. Training Strategy

### 7.1 Transfer Learning

| Model | Pretrained | Fine-tune |
|-------|------------|-----------|
| YOLOv8m | COCO | Stanford Dogs + Open Images |
| EfficientNet-B4 | ImageNet-1K | Stanford Dogs |
| HRNet-W32 | COCO Keypoints | DogFLW |
| MLP | - | Dog Emotions 5C |

### 7.2 Data Augmentation

```python
# Dla wszystkich modeli
transforms = [
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.2, contrast=0.2),
    RandomResizedCrop(scale=(0.8, 1.0)),
]

# Dla keypoints - specjalne
keypoint_transforms = [
    RandomHorizontalFlip(p=0.5),  # + flip keypoints!
    RandomAffine(degrees=10, translate=(0.1, 0.1)),
]
```

### 7.3 Harmonogram treningu

| Model | Epochs | Batch Size | LR | Scheduler |
|-------|--------|------------|----|-----------|
| YOLOv8m | 100 | 16 | 0.01 | Cosine |
| EfficientNet-B4 | 50 | 32 | 0.001 | StepLR |
| HRNet-W32 | 100 | 32 | 0.001 | Cosine |
| MLP | 100 | 64 | 0.001 | ReduceOnPlateau |

---

## 8. Wymagania sprzętowe

### 8.1 Training

| Zasób | Minimum | Rekomendowane |
|-------|---------|---------------|
| GPU | GTX 1080 (8GB) | RTX 3090 (24GB) |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB SSD | 100 GB NVMe |
| CUDA | 11.7+ | 12.x |

### 8.2 Inference

| Zasób | CPU Only | GPU |
|-------|----------|-----|
| Processor | i5-8400 | Any with CUDA |
| GPU | - | GTX 1060+ |
| RAM | 8 GB | 8 GB |
| Speed | ~3 FPS | ~35 FPS |

---

## 9. Alternatywne rozwiązania

### 9.1 Jeśli potrzebna wyższa dokładność

| Current | Upgrade | Trade-off |
|---------|---------|-----------|
| YOLOv8m | YOLOv8l / YOLO11l | +5% mAP, -40% speed |
| EfficientNet-B4 | ViT-B/16 | +2% acc, wymaga więcej danych |
| HRNet-W32 | HRNet-W48 | +3% PCK, -40% speed |

### 9.2 Jeśli potrzebna wyższa szybkość

| Current | Downgrade | Trade-off |
|---------|-----------|-----------|
| YOLOv8m | YOLOv8s | -5% mAP, +2x speed |
| EfficientNet-B4 | MobileNetV3 | -5% acc, +3x speed |
| HRNet-W32 | RTMPose-m | -3% PCK, +4x speed |

---

## 10. Źródła

### YOLO
1. [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
2. [YOLO11 vs Previous Models](https://www.ultralytics.com/blog/comparing-ultralytics-yolo11-vs-previous-yolo-models)
3. [YOLOv8 vs YOLOv9 vs YOLOv10 Comparison](https://docs.ultralytics.com/compare/)

### EfficientNet / ViT
4. [Comparing ViT and EfficientNet - Exness Blog](https://medium.com/exness-blog/comparing-vit-and-efficientnet-in-terms-of-image-classification-problems-605dfdd843c7)
5. [Vision Transformer - viso.ai](https://viso.ai/deep-learning/vision-transformer-vit/)

### HRNet / Keypoints
6. [HRNet GitHub](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)
7. [MMPose Documentation](https://mmpose.readthedocs.io/)
8. [DogFLW Paper - Nature Scientific Reports](https://www.nature.com/articles/s41598-025-07040-3)

### Emotion Classification
9. [Automated recognition of emotional states - Nature (2022)](https://www.nature.com/articles/s41598-022-27079-w)

---

*Dokument wygenerowany w ramach projektu Dog FACS Dataset dla Politechniki Gdańskiej (WETI).*
