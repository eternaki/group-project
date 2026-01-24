# Model Weights

Wagi modeli AI dla Dog FACS Demo.

## Modele

| Plik | Rozmiar | Opis |
|------|---------|------|
| `yolov8m.pt` | 52 MB | Model detekcji psów (YOLOv8m) |
| `breed.pt` | 71 MB | Model klasyfikacji ras (EfficientNet-B4, 120 ras) |
| `keypoints_best.pt` | 136 MB | Model punktów kluczowych (SimpleBaseline/ResNet50, 46 keypoints) |
| `emotion_keypoints.pt` | ~1 MB | Model klasyfikacji emocji (MLP, 6 klas) - **wymaga treningu** |

## Architektura Pipeline

```
Obraz → BBox (YOLOv8) → Crop
                          ↓
                    Breed (EfficientNet-B4) → rasa
                          ↓
                    Keypoints (SimpleBaseline) → 46 punktów
                          ↓
                    Emotion (MLP) → 6 emocji
```

## Emocje (6 klas)

| ID | Emocja | Opis |
|----|--------|------|
| 0 | happy | Szczęśliwy, radosny |
| 1 | sad | Smutny, przygnębiony |
| 2 | angry | Zły, agresywny |
| 3 | fearful | Przestraszony, lękliwy |
| 4 | relaxed | Zrelaksowany, spokojny |
| 5 | neutral | Neutralny, bez emocji |

## Keypoints (46 punktów DogFLW)

| Grupa | Zakres | Punkty |
|-------|--------|--------|
| Oczy | 0-1 | Kąciki oczu |
| Kontur | 2-13 | Obrys pyska |
| Nos | 14-19 | Nos i nozdrza |
| Pysk | 20-31 | Usta i wargi |
| Pozostałe | 32-45 | Brwi, uszy, czoło |

## Trening modelu emocji

Model emocji wymaga treningu na danych z keypointami:

```bash
# Trening na własnych danych
python scripts/training/train_emotion_keypoints.py --data_path data/emotions_keypoints.csv

# Test pipeline (syntetyczne dane)
python scripts/training/train_emotion_keypoints.py --synthetic_samples 5000 --epochs 10
```

Format danych CSV:
- 138 kolumn z keypoints: `kp_0_x, kp_0_y, kp_0_v, kp_1_x, ..., kp_45_v`
- 1 kolumna z etykietą: `emotion` (0-5)

## Git LFS

Pliki `.pt` są przechowywane za pomocą Git Large File Storage (LFS).

### Instalacja Git LFS

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Windows - pobierz z https://git-lfs.github.com
```

### Po klonowaniu repozytorium

```bash
git lfs install
git lfs pull
```

### Weryfikacja

```bash
ls -la models/*.pt
```

## Stara architektura (deprecated)

Plik `emotion_old_cnn.pt` zawiera starą wersję modelu emocji opartą na CNN (EfficientNet-B0).
Ta wersja **nie używała keypoints** i jest zachowana tylko dla kompatybilności wstecznej.

Nowa architektura (MLP na keypoints) jest zgodna z podejściem DogFACS, gdzie emocje
są pochodną ruchów mięśni twarzy, a nie bezpośrednio pikseli obrazu.
