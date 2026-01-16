# Dataset Evaluation Report - Dog FACS Dataset

**Data:** 2025-01-16
**Autorzy:** Anton Shkrebela (U2), Mariia Volkova (U4), z pomocą Claude AI
**Sprint:** 1 - Project Setup
**Story:** 1.3 - Dataset Analysis

---

## 1. Podsumowanie Wykonawcze

### 1.1 Przeanalizowane Datasety

| Dataset | Źródło | Cel | Obrazów | Rekomendacja |
|---------|--------|-----|---------|--------------|
| DogFLW | Kaggle | Keypoints | 4,335 | ✅ **Główny dla keypoints** |
| Stanford Dogs | Stanford/Kaggle | Breed Classification | 20,580 | ✅ **Główny dla ras** |
| Dog Emotions - 5 Classes | Kaggle | Emotion Classification | 9,325 | ✅ **Główny dla emocji** |
| Dog_Emotion_Dataset_v2 | HuggingFace | Emotion Classification | 4,000 | ⚠️ Backup/augmentacja |
| Dog Emotions Prediction | Kaggle | Emotion Classification | ~4,000 | ⚠️ Backup |
| Open Images V7 (dogs) | Google | Detection | ~100k+ | ⚠️ Augmentacja detekcji |

### 1.2 Kluczowe Wnioski

1. **Keypoints:** DogFLW jest jedynym wysokiej jakości datasetem z 46 punktami anatomicznymi
2. **Breed:** Stanford Dogs to de facto standard z 120 rasami
3. **Emotions:** Dog Emotions - 5 Classes najlepszy balans jakości i wielkości
4. **Detection:** Można użyć pretrenowanego YOLOv8 lub fine-tuning na Open Images

---

## 2. Szczegółowa Analiza Datasetów

### 2.1 DogFLW (Dog Facial Landmarks in the Wild)

**Źródło:** [Kaggle - DogFLW](https://www.kaggle.com/datasets/georgemartvel/dogflw)

| Parametr | Wartość |
|----------|---------|
| **Liczba obrazów** | 4,335 |
| **Keypoints** | 46 punktów anatomicznych |
| **Bounding boxes** | Tak |
| **Rozmiar** | 1.43 GB |
| **Licencja** | CC BY-NC 4.0 |
| **Format** | ZIP (obrazy + anotacje) |

**Cechy:**
- ✅ Punkty oparte na anatomii twarzy psa i DogFACS
- ✅ Visibility flags dla zasłoniętych punktów (0/1)
- ✅ Różnorodne warunki oświetleniowe i tła
- ✅ 120 różnych ras (pochodzi ze Stanford Dogs)
- ⚠️ Bez podziału train/test (trzeba stworzyć samemu)

**Publikacja:** Martvel et al. (2025) - "Dog facial landmarks detection and its applications for facial analysis", Scientific Reports

**Rekomendacja:** ✅ **GŁÓWNY DATASET DLA KEYPOINTS**

---

### 2.2 Stanford Dogs Dataset

**Źródło:** [Stanford Vision Lab](http://vision.stanford.edu/aditya86/ImageNetDogs/) | [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)

| Parametr | Wartość |
|----------|---------|
| **Liczba obrazów** | 20,580 |
| **Liczba ras** | 120 |
| **Train set** | 12,000 obrazów |
| **Test set** | 8,580 obrazów |
| **Bounding boxes** | Tak (dla train) |
| **Licencja** | Research only |

**Cechy:**
- ✅ De facto standard dla klasyfikacji ras
- ✅ Gotowy podział train/test
- ✅ Wysokiej jakości obrazy z ImageNet
- ✅ Szeroka różnorodność ras
- ⚠️ Licencja tylko do badań

**Publikacja:** Khosla et al. (2011) - "Novel dataset for Fine-Grained Image Categorization", CVPR

**Rekomendacja:** ✅ **GŁÓWNY DATASET DLA KLASYFIKACJI RAS**

---

### 2.3 Dog Emotions - 5 Classes

**Źródło:** [Kaggle - Dog Emotions 5 Classes](https://www.kaggle.com/datasets/dougandrade/dog-emotions-5-classes)

| Parametr | Wartość |
|----------|---------|
| **Liczba obrazów** | 9,325 |
| **Klasy emocji** | 5 |
| **Rozmiar** | ~231 MB |
| **Licencja** | CC0 Public Domain |
| **Format** | ZIP |

**Klasy emocji:**
| Klasa | Opis |
|-------|------|
| `alert` | Czujność, uwaga, sztywne ciało |
| `angry` | Warczenie, odsłonięte zęby |
| `frown` | Przygnębienie, ból |
| `happy` | Język na wierzchu, "uśmiech" |
| `relax` | Leżenie, odpoczynek |

**Cechy:**
- ✅ Oczyszczony z nie-psów i złych obrazów
- ✅ Public Domain (bez ograniczeń)
- ✅ 5 klas (więcej niż większość datasetów)
- ✅ Duża liczba obrazów
- ⚠️ Klasy nie mapują się dokładnie na nasze 6 kategorii

**Mapowanie na nasze kategorie:**
| Dataset | Nasza kategoria |
|---------|-----------------|
| happy | happy |
| angry | angry |
| frown | sad |
| relax | relaxed |
| alert | neutral (najbliżej) |
| - | fearful (BRAK!) |

**Rekomendacja:** ✅ **GŁÓWNY DATASET DLA EMOCJI** (z augmentacją dla fearful)

---

### 2.4 Dog_Emotion_Dataset_v2 (HuggingFace)

**Źródło:** [HuggingFace - Dewa/Dog_Emotion_Dataset_v2](https://huggingface.co/datasets/Dewa/Dog_Emotion_Dataset_v2)

| Parametr | Wartość |
|----------|---------|
| **Liczba obrazów** | 4,000 |
| **Train set** | 3,200 |
| **Test set** | 800 |
| **Klasy emocji** | 4 |
| **Rozmiar** | 162 MB |
| **Licencja** | CreativeML OpenRAIL-M |
| **Format** | Parquet |

**Klasy emocji:**
| ID | Emocja |
|----|--------|
| 0 | sad |
| 1 | angry |
| 2 | relaxed |
| 3 | happy |

**Cechy:**
- ✅ Gotowy do użycia z HuggingFace datasets
- ✅ Podział train/test
- ✅ Parquet format (szybki dostęp)
- ⚠️ Mniejszy niż Dog Emotions - 5 Classes
- ⚠️ Tylko 4 klasy (brak neutral i fearful)

**Rekomendacja:** ⚠️ **BACKUP/AUGMENTACJA** - użyć do walidacji modelu

---

### 2.5 Dog Emotions Prediction (Kaggle)

**Źródło:** [Kaggle - Dog Emotions Prediction](https://www.kaggle.com/datasets/devzohaib/dog-emotions-prediction)

| Parametr | Wartość |
|----------|---------|
| **Liczba obrazów** | ~4,000 |
| **Klasy emocji** | 4 (Happy, Sad, Relaxed, Angry) |
| **Licencja** | TBD |

**Cechy:**
- ⚠️ Oryginalny dataset z którego powstały inne
- ⚠️ Nie oczyszczony jak Dog Emotions - 5 Classes
- ⚠️ Podobna zawartość do HuggingFace wersji

**Rekomendacja:** ⚠️ **NIE UŻYWAĆ** - lepsze alternatywy dostępne

---

### 2.6 Open Images V7 (Dogs Subset)

**Źródło:** [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)

| Parametr | Wartość |
|----------|---------|
| **Obrazów (pełny)** | ~9 milionów |
| **Dogs subset** | ~100,000+ (szacunkowo) |
| **Bounding boxes** | Tak |
| **Segmentation masks** | Tak |
| **Licencja** | CC BY 2.0 |

**Cechy:**
- ✅ Ogromna liczba obrazów
- ✅ Wysokiej jakości anotacje bbox
- ✅ Maski segmentacji
- ✅ Można filtrować tylko psy
- ⚠️ Duży rozmiar do pobrania
- ⚠️ Wymaga preprocessingu

**Rekomendacja:** ⚠️ **AUGMENTACJA** - dla fine-tuning detekcji jeśli potrzebne

---

## 3. Porównanie Datasetów

### 3.1 Porównanie dla Keypoints

| Dataset | Obrazy | Keypoints | Visibility | Rekomendacja |
|---------|--------|-----------|------------|--------------|
| DogFLW | 4,335 | 46 | Tak | ✅ Główny |
| (inne brak) | - | - | - | - |

**Wniosek:** DogFLW to jedyny odpowiedni dataset dla keypoints.

### 3.2 Porównanie dla Breed Classification

| Dataset | Obrazy | Rasy | Bbox | Rekomendacja |
|---------|--------|------|------|--------------|
| Stanford Dogs | 20,580 | 120 | Tak | ✅ Główny |
| Open Images | ~100k | ~1 | Tak | Augmentacja |

**Wniosek:** Stanford Dogs jest optymalny dla klasyfikacji ras.

### 3.3 Porównanie dla Emotion Classification

| Dataset | Obrazy | Klasy | Jakość | Licencja | Rekomendacja |
|---------|--------|-------|--------|----------|--------------|
| Dog Emotions 5C | 9,325 | 5 | ⭐⭐⭐⭐ | CC0 | ✅ Główny |
| HuggingFace v2 | 4,000 | 4 | ⭐⭐⭐ | OpenRAIL | Backup |
| Kaggle Original | ~4,000 | 4 | ⭐⭐ | TBD | Nie używać |

**Wniosek:** Dog Emotions - 5 Classes najlepszy kompromis między jakością a wielkością.

---

## 4. Zidentyfikowane Luki w Danych

### 4.1 Brakujące kategorie emocji

| Kategoria | Dostępność | Rozwiązanie |
|-----------|------------|-------------|
| happy | ✅ Dostępny | Bezpośrednie użycie |
| sad | ✅ Dostępny (frown) | Mapowanie |
| angry | ✅ Dostępny | Bezpośrednie użycie |
| fearful | ❌ **BRAK** | Web scraping / manual |
| relaxed | ✅ Dostępny | Bezpośrednie użycie |
| neutral | ⚠️ Częściowo (alert) | Mapowanie z alert |

### 4.2 Plan uzupełnienia luk

1. **Fearful:**
   - Zebrać z YouTube (szukaj: "scared dog", "fearful dog reaction")
   - Manual labeling ~500-1000 obrazów
   - Transfer learning z innych emocji

2. **Neutral:**
   - Użyć części "alert" z Dog Emotions 5C
   - Dodać obrazy z low confidence w innych klasach

---

## 5. Rekomendacje dla Projektu

### 5.1 Finalne wybory datasetów

| Model | Dataset(s) | Uzasadnienie |
|-------|-----------|--------------|
| **BBox Detection** | Pretrained YOLOv8 + Open Images (optional) | YOLOv8 już działa na psach |
| **Breed Classification** | Stanford Dogs | Standard branżowy, 120 ras |
| **Keypoints** | DogFLW | Jedyny z anatomicznymi punktami |
| **Emotion** | Dog Emotions 5C + augmentacja | Najlepsza jakość, uzupełnić fearful |

### 5.2 Plan pobierania danych

```bash
# 1. Stanford Dogs
kaggle datasets download jessicali9530/stanford-dogs-dataset

# 2. DogFLW
kaggle datasets download georgemartvel/dogflw

# 3. Dog Emotions - 5 Classes
kaggle datasets download dougandrade/dog-emotions-5-classes

# 4. HuggingFace (opcjonalnie)
from datasets import load_dataset
dataset = load_dataset('Dewa/Dog_Emotion_Dataset_v2')
```

### 5.3 Struktura katalogów

```
data/
├── raw/
│   ├── stanford_dogs/
│   ├── dogflw/
│   ├── dog_emotions_5c/
│   └── huggingface_emotions/
├── processed/
│   ├── bbox/          # YOLO format
│   ├── breed/         # ImageFolder format
│   ├── keypoints/     # COCO keypoints format
│   └── emotions/      # ImageFolder format
└── annotations/
    └── coco/          # Final COCO JSON
```

---

## 6. Licencje i Użycie

| Dataset | Licencja | Użycie komercyjne | Wymagana cytacja |
|---------|----------|-------------------|------------------|
| DogFLW | CC BY-NC 4.0 | ❌ | Tak |
| Stanford Dogs | Research only | ❌ | Tak |
| Dog Emotions 5C | CC0 | ✅ | Nie |
| HuggingFace v2 | OpenRAIL-M | ⚠️ Warunki | Nie |
| Open Images | CC BY 2.0 | ✅ | Tak |

**Wniosek:** Projekt akademicki - wszystkie licencje OK. Dla komercyjnego użycia wymagałoby ponownej analizy.

---

## 7. Źródła

1. [DogFLW - Kaggle](https://www.kaggle.com/datasets/georgemartvel/dogflw)
2. [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
3. [Dog Emotions - 5 Classes - Kaggle](https://www.kaggle.com/datasets/dougandrade/dog-emotions-5-classes)
4. [Dog_Emotion_Dataset_v2 - HuggingFace](https://huggingface.co/datasets/Dewa/Dog_Emotion_Dataset_v2)
5. [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)
6. [DogFLW Paper - Nature Scientific Reports](https://www.nature.com/articles/s41598-025-07040-3)

---

*Dokument wygenerowany w ramach projektu Dog FACS Dataset dla Politechniki Gdańskiej (WETI).*
