# Struktura Datasetu Dog FACS

## Przegląd

Dokument opisuje strukturę finalnego datasetu Dog FACS, format anotacji COCO oraz procedury eksportu.

---

## Struktura Katalogów

### Katalog roboczy (`data/`)

```
data/
├── raw/                    # Pobrane wideo (per emocja)
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── relaxed/
│   ├── fearful/
│   └── neutral/
├── frames/                 # Wyekstrahowane klatki
│   └── {emotion}/{video_id}/
│       └── {video_id}_{frame:06d}.jpg
├── annotations/            # Pliki anotacji
│   ├── annotations.json    # Anotacje automatyczne
│   ├── merged.json         # Scalone z korektami
│   └── progress.json       # Postęp przetwarzania
├── collection/             # Metadane kolekcji
│   ├── metadata.json
│   └── search_queries.json
├── verification/           # Dane weryfikacji
│   ├── corrections.json
│   ├── sample_ids.json
│   └── agreement_metrics.json
└── quality/                # Metryki jakości
    ├── quality_metrics.json
    └── benchmark_results.json
```

### Katalog finalny (`data/final/dog-facs-dataset/`)

```
dog-facs-dataset/
├── annotations/
│   ├── train.json          # ~80% danych
│   ├── val.json            # ~10% danych
│   └── test.json           # ~10% danych
├── images/                 # (opcjonalnie)
│   ├── train/
│   ├── val/
│   └── test/
├── README.md               # Dokumentacja datasetu
└── statistics.json         # Statystyki
```

---

## Format COCO

### Struktura główna

```json
{
  "info": {
    "description": "Dog FACS Dataset",
    "version": "1.0",
    "year": 2026,
    "date_created": "2026-01-17"
  },
  "licenses": [],
  "categories": [...],
  "images": [...],
  "annotations": [...]
}
```

### Kategorie

```json
{
  "categories": [
    {"id": 1, "name": "dog", "supercategory": "animal"}
  ]
}
```

### Obrazy

```json
{
  "id": 1,
  "file_name": "happy/video001/video001_000001.jpg",
  "width": 1280,
  "height": 720,
  "video_id": "video001",
  "frame_number": 1,
  "emotion_label": "happy"
}
```

### Anotacje

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [100, 150, 400, 300],
  "area": 120000,
  "iscrowd": 0,
  "score": 0.95,

  "breed": {
    "id": 5,
    "name": "Labrador Retriever",
    "confidence": 0.92
  },

  "emotion": {
    "id": 1,
    "name": "happy",
    "confidence": 0.87
  },

  "keypoints": [
    120.5, 180.3, 2,
    125.1, 175.8, 2,
    ...
  ],
  "num_keypoints": 46,

  "verified": true,
  "verified_by": "annotator1"
}
```

---

## Keypoints (46 punktów)

Punkty kluczowe twarzy psa według schematu DogFLW:

| Grupa | Indeksy | Opis |
|-------|---------|------|
| Nos | 0-3 | Czubek i boki nosa |
| Oczy | 4-11 | Kontury oczu (lewe/prawe) |
| Uszy | 12-19 | Kontury uszu (lewe/prawe) |
| Pysk | 20-35 | Kontur pyska |
| Brwi | 36-41 | Obszar brwi |
| Czoło | 42-45 | Kontur czoła |

Wartość visibility:
- `0` - nie widoczny
- `1` - widoczny ale zasłonięty
- `2` - widoczny

---

## Emocje

| ID | Nazwa | Opis |
|----|-------|------|
| 1 | happy | Szczęśliwy - machanie ogonem, "uśmiech" |
| 2 | sad | Smutny - opuszczone uszy, przygnębiony |
| 3 | angry | Zły - warczenie, pokazywanie zębów |
| 4 | relaxed | Zrelaksowany - spokojny, odpoczywający |
| 5 | fearful | Przestraszony - ogon między nogami |
| 6 | neutral | Neutralny - brak wyraźnych cech |

---

## Pipeline Eksportu

### 1. Scalanie anotacji

```bash
python scripts/annotation/merge_annotations.py \
    --auto data/annotations/annotations.json \
    --corrections data/verification/corrections.json \
    --output data/annotations/merged.json
```

### 2. Walidacja COCO

```bash
python scripts/annotation/validate_coco.py \
    --input data/annotations/merged.json \
    --strict
```

### 3. Eksport finalny

```bash
python scripts/annotation/export_dataset.py \
    --input data/annotations/merged.json \
    --output-dir data/final/dog-facs-dataset \
    --split 0.8 0.1 0.1 \
    --package
```

---

## Walidacja

### Wymagania

- Wszystkie obrazy mają unikalne ID
- Wszystkie anotacje mają prawidłowe `image_id`
- Bbox w formacie `[x, y, width, height]`
- Keypoints w formacie `[x1, y1, v1, x2, y2, v2, ...]`

### Sprawdzenie

```bash
# Podstawowa walidacja
python scripts/annotation/validate_coco.py --input annotations.json

# Walidacja z pycocotools
python scripts/annotation/validate_coco.py --input annotations.json --strict
```

---

## Cele Datasetu

| Metryka | Cel | Status |
|---------|-----|--------|
| Łączne klatki | 25,000 | - |
| Per emocja (min) | 400 | - |
| Różnorodność ras | 20+ | - |
| Weryfikacja manualna | 25% | - |

---

*Dokument techniczny - Dog FACS Dataset Project*
