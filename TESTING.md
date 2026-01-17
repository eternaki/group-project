# Instrukcja Testowania - Dog FACS Dataset

Ten dokument opisuje jak przetestować wszystkie komponenty projektu.

---

## Spis treści

1. [Wymagania](#1-wymagania)
2. [Instalacja](#2-instalacja)
3. [Testy jednostkowe](#3-testy-jednostkowe)
4. [Testy modułów](#4-testy-modułów)
5. [Testy aplikacji](#5-testy-aplikacji)
6. [Testy skryptów](#6-testy-skryptów)
7. [Testy end-to-end](#7-testy-end-to-end)
8. [Checklist testowania](#8-checklist-testowania)
9. [Rozwiązywanie problemów](#9-rozwiązywanie-problemów)

---

## 1. Wymagania

### 1.1 Wymagania systemowe

| Komponent | Minimum | Zalecane |
|-----------|---------|----------|
| Python | 3.10+ | 3.11+ |
| RAM | 8 GB | 16 GB |
| GPU | - | NVIDIA RTX 3060+ |
| Dysk | 10 GB | 50 GB |

### 1.2 Wymagane pakiety

```bash
# Podstawowe
pip install -e .

# Pełne (z dev dependencies)
pip install -e ".[dev,download,notebooks]"
```

---

## 2. Instalacja

### 2.1 Klonowanie repozytorium

```bash
git clone https://github.com/eternaki/group-project.git
cd group-project
git checkout develop
```

### 2.2 Utworzenie środowiska wirtualnego

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# lub
.\venv\Scripts\activate  # Windows
```

### 2.3 Instalacja zależności

```bash
pip install --upgrade pip
pip install -e ".[dev,download,notebooks]"
```

### 2.4 Weryfikacja instalacji

```bash
python3 -c "import packages; print('✅ Pakiet zainstalowany poprawnie')"
```

**Oczekiwany wynik:** `✅ Pakiet zainstalowany poprawnie`

---

## 3. Testy jednostkowe

### 3.1 Uruchomienie wszystkich testów

```bash
pytest tests/ -v
```

**Oczekiwany wynik:** Wszystkie testy przechodzą (zielone)

### 3.2 Uruchomienie testów z pokryciem

```bash
pytest tests/ --cov=packages --cov-report=html
```

**Oczekiwany wynik:** Raport HTML w `htmlcov/index.html`

### 3.3 Testy pojedynczych modułów

```bash
# Testy modelu bbox
pytest tests/test_models/test_bbox.py -v

# Testy modelu breed
pytest tests/test_models/test_breed.py -v
```

---

## 4. Testy modułów

### 4.1 Test importów

Uruchom poniższy skrypt, aby sprawdzić czy wszystkie moduły importują się poprawnie:

```bash
python3 << 'EOF'
print("Testowanie importów...")

# Modele
try:
    from packages.models import BBoxModel, BreedModel, KeypointModel, EmotionModel
    print("✅ packages.models - OK")
except Exception as e:
    print(f"❌ packages.models - BŁĄD: {e}")

# Pipeline
try:
    from packages.pipeline import InferencePipeline, VideoProcessor
    print("✅ packages.pipeline - OK")
except Exception as e:
    print(f"❌ packages.pipeline - BŁĄD: {e}")

# Data
try:
    from packages.data import COCODataset, COCOWriter
    print("✅ packages.data - OK")
except Exception as e:
    print(f"❌ packages.data - BŁĄD: {e}")

print("\nWszystkie importy zakończone.")
EOF
```

**Oczekiwany wynik:**
```
Testowanie importów...
✅ packages.models - OK
✅ packages.pipeline - OK
✅ packages.data - OK
Wszystkie importy zakończone.
```

### 4.2 Test modelu BBox (stub mode)

```bash
python3 << 'EOF'
from packages.models import BBoxModel
import numpy as np

model = BBoxModel()
print(f"Model załadowany: {model}")
print(f"Device: {model.device}")

# Test z dummy obrazem
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
results = model.predict(dummy_image)
print(f"Wynik predykcji (stub): {len(results)} detekcji")
print("✅ BBoxModel działa poprawnie")
EOF
```

### 4.3 Test modelu Breed (stub mode)

```bash
python3 << 'EOF'
from packages.models import BreedModel
import numpy as np

model = BreedModel()
print(f"Model załadowany: {model}")
print(f"Liczba klas: {model.num_classes}")

# Test z dummy obrazem
dummy_crop = np.zeros((224, 224, 3), dtype=np.uint8)
breed, confidence = model.predict(dummy_crop)
print(f"Wynik: {breed} ({confidence:.2f})")
print("✅ BreedModel działa poprawnie")
EOF
```

### 4.4 Test Pipeline (stub mode)

```bash
python3 << 'EOF'
from packages.pipeline import InferencePipeline
import numpy as np

pipeline = InferencePipeline()
print(f"Pipeline załadowany")

# Test z dummy obrazem
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
results = pipeline.process_image(dummy_image)
print(f"Wynik pipeline: {len(results)} detekcji")
print("✅ InferencePipeline działa poprawnie")
EOF
```

---

## 5. Testy aplikacji

### 5.1 Aplikacja Demo (Streamlit)

```bash
streamlit run apps/demo/app.py
```

**Oczekiwany wynik:**
1. Otworzy się przeglądarka na `http://localhost:8501`
2. Widoczny interfejs z:
   - Opcją uploadu obrazu
   - Opcją uploadu wideo
   - Slidery dla parametrów
3. Po uploadzie obrazu - wizualizacja wyników

**Test manualny:**
- [ ] Aplikacja uruchamia się bez błędów
- [ ] Można uploadować obraz JPG/PNG
- [ ] Wyniki są wyświetlane (bbox, keypoints, emocje)
- [ ] Można pobrać COCO JSON

### 5.2 Aplikacja Weryfikacji

```bash
streamlit run apps/verification/app.py
```

**Oczekiwany wynik:**
1. Interfejs do weryfikacji anotacji
2. Przyciski Accept/Correct/Reject

---

## 6. Testy skryptów

### 6.1 Skrypty do pobierania danych

```bash
# Sprawdzenie pomocy (bez pobierania)
python3 scripts/download/download_videos.py --help
python3 scripts/download/preprocess_videos.py --help
python3 scripts/download/collection_tracker.py --help
```

**Oczekiwany wynik:** Wyświetla się pomoc dla każdego skryptu

### 6.2 Skrypty anotacji

```bash
python3 scripts/annotation/batch_annotate.py --help
python3 scripts/annotation/quality_monitor.py --help
python3 scripts/annotation/merge_annotations.py --help
python3 scripts/annotation/validate_coco.py --help
python3 scripts/annotation/export_dataset.py --help
```

### 6.3 Skrypty weryfikacji

```bash
python3 scripts/verification/sample_selector.py --help
python3 scripts/verification/agreement_calculator.py --help
```

### 6.4 Skrypty treningu

```bash
python3 scripts/training/prepare_bbox_data.py --help
python3 scripts/training/train_bbox.py --help
python3 scripts/training/evaluate_bbox.py --help
python3 scripts/training/prepare_breed_data.py --help
python3 scripts/training/train_breed.py --help
python3 scripts/training/evaluate_breed.py --help
```

---

## 7. Testy end-to-end

### 7.1 Test pełnego pipeline'u z obrazem

```bash
# Pobierz testowy obraz
curl -o test_dog.jpg "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=640"

# Uruchom test pipeline'u
python3 scripts/demo/test_pipeline.py --image test_dog.jpg --output test_output.json

# Sprawdź wynik
cat test_output.json | python3 -m json.tool | head -50
```

**Oczekiwany wynik:** Plik JSON z strukturą COCO

### 7.2 Test walidacji COCO

```bash
# Stwórz przykładowy plik COCO
python3 << 'EOF'
import json

coco = {
    "images": [
        {"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 200, 150],
            "area": 30000,
            "iscrowd": 0
        }
    ],
    "categories": [
        {"id": 1, "name": "dog", "supercategory": "animal"}
    ]
}

with open("test_coco.json", "w") as f:
    json.dump(coco, f, indent=2)

print("Utworzono test_coco.json")
EOF

# Waliduj
python3 scripts/annotation/validate_coco.py --input test_coco.json --report
```

**Oczekiwany wynik:** Raport walidacji bez błędów

### 7.3 Test COCO Reader/Writer

```bash
python3 << 'EOF'
from packages.data import COCODataset, COCOWriter
import tempfile
import os

# Test writer
writer = COCOWriter()
writer.add_image(1, "test.jpg", 640, 480)
writer.add_annotation(
    annotation_id=1,
    image_id=1,
    category_id=1,
    bbox=[100, 100, 200, 150]
)
writer.add_category(1, "dog", "animal")

# Zapisz
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    temp_path = f.name
    writer.save(temp_path)
    print(f"Zapisano do: {temp_path}")

# Odczytaj
dataset = COCODataset(temp_path)
print(f"Obrazów: {len(dataset.images)}")
print(f"Anotacji: {len(dataset.annotations)}")
print(f"Kategorii: {len(dataset.categories)}")

# Cleanup
os.unlink(temp_path)
print("✅ COCO Reader/Writer działa poprawnie")
EOF
```

---

## 8. Checklist testowania

### 8.1 Podstawowe testy

- [ ] `pip install -e .` - instalacja bez błędów
- [ ] `pytest tests/ -v` - wszystkie testy przechodzą
- [ ] Import wszystkich modułów działa
- [ ] Aplikacja demo uruchamia się

### 8.2 Testy modeli (stub mode)

- [ ] BBoxModel - predict() zwraca listę
- [ ] BreedModel - predict() zwraca (breed, confidence)
- [ ] KeypointModel - predict() zwraca keypoints
- [ ] EmotionModel - predict() zwraca emotion

### 8.3 Testy aplikacji

- [ ] `streamlit run apps/demo/app.py` - działa
- [ ] `streamlit run apps/verification/app.py` - działa
- [ ] Upload obrazu w demo - wizualizacja wyników
- [ ] Eksport COCO JSON - poprawna struktura

### 8.4 Testy skryptów CLI

- [ ] Wszystkie skrypty w `scripts/` mają działający `--help`
- [ ] `validate_coco.py` waliduje poprawne pliki

### 8.5 Testy dokumentacji

- [ ] README.md jest aktualny
- [ ] CLAUDE.md zawiera instrukcje dla AI
- [ ] `docs/` zawiera dokumentację techniczną

---

## 9. Rozwiązywanie problemów

### 9.1 ModuleNotFoundError

**Problem:** `ModuleNotFoundError: No module named 'packages'`

**Rozwiązanie:**
```bash
pip install -e .
```

### 9.2 CUDA not available

**Problem:** PyTorch nie wykrywa GPU

**Rozwiązanie:**
```bash
# Sprawdź wersję CUDA
nvidia-smi

# Zainstaluj PyTorch z CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 9.3 Streamlit nie uruchamia się

**Problem:** Port 8501 zajęty

**Rozwiązanie:**
```bash
streamlit run apps/demo/app.py --server.port 8502
```

### 9.4 yt-dlp nie działa

**Problem:** Błędy przy pobieraniu wideo

**Rozwiązanie:**
```bash
pip install --upgrade yt-dlp
```

### 9.5 Testy nie przechodzą

**Problem:** pytest zwraca błędy

**Rozwiązanie:**
```bash
# Sprawdź wersje pakietów
pip list | grep -E "torch|pytest|numpy"

# Reinstaluj dev dependencies
pip install -e ".[dev]" --force-reinstall
```

---

## 10. Struktura testów

```
tests/
├── test_models/
│   ├── __init__.py
│   ├── test_bbox.py      # Testy detekcji
│   └── test_breed.py     # Testy klasyfikacji ras
├── test_pipeline/
│   └── test_inference.py # Testy pipeline'u
└── test_data/
    └── test_coco.py      # Testy formatu COCO
```

---

## 11. Raportowanie błędów

Jeśli znajdziesz błąd:

1. Sprawdź czy błąd nie jest już zgłoszony w Issues
2. Utwórz nowe Issue z:
   - Opisem błędu
   - Krokami do reprodukcji
   - Oczekiwanym vs faktycznym zachowaniem
   - Wersją Python i systemu operacyjnego
   - Pełnym traceback'iem błędu

---

*Dokument wygenerowany automatycznie. Ostatnia aktualizacja: Styczeń 2026*
