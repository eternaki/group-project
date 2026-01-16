# Tech Stack - Dog FACS Dataset

**Data:** 2025-01-16
**Autorzy:** Cały zespół, z pomocą Claude AI
**Sprint:** 1 - Project Setup
**Story:** 1.5 - Tech Stack Finalization

---

## 1. Podsumowanie Stosu Technologicznego

| Kategoria | Technologia | Wersja | Uzasadnienie |
|-----------|-------------|--------|--------------|
| **Runtime** | Python | 3.10+ | LTS, pełne wsparcie dla ML bibliotek |
| **ML Framework** | PyTorch | 2.0+ | Dynamiczne grafy, ekosystem, torch.compile |
| **GPU** | CUDA | 11.8+ | Kompatybilność z PyTorch 2.0+ |
| **Detection** | Ultralytics | 8.0+ | YOLOv8, najlepszy ekosystem |
| **Classification** | timm | 0.9+ | EfficientNet, ViT, pretrained weights |
| **Demo** | Streamlit | 1.28+ | Szybkie prototypowanie UI |
| **Linting** | Ruff | 0.1+ | Najszybszy linter Python |

---

## 2. Core Stack

### 2.1 Python Runtime

```yaml
Wersja: Python 3.10+
Uzasadnienie:
  - LTS (Long Term Support) do 2026
  - Pełne wsparcie dla wszystkich bibliotek ML
  - Nowe features: match-case, parametric typing
  - Kompatybilność z CUDA i cuDNN
```

**Weryfikacja:**
```bash
python --version
# Oczekiwane: Python 3.10.x lub wyżej
```

### 2.2 PyTorch

```yaml
Wersja: PyTorch 2.0+
Features:
  - torch.compile() - znaczące przyspieszenie
  - Stabilne API dla custom models
  - TorchScript dla deployment
  - Native CUDA Graphs support
```

**Instalacja z CUDA:**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Weryfikacja:**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### 2.3 CUDA / cuDNN

```yaml
CUDA: 11.8+ (zalecane 12.1)
cuDNN: 8.6+ (automatycznie z PyTorch)
Kompatybilność:
  - PyTorch 2.0: CUDA 11.7, 11.8, 12.1
  - PyTorch 2.1+: CUDA 11.8, 12.1, 12.4
```

**Sprawdzenie CUDA:**
```bash
nvcc --version
nvidia-smi
```

---

## 3. Biblioteki ML

### 3.1 Ultralytics (YOLOv8)

```yaml
Wersja: 8.0+
Zastosowanie: Dog detection (bounding boxes)
Features:
  - Pretrained na COCO (80 klas, w tym "dog")
  - Export do ONNX, TensorRT, CoreML
  - Tracking (dla wideo)
  - Segmentation (opcjonalnie)
```

**Użycie:**
```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model('dog.jpg')
```

### 3.2 timm (PyTorch Image Models)

```yaml
Wersja: 0.9+
Zastosowanie: Breed classification (EfficientNet-B4)
Features:
  - 700+ pretrained models
  - ImageNet-1K i ImageNet-21K weights
  - Unified API dla wszystkich architektur
```

**Użycie:**
```python
import timm

model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=120)
```

### 3.3 MMPose (opcjonalnie)

```yaml
Wersja: 1.0+ (jeśli potrzebne)
Zastosowanie: HRNet keypoint detection
Alternatywa: Własna implementacja na PyTorch
```

### 3.4 OpenCV

```yaml
Wersja: 4.8+
Zastosowanie:
  - Odczyt/zapis obrazów i wideo
  - Preprocessing (resize, crop, augmentation)
  - Ekstrakcja klatek z wideo
```

### 3.5 pycocotools

```yaml
Wersja: 2.0+
Zastosowanie:
  - Walidacja COCO JSON
  - Kalkulacja metryk (mAP, PCK)
  - Wizualizacja anotacji
```

---

## 4. Data & Download

### 4.1 yt-dlp

```yaml
Wersja: 2023.10+
Zastosowanie: Pobieranie filmów YouTube
Features:
  - Aktywnie rozwijany fork youtube-dl
  - Obsługa wielu formatów
  - Rate limiting i retry
```

**Użycie:**
```bash
yt-dlp -f 'best[height<=720]' --output 'data/raw/%(id)s.%(ext)s' URL
```

### 4.2 Pandas

```yaml
Wersja: 2.0+
Zastosowanie:
  - Analiza datasetów
  - Przetwarzanie metadanych
  - Statystyki i raporty
```

### 4.3 NumPy

```yaml
Wersja: 1.24+
Zastosowanie:
  - Operacje na obrazach
  - Keypoint manipulation
  - Data augmentation calculations
```

---

## 5. Demo Application

### 5.1 Streamlit

```yaml
Wersja: 1.28+
Zastosowanie: Interaktywna aplikacja demo
Features:
  - Szybki development UI
  - File upload (obrazy/wideo)
  - Real-time preview
  - Session state management
```

**Uruchomienie:**
```bash
streamlit run apps/demo/app.py
```

### 5.2 Pillow

```yaml
Wersja: 10.0+
Zastosowanie:
  - Obsługa obrazów w Streamlit
  - Konwersja formatów
  - Podstawowe operacje
```

---

## 6. Development Tools

### 6.1 Ruff

```yaml
Wersja: 0.1+
Zastosowanie: Linting + formatting
Features:
  - 10-100x szybszy niż flake8
  - Zastępuje: flake8, isort, pyupgrade
  - Auto-fix dla większości błędów
```

**Konfiguracja (pyproject.toml):**
```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]
```

**Użycie:**
```bash
ruff check .
ruff check . --fix
```

### 6.2 mypy

```yaml
Wersja: 1.5+
Zastosowanie: Static type checking
Features:
  - Wykrywanie błędów typów
  - Lepsza dokumentacja kodu
  - IDE integration
```

**Konfiguracja (pyproject.toml):**
```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true
```

**Użycie:**
```bash
mypy packages/
```

### 6.3 pytest

```yaml
Wersja: 7.4+
Zastosowanie: Unit testing
Plugins:
  - pytest-cov: coverage reports
```

**Użycie:**
```bash
pytest
pytest --cov=packages
```

---

## 7. Infrastruktura Treningowa

### 7.1 Opcje GPU

| Opcja | GPU | VRAM | Koszt | Dostępność |
|-------|-----|------|-------|------------|
| **Politechnika GUT** | Tesla V100 | 32GB | Bezpłatne | Kolejki |
| **Google Colab** | T4/V100 | 16GB | Free tier | Limity sesji |
| **Kaggle Kernels** | P100/T4 | 16GB | 30h/tyg | Stabilne |
| **Local RTX 3090** | RTX 3090 | 24GB | Hardware | Bez limitów |

### 7.2 Rekomendacja

```yaml
Development: Local GPU lub Google Colab
Prototyping: Kaggle Kernels (30h/tyg darmowe)
Training produkcyjny: Klaster Politechniki GUT
Backup: Google Colab Pro ($10/mies)
```

### 7.3 Wymagania minimalne (Training)

| Model | Min VRAM | Batch Size | Szacowany czas |
|-------|----------|------------|----------------|
| YOLOv8m | 8GB | 16 | ~2h (100 epochs) |
| EfficientNet-B4 | 8GB | 32 | ~1h (50 epochs) |
| HRNet-W32 | 12GB | 32 | ~4h (100 epochs) |
| MLP Emotion | 2GB | 64 | ~10min (100 epochs) |

### 7.4 Wymagania minimalne (Inference)

| Scenariusz | Hardware | Speed |
|------------|----------|-------|
| GPU inference | GTX 1060+ | ~35 FPS |
| CPU inference | i5-8400+ | ~3 FPS |
| Demo (single image) | Any | <2s |

---

## 8. Wersje Zależności (pyproject.toml)

```toml
[project]
requires-python = ">=3.10"

dependencies = [
    # Core ML
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "ultralytics>=8.0.0",
    "timm>=0.9.0",

    # Image/Video
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",

    # Data
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pycocotools>=2.0.0",

    # Demo
    "streamlit>=1.28.0",

    # Utils
    "tqdm>=4.65.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]
download = [
    "yt-dlp>=2023.10.0",
]
notebooks = [
    "jupyter>=1.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

---

## 9. Setup Instructions

### 9.1 Szybki start

```bash
# 1. Klonowanie repozytorium
git clone https://github.com/eternaki/group-project.git
cd group-project

# 2. Tworzenie virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Instalacja podstawowa
pip install -e .

# 4. Instalacja dev tools
pip install -e ".[dev]"

# 5. Instalacja wszystkiego
pip install -e ".[dev,download,notebooks]"
```

### 9.2 Weryfikacja instalacji

```bash
# Sprawdź Python
python --version

# Sprawdź PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Sprawdź YOLOv8
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"

# Sprawdź timm
python -c "import timm; print(f'timm {timm.__version__}')"

# Uruchom testy
pytest

# Sprawdź linting
ruff check .
```

### 9.3 GPU Setup (opcjonalnie)

```bash
# Sprawdź CUDA
nvcc --version
nvidia-smi

# Instalacja PyTorch z CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Lub z CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 10. Środowisko Deweloperskie

### 10.1 Zalecane IDE

| IDE | Zalety | Konfiguracja |
|-----|--------|--------------|
| **VS Code** | Lekkie, dobre rozszerzenia | Python + Pylance + Ruff |
| **PyCharm** | Pełne IDE, debugging | Professional (dla studentów darmowe) |
| **Cursor** | AI-assisted coding | Fork VS Code |

### 10.2 VS Code Extensions

```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "ms-python.mypy-type-checker",
        "github.copilot"
    ]
}
```

### 10.3 Git Hooks (opcjonalnie)

```bash
# Pre-commit hook dla ruff
pip install pre-commit
pre-commit install
```

---

## 11. Zgodność i Testowanie

### 11.1 Testowane konfiguracje

| OS | Python | PyTorch | CUDA | Status |
|----|--------|---------|------|--------|
| Ubuntu 22.04 | 3.10 | 2.1 | 11.8 | ✅ |
| macOS 14 (M1) | 3.11 | 2.1 | MPS | ✅ |
| Windows 11 | 3.10 | 2.1 | 12.1 | ✅ |
| Google Colab | 3.10 | 2.1 | T4 | ✅ |

### 11.2 Znane problemy

| Problem | Rozwiązanie |
|---------|-------------|
| pycocotools na Windows | `pip install pycocotools-windows` |
| CUDA out of memory | Zmniejsz batch size |
| M1/M2 Mac - brak CUDA | Użyj MPS backend (`device='mps'`) |

---

## 12. Źródła

1. [PyTorch Installation](https://pytorch.org/get-started/locally/)
2. [Ultralytics Documentation](https://docs.ultralytics.com/)
3. [timm Documentation](https://huggingface.co/docs/timm/)
4. [Ruff Documentation](https://docs.astral.sh/ruff/)
5. [Streamlit Documentation](https://docs.streamlit.io/)

---

*Dokument wygenerowany w ramach projektu Dog FACS Dataset dla Politechniki Gdańskiej (WETI).*
