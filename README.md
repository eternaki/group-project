# Dog FACS Dataset

AI-powered dog emotion annotation pipeline and dataset creation tool.

## Project Overview

This project creates a high-quality, publicly available dataset for dog emotion analysis containing:
- Minimum 25,000 annotated frames from YouTube videos
- COCO format annotations with: bounding boxes, breed classification, facial keypoints, emotion labels
- AI pipeline for automatic annotation using DogFACS methodology

## Team

| Member | Role |
|--------|------|
| Danylo Lohachov | Project Coordinator / Documentation / QA / Frontend |
| Anton Shkrebela | AI/ML Specialist (Keypoints & DogFACS models) |
| Danylo Zherzdiev | Backend (BBox & Breed models, Pipeline, COCO) |
| Mariia Volkova | Data Engineer (Data collection & Manual verification) |

**Supervisor:** dr hab. inż. Michał Czubenko
**Institution:** Gdańsk University of Technology, Faculty of Electronics, Telecommunications and Informatics

## Tech Stack

| Category | Technology | Version |
|----------|------------|---------|
| Runtime | Python | 3.10+ |
| ML Framework | PyTorch | 2.0+ |
| Detection | Ultralytics (YOLOv8) | 8.0+ |
| Classification | timm (EfficientNet) | 0.9+ |
| Keypoints | HRNet (custom) | - |
| Demo | Streamlit | 1.28+ |
| Linting | Ruff | 0.1+ |

Full documentation: [docs/plans/2025-01-16-tech-stack.md](docs/plans/2025-01-16-tech-stack.md)

## Installation

### System Requirements

- **Python 3.10+**
- **FFmpeg** (required for video export with audio)
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt install ffmpeg

  # Windows - download from https://ffmpeg.org/download.html
  ```

### Quick Start

```bash
# Install Git LFS (required for model weights)
# macOS:
brew install git-lfs

# Ubuntu/Debian:
sudo apt install git-lfs

# Windows: download from https://git-lfs.github.com

# Initialize Git LFS
git lfs install

# Clone repository (models will be downloaded automatically)
git clone https://github.com/eternaki/group-project.git
cd group-project

# If you already cloned without LFS, pull the models:
git lfs pull

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"

# Install all optional dependencies
pip install -e ".[dev,download,notebooks]"
```

### GPU Support (Optional)

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation

```bash
# Check Python
python --version

# Check PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run tests
pytest

# Check linting
ruff check .
```

## Project Structure

```
dog-facs/
├── apps/demo/           # Streamlit demo application
├── packages/
│   ├── models/          # AI models (bbox, breed, keypoints, emotion)
│   ├── pipeline/        # Unified inference pipeline
│   └── data/            # COCO format utilities
├── scripts/
│   ├── download/        # YouTube video downloading
│   ├── training/        # Model training scripts
│   └── annotation/      # Batch annotation scripts
├── notebooks/           # Jupyter notebooks for analysis
├── docs/
│   ├── plans/           # Design documents
│   └── reports/         # Final reports
├── data/                # Local data storage (gitignored)
└── tests/
```

## Usage

### Run Demo Application
```bash
streamlit run apps/demo/app.py
```

### Run Tests
```bash
pytest
```

### Lint Code
```bash
ruff check .
```

## License

MIT License - see [LICENSE](LICENSE) for details.
