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

## Installation

```bash
# Clone repository
git clone https://github.com/eternaki/group-project.git
cd group-project

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
