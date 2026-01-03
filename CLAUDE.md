# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dog FACS Dataset - AI-powered dog emotion annotation pipeline. Creates a COCO-format dataset with bounding boxes, breed classification, facial keypoints, and emotion labels from YouTube dog videos.

## Build & Development Commands

```bash
# Install dependencies
pip install -e .
pip install -e ".[dev,download,notebooks]"

# Run demo
streamlit run apps/demo/app.py

# Run tests
pytest

# Lint
ruff check .
ruff check . --fix

# Type check
mypy packages/
```

## Architecture

### Monorepo Structure
- `apps/demo/` - Streamlit demo application
- `packages/models/` - AI models (YOLOv8 bbox, ViT breed, HRNet keypoints, emotion classifier)
- `packages/pipeline/` - Unified inference orchestrating all models
- `packages/data/` - COCO format read/write utilities
- `scripts/` - Training, downloading, batch annotation scripts
- `notebooks/` - Dataset statistics and analysis
- `docs/plans/` - Design documents for each implementation phase

### AI Pipeline Flow
```
Image/Frame → BBox Detection → Crop → Breed Classification
                                   → Keypoints Detection → Emotion Classification

Output: COCO JSON with all annotations
```

### Key Models
| Model | Architecture | Purpose |
|-------|--------------|---------|
| BBox | YOLOv8 | Dog detection |
| Breed | ViT/EfficientNet | Breed classification |
| Keypoints | HRNet | Facial keypoints (20+ points) |
| Emotion | Classifier on keypoints | DogFACS emotion labels |

## Documentation Convention

After implementing any feature, create/update a design document in `docs/plans/` following the pattern:
`YYYY-MM-DD-<feature-name>.md`

## Data Directories

`data/` is gitignored. Structure:
- `data/raw/` - Downloaded videos
- `data/frames/` - Extracted frames
- `data/annotations/` - COCO JSON outputs
