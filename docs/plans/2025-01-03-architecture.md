# Architecture Design Document

**Project:** Dog FACS Dataset
**Version:** 1.0
**Date:** 2025-01-03

---

## 1. Overview

This document describes the technical architecture for the Dog FACS annotation pipeline and dataset creation system.

### 1.1 System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dog FACS System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │  YouTube │───▶│   Scripts    │───▶│   Data Storage        │ │
│  │  Videos  │    │  (Download)  │    │   (data/raw/)         │ │
│  └──────────┘    └──────────────┘    └───────────────────────┘ │
│                                                │                │
│                                                ▼                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    AI Pipeline                            │  │
│  │  ┌────────┐  ┌────────┐  ┌───────────┐  ┌─────────────┐  │  │
│  │  │  BBox  │─▶│ Breed  │  │ Keypoints │─▶│  Emotion    │  │  │
│  │  │ (YOLO) │  │ (ViT)  │  │  (HRNet)  │  │ (DogFACS)   │  │  │
│  │  └────────┘  └────────┘  └───────────┘  └─────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                │                │
│                                                ▼                │
│  ┌───────────────────────┐    ┌──────────────────────────────┐ │
│  │   COCO Exporter       │───▶│   Dataset (annotations/)     │ │
│  └───────────────────────┘    └──────────────────────────────┘ │
│                                                │                │
│                                                ▼                │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                  Streamlit Demo App                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Package Structure

```
packages/
├── models/
│   ├── __init__.py          # Model registry and factory
│   ├── base.py              # Abstract base class
│   ├── bbox.py              # YOLOv8 dog detector
│   ├── breed.py             # ViT/EfficientNet classifier
│   ├── keypoints.py         # HRNet keypoint detector
│   └── emotion.py           # Emotion classifier
├── pipeline/
│   ├── __init__.py
│   ├── inference.py         # Single frame inference
│   ├── video.py             # Video frame extraction
│   └── batch.py             # Batch processing
└── data/
    ├── __init__.py
    ├── coco.py              # COCO format handler
    ├── schemas.py           # Data classes
    └── transforms.py        # Image preprocessing
```

### 2.2 Model Interface

All models implement a common interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelConfig:
    weights_path: str
    device: str = "cuda"
    confidence_threshold: float = 0.5

class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> dict:
        """Run inference and return predictions."""
        pass

    @abstractmethod
    def postprocess(self, predictions: dict) -> dict:
        """Format predictions to standard output."""
        pass
```

---

## 3. Data Flow

### 3.1 Inference Pipeline

```
┌─────────────┐
│ Input Image │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ BBox Detection   │
│ (YOLOv8)         │
├──────────────────┤
│ Output:          │
│ - bbox [x,y,w,h] │
│ - confidence     │
└────────┬─────────┘
         │
         ▼
┌────────────────────┐
│ Crop Dog Region    │
└────────┬───────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌────────────┐
│ Breed  │  │ Keypoints  │
│ (ViT)  │  │ (HRNet)    │
├────────┤  ├────────────┤
│Output: │  │Output:     │
│-breed  │  │-20 points  │
│-top5   │  │-visibility │
└────────┘  └─────┬──────┘
                  │
                  ▼
           ┌────────────┐
           │  Emotion   │
           │ (DogFACS)  │
           ├────────────┤
           │Output:     │
           │-emotion    │
           │-confidence │
           └────────────┘
                  │
                  ▼
           ┌────────────┐
           │ Annotation │
           │ (Combined) │
           └────────────┘
```

### 3.2 Pipeline Code Structure

```python
# packages/pipeline/inference.py

from dataclasses import dataclass
from typing import Optional
import numpy as np

from packages.models import BBoxModel, BreedModel, KeypointsModel, EmotionModel
from packages.data.schemas import Annotation, BBox, Keypoints

@dataclass
class PipelineConfig:
    bbox_weights: str
    breed_weights: str
    keypoints_weights: str
    emotion_weights: str
    device: str = "cuda"

class InferencePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.bbox_model: Optional[BBoxModel] = None
        self.breed_model: Optional[BreedModel] = None
        self.keypoints_model: Optional[KeypointsModel] = None
        self.emotion_model: Optional[EmotionModel] = None

    def load(self) -> None:
        """Load all models."""
        self.bbox_model = BBoxModel(self.config.bbox_weights)
        self.breed_model = BreedModel(self.config.breed_weights)
        self.keypoints_model = KeypointsModel(self.config.keypoints_weights)
        self.emotion_model = EmotionModel(self.config.emotion_weights)

        for model in [self.bbox_model, self.breed_model,
                      self.keypoints_model, self.emotion_model]:
            model.load()

    def process_frame(self, image: np.ndarray) -> list[Annotation]:
        """Process single frame through all models."""
        annotations = []

        # Step 1: Detect dogs
        detections = self.bbox_model.predict(image)

        for det in detections:
            # Step 2: Crop dog region
            crop = self._crop_image(image, det.bbox)

            # Step 3: Classify breed
            breed = self.breed_model.predict(crop)

            # Step 4: Detect keypoints
            keypoints = self.keypoints_model.predict(crop)

            # Step 5: Classify emotion (using keypoints)
            emotion = self.emotion_model.predict(keypoints.features)

            # Combine annotation
            annotation = Annotation(
                bbox=det.bbox,
                confidence_bbox=det.confidence,
                breed_id=breed.class_id,
                breed_name=breed.class_name,
                confidence_breed=breed.confidence,
                keypoints=keypoints.points,
                visibility=keypoints.visibility,
                emotion_id=emotion.class_id,
                emotion_name=emotion.class_name,
                confidence_emotion=emotion.confidence
            )
            annotations.append(annotation)

        return annotations

    def _crop_image(self, image: np.ndarray, bbox: BBox) -> np.ndarray:
        """Crop image to bounding box region."""
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
```

---

## 4. Data Schemas

### 4.1 Internal Data Classes

```python
# packages/data/schemas.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]

@dataclass
class Keypoint:
    x: float
    y: float
    visibility: int  # 0=not labeled, 1=labeled but not visible, 2=visible

@dataclass
class Keypoints:
    points: list[Keypoint]
    names: list[str] = field(default_factory=lambda: [
        "left_eye", "right_eye", "nose", "left_ear_base", "right_ear_base",
        "left_ear_tip", "right_ear_tip", "left_mouth_corner", "right_mouth_corner",
        "upper_lip", "lower_lip", "chin", "left_cheek", "right_cheek",
        "forehead", "left_eyebrow", "right_eyebrow", "muzzle_top",
        "muzzle_left", "muzzle_right"
    ])

    def to_coco_format(self) -> list[float]:
        """Convert to COCO keypoints format [x1,y1,v1,x2,y2,v2,...]"""
        result = []
        for point in self.points:
            result.extend([point.x, point.y, point.visibility])
        return result

@dataclass
class Annotation:
    bbox: BBox
    confidence_bbox: float
    breed_id: int
    breed_name: str
    confidence_breed: float
    keypoints: Keypoints
    emotion_id: int
    emotion_name: str
    confidence_emotion: float
    image_id: Optional[int] = None
    annotation_id: Optional[int] = None
```

### 4.2 COCO Format Handler

```python
# packages/data/coco.py

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .schemas import Annotation, Keypoints

@dataclass
class COCOImage:
    id: int
    file_name: str
    width: int
    height: int
    source_video: Optional[str] = None
    frame_number: Optional[int] = None

@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: list[int]
    area: float
    keypoints: list[float]
    num_keypoints: int
    breed_id: int
    emotion_id: int
    confidence: dict
    iscrowd: int = 0

@dataclass
class COCOCategory:
    id: int
    name: str
    supercategory: str
    keypoints: list[str]
    skeleton: list[list[int]]

class COCODataset:
    def __init__(self):
        self.info = {
            "description": "Dog FACS Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Gdansk University of Technology",
            "url": "https://github.com/eternaki/group-project"
        }
        self.licenses = []
        self.images: list[COCOImage] = []
        self.annotations: list[COCOAnnotation] = []
        self.categories: list[COCOCategory] = []

        self._init_categories()

    def _init_categories(self):
        """Initialize dog category with keypoints."""
        keypoint_names = Keypoints([]).names
        skeleton = [
            [0, 1],   # left_eye - right_eye
            [0, 2],   # left_eye - nose
            [1, 2],   # right_eye - nose
            [3, 5],   # left_ear_base - left_ear_tip
            [4, 6],   # right_ear_base - right_ear_tip
            [7, 9],   # left_mouth - upper_lip
            [8, 9],   # right_mouth - upper_lip
            [9, 10],  # upper_lip - lower_lip
        ]

        self.categories.append(COCOCategory(
            id=1,
            name="dog",
            supercategory="animal",
            keypoints=keypoint_names,
            skeleton=skeleton
        ))

    def add_image(self, image: COCOImage) -> None:
        self.images.append(image)

    def add_annotation(self, annotation: Annotation, image_id: int) -> int:
        """Add annotation and return annotation ID."""
        ann_id = len(self.annotations) + 1

        coco_ann = COCOAnnotation(
            id=ann_id,
            image_id=image_id,
            category_id=1,  # dog
            bbox=annotation.bbox.to_list(),
            area=annotation.bbox.width * annotation.bbox.height,
            keypoints=annotation.keypoints.to_coco_format(),
            num_keypoints=len(annotation.keypoints.points),
            breed_id=annotation.breed_id,
            emotion_id=annotation.emotion_id,
            confidence={
                "bbox": annotation.confidence_bbox,
                "breed": annotation.confidence_breed,
                "emotion": annotation.confidence_emotion
            }
        )

        self.annotations.append(coco_ann)
        return ann_id

    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        data = {
            "info": self.info,
            "licenses": self.licenses,
            "images": [asdict(img) for img in self.images],
            "annotations": [asdict(ann) for ann in self.annotations],
            "categories": [asdict(cat) for cat in self.categories]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "COCODataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)

        dataset = cls()
        dataset.info = data["info"]
        dataset.licenses = data["licenses"]
        dataset.images = [COCOImage(**img) for img in data["images"]]
        dataset.annotations = [COCOAnnotation(**ann) for ann in data["annotations"]]

        return dataset
```

---

## 5. Model Specifications

### 5.1 BBox Detection (YOLOv8)

| Property | Value |
|----------|-------|
| Architecture | YOLOv8m |
| Input Size | 640x640 |
| Output | Bounding boxes + confidence |
| Base Weights | yolov8m.pt (COCO pretrained) |
| Fine-tuning Data | Dog images from Stanford Dogs, Open Images |

### 5.2 Breed Classification (ViT)

| Property | Value |
|----------|-------|
| Architecture | ViT-B/16 or EfficientNet-B4 |
| Input Size | 224x224 |
| Output | Class probabilities (50+ breeds) |
| Base Weights | ImageNet pretrained |
| Fine-tuning Data | Stanford Dogs Dataset |

### 5.3 Keypoints Detection (HRNet)

| Property | Value |
|----------|-------|
| Architecture | HRNet-W32 |
| Input Size | 256x256 |
| Output | 20 keypoints with visibility |
| Base Weights | COCO Keypoints pretrained |
| Fine-tuning Data | Kaggle DogFLW |

### 5.4 Emotion Classification

| Property | Value |
|----------|-------|
| Architecture | MLP or small CNN |
| Input | Keypoint features (flattened) |
| Output | 6 emotion classes |
| Classes | happy, sad, angry, fearful, relaxed, neutral |
| Training Data | HuggingFace emotion datasets |

---

## 6. Demo Application Architecture

### 6.1 Streamlit App Structure

```
apps/demo/
├── __init__.py
├── app.py              # Main entry point
├── config.py           # App configuration
└── components/
    ├── __init__.py
    ├── upload.py       # File upload widget
    ├── viewer.py       # Image/video viewer
    ├── results.py      # Results display
    └── export.py       # Export functionality
```

### 6.2 App Flow

```python
# apps/demo/app.py

import streamlit as st
from pathlib import Path

from packages.pipeline import InferencePipeline, PipelineConfig
from .components import upload, viewer, results, export

st.set_page_config(page_title="Dog FACS Demo", layout="wide")

@st.cache_resource
def load_pipeline():
    config = PipelineConfig(
        bbox_weights="models/bbox.pt",
        breed_weights="models/breed.pt",
        keypoints_weights="models/keypoints.pt",
        emotion_weights="models/emotion.pt"
    )
    pipeline = InferencePipeline(config)
    pipeline.load()
    return pipeline

def main():
    st.title("Dog FACS - Emotion Analysis Demo")

    pipeline = load_pipeline()

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.header("Input")
        uploaded_file = upload.file_uploader()

        if uploaded_file:
            image = upload.load_image(uploaded_file)
            viewer.show_image(image, "Original")

    with col2:
        st.header("Results")

        if uploaded_file and st.button("Analyze"):
            with st.spinner("Processing..."):
                annotations = pipeline.process_frame(image)

            annotated_image = viewer.draw_annotations(image, annotations)
            viewer.show_image(annotated_image, "Annotated")

            results.show_details(annotations)
            export.download_button(annotations)

if __name__ == "__main__":
    main()
```

---

## 7. Batch Processing

### 7.1 Video Processing Pipeline

```python
# packages/pipeline/video.py

import cv2
from pathlib import Path
from typing import Iterator
import numpy as np

class VideoProcessor:
    def __init__(self, fps_sample: float = 1.0):
        self.fps_sample = fps_sample

    def extract_frames(self, video_path: Path) -> Iterator[tuple[int, np.ndarray]]:
        """Extract frames at specified FPS."""
        cap = cv2.VideoCapture(str(video_path))

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / self.fps_sample)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_idx, frame_rgb

            frame_idx += 1

        cap.release()
```

### 7.2 Batch Annotation Script

```python
# scripts/annotation/batch_annotate.py

from pathlib import Path
from tqdm import tqdm

from packages.pipeline import InferencePipeline, PipelineConfig
from packages.pipeline.video import VideoProcessor
from packages.data.coco import COCODataset, COCOImage

def batch_annotate(
    video_dir: Path,
    output_path: Path,
    config: PipelineConfig
) -> None:
    """Process all videos and create COCO dataset."""

    pipeline = InferencePipeline(config)
    pipeline.load()

    video_processor = VideoProcessor(fps_sample=1.0)
    dataset = COCODataset()

    video_files = list(video_dir.glob("*.mp4"))
    image_id = 0

    for video_path in tqdm(video_files, desc="Processing videos"):
        for frame_idx, frame in video_processor.extract_frames(video_path):
            image_id += 1

            # Save frame
            frame_name = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            # ... save frame to disk ...

            # Add image to dataset
            h, w = frame.shape[:2]
            dataset.add_image(COCOImage(
                id=image_id,
                file_name=frame_name,
                width=w,
                height=h,
                source_video=video_path.name,
                frame_number=frame_idx
            ))

            # Process frame
            annotations = pipeline.process_frame(frame)

            # Add annotations
            for ann in annotations:
                dataset.add_annotation(ann, image_id)

    # Save dataset
    dataset.save(output_path)
```

---

## 8. Configuration Management

### 8.1 Environment Variables

```bash
# .env.example
DEVICE=cuda
BBOX_WEIGHTS=models/bbox.pt
BREED_WEIGHTS=models/breed.pt
KEYPOINTS_WEIGHTS=models/keypoints.pt
EMOTION_WEIGHTS=models/emotion.pt
DATA_DIR=data/
```

### 8.2 Config Files

```yaml
# configs/pipeline.yaml
models:
  bbox:
    architecture: yolov8m
    weights: models/bbox.pt
    confidence_threshold: 0.5

  breed:
    architecture: vit_b_16
    weights: models/breed.pt
    num_classes: 50

  keypoints:
    architecture: hrnet_w32
    weights: models/keypoints.pt
    num_keypoints: 20

  emotion:
    architecture: mlp
    weights: models/emotion.pt
    num_classes: 6

processing:
  device: cuda
  batch_size: 8
  fps_sample: 1.0
```

---

## 9. Testing Strategy

### 9.1 Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Fixtures
├── test_models/
│   ├── test_bbox.py
│   ├── test_breed.py
│   ├── test_keypoints.py
│   └── test_emotion.py
├── test_pipeline/
│   ├── test_inference.py
│   └── test_video.py
└── test_data/
    ├── test_coco.py
    └── test_schemas.py
```

### 9.2 Test Examples

```python
# tests/test_pipeline/test_inference.py

import pytest
import numpy as np
from packages.pipeline import InferencePipeline

@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def pipeline(tmp_path):
    # Use mock weights for testing
    config = PipelineConfig(...)
    pipeline = InferencePipeline(config)
    pipeline.load()
    return pipeline

def test_process_frame_returns_annotations(pipeline, sample_image):
    annotations = pipeline.process_frame(sample_image)
    assert isinstance(annotations, list)

def test_annotation_has_required_fields(pipeline, sample_image):
    annotations = pipeline.process_frame(sample_image)
    if annotations:
        ann = annotations[0]
        assert hasattr(ann, 'bbox')
        assert hasattr(ann, 'breed_id')
        assert hasattr(ann, 'keypoints')
        assert hasattr(ann, 'emotion_id')
```

---

## 10. Deployment

### 10.1 Local Development

```bash
# Install
pip install -e ".[dev]"

# Run demo
streamlit run apps/demo/app.py

# Run tests
pytest
```

### 10.2 Docker (Optional)

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install .

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "apps/demo/app.py"]
```

---

## 11. Future Considerations

- Multi-dog detection support
- Real-time video streaming
- Model quantization for faster inference
- Web API (FastAPI) for remote access
- DVC integration for data versioning
