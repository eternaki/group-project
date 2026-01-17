"""
Pakiet modeli AI dla projektu Dog FACS.

Zawiera:
- BBoxModel: Detekcja psów (YOLOv8)
- BreedModel: Klasyfikacja ras (EfficientNet-B4)
- KeypointsModel: Detekcja punktów kluczowych (SimpleBaseline)
- EmotionModel: Klasyfikacja emocji (EfficientNet-B0)
"""

from .base import BaseModel, ModelConfig
from .bbox import BBoxConfig, BBoxModel, Detection
from .breed import BreedConfig, BreedModel, BreedPrediction
from .keypoints import KeypointsConfig, KeypointsModel, KeypointsPrediction
from .emotion import (
    EmotionConfig,
    EmotionModel,
    EmotionPrediction,
    EMOTION_CLASSES,
    NUM_EMOTIONS,
)

__all__ = [
    "BaseModel",
    "ModelConfig",
    "BBoxConfig",
    "BBoxModel",
    "Detection",
    "BreedConfig",
    "BreedModel",
    "BreedPrediction",
    "KeypointsConfig",
    "KeypointsModel",
    "KeypointsPrediction",
    "EmotionConfig",
    "EmotionModel",
    "EmotionPrediction",
    "EMOTION_CLASSES",
    "NUM_EMOTIONS",
]
