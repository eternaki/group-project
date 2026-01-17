"""
Pakiet modeli AI dla projektu Dog FACS.

Zawiera:
- BBoxModel: Detekcja psów (YOLOv8)
- BreedModel: Klasyfikacja ras (EfficientNet-B4)
- KeypointsModel: Detekcja punktów kluczowych (HRNet/SimpleBaseline)
- EmotionModel: Klasyfikacja emocji (do implementacji)
"""

from .base import BaseModel, ModelConfig
from .bbox import BBoxConfig, BBoxModel, Detection
from .breed import BreedConfig, BreedModel, BreedPrediction
from .keypoints import KeypointsConfig, KeypointsModel, KeypointsPrediction

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
]
