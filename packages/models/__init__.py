"""
Pakiet modeli AI dla projektu Dog FACS.

Zawiera:
- BBoxModel: Detekcja psów (YOLOv8)
- BreedModel: Klasyfikacja ras (EfficientNet-B4)
- KeypointsModel: Detekcja punktów kluczowych (do implementacji)
- EmotionModel: Klasyfikacja emocji (do implementacji)
"""

from .base import BaseModel, ModelConfig
from .bbox import BBoxConfig, BBoxModel, Detection
from .breed import BreedConfig, BreedModel, BreedPrediction

__all__ = [
    "BaseModel",
    "ModelConfig",
    "BBoxConfig",
    "BBoxModel",
    "Detection",
    "BreedConfig",
    "BreedModel",
    "BreedPrediction",
]
