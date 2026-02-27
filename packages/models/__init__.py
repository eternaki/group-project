"""
Pakiet modeli AI dla projektu Dog FACS.

Zawiera:
- BBoxModel: Detekcja psów (YOLOv8)
- BreedModel: Klasyfikacja ras (EfficientNet-B4)
- KeypointsModel: Detekcja 46 punktów kluczowych (SimpleBaseline / DogFLW)
- DogFACSRuleEngine: Klasyfikacja emocji oparta na regułach DogFACS (bez ML)
- HeadPoseEstimator: Estymacja pozycji głowy
- DeltaActionUnitsExtractor: Ekstrakcja 21 Action Units z keypoints (delta vs. neutralna)
"""

from .base import BaseModel, ModelConfig
from .bbox import BBoxConfig, BBoxModel, Detection
from .breed import BreedConfig, BreedModel, BreedPrediction
from .delta_action_units import (
    ACTION_UNIT_NAMES,
    NUM_ACTION_UNITS,
    DeltaActionUnit,
    DeltaActionUnitsExtractor,
    extract_delta_action_units,
)
from .emotion import (
    EMOTION_CLASSES,
    EMOTION_RULES,
    NUM_EMOTIONS,
    DogFACSRuleEngine,
    EmotionPrediction,
    EmotionRule,
    classify_emotion_from_delta_aus,
)
from .head_pose import (
    HeadPose,
    HeadPoseEstimator,
    estimate_head_pose,
    validate_head_pose,
)
from .keypoints import KeypointsConfig, KeypointsModel, KeypointsPrediction

__all__ = [
    # Base
    "BaseModel",
    "ModelConfig",
    # BBox
    "BBoxConfig",
    "BBoxModel",
    "Detection",
    # Breed
    "BreedConfig",
    "BreedModel",
    "BreedPrediction",
    # Keypoints
    "KeypointsConfig",
    "KeypointsModel",
    "KeypointsPrediction",
    # Emocje (rule-based, bez ML)
    "EmotionPrediction",
    "EmotionRule",
    "DogFACSRuleEngine",
    "EMOTION_CLASSES",
    "NUM_EMOTIONS",
    "EMOTION_RULES",
    "classify_emotion_from_delta_aus",
    # Poza głowy
    "HeadPose",
    "HeadPoseEstimator",
    "estimate_head_pose",
    "validate_head_pose",
    # Delta Action Units (21 AU DogFACS)
    "DeltaActionUnit",
    "DeltaActionUnitsExtractor",
    "ACTION_UNIT_NAMES",
    "NUM_ACTION_UNITS",
    "extract_delta_action_units",
]
