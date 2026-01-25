"""
Pakiet modeli AI dla projektu Dog FACS.

Zawiera:
- BBoxModel: Detekcja psów (YOLOv8)
- BreedModel: Klasyfikacja ras (EfficientNet-B4)
- KeypointsModel: Detekcja punktów kluczowych (SimpleBaseline)
- EmotionModel: Klasyfikacja emocji (MLP lub Rule-based)
- HeadPose: Estymacja pozycji głowy (filtrowanie nie-frontalnych)
- ActionUnits: Ekstrakcja Action Units z keypoints
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
    classify_emotion_from_au,
    classify_emotion_from_keypoints,
)
from .head_pose import (
    HeadPose,
    HeadPoseEstimator,
    estimate_head_pose,
    validate_head_pose,
)
from .action_units import (
    ActionUnitsExtractor,
    ActionUnitsPrediction,
    ACTION_UNIT_NAMES,
    NUM_ACTION_UNITS,
    extract_action_units,
)

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
    # Emotion
    "EmotionConfig",
    "EmotionModel",
    "EmotionPrediction",
    "EMOTION_CLASSES",
    "NUM_EMOTIONS",
    "classify_emotion_from_au",
    "classify_emotion_from_keypoints",
    # Head Pose
    "HeadPose",
    "HeadPoseEstimator",
    "estimate_head_pose",
    "validate_head_pose",
    # Action Units
    "ActionUnitsExtractor",
    "ActionUnitsPrediction",
    "ACTION_UNIT_NAMES",
    "NUM_ACTION_UNITS",
    "extract_action_units",
]
