"""Модуль данных для проекта Dog FACS Dataset."""

from .schemas import (
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    SKELETON_CONNECTIONS,
    Keypoint,
    KeypointsAnnotation,
    get_keypoint_color,
)
from .coco import COCODataset, COCOInfo, COCOCategory

__all__ = [
    "Keypoint",
    "KeypointsAnnotation",
    "KEYPOINT_NAMES",
    "NUM_KEYPOINTS",
    "SKELETON_CONNECTIONS",
    "get_keypoint_color",
    "COCODataset",
    "COCOInfo",
    "COCOCategory",
]
