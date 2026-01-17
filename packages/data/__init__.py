"""Модуль данных для проекта Dog FACS Dataset."""

from .schemas import (
    KEYPOINT_COLORS,
    KEYPOINT_GROUPS,
    KEYPOINT_NAMES,
    KEYPOINT_NAMES_RU,
    NUM_KEYPOINTS,
    SKELETON_CONNECTIONS,
    Keypoint,
    KeypointIndex,
    KeypointsAnnotation,
    get_keypoint_color,
    get_keypoint_name,
)

__all__ = [
    "Keypoint",
    "KeypointIndex",
    "KeypointsAnnotation",
    "KEYPOINT_NAMES",
    "KEYPOINT_NAMES_RU",
    "KEYPOINT_COLORS",
    "KEYPOINT_GROUPS",
    "NUM_KEYPOINTS",
    "SKELETON_CONNECTIONS",
    "get_keypoint_name",
    "get_keypoint_color",
]
