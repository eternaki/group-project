"""
Head Pose Estimation из keypoints.

Вычисляет ориентацию головы собаки (yaw, pitch, roll) для фильтрации
не-фронтальных изображений. Только фронтальные морды дают надёжные AU.

Используется для:
- Фильтрации кадров в видео (пропуск повёрнутых голов)
- Оценки качества детекции keypoints
- Валидации перед вычислением эмоций
"""

from dataclasses import dataclass
import numpy as np
import math
from typing import Optional

from packages.data.schemas import NUM_KEYPOINTS


# Индексы keypoints для head pose
KP_LEFT_EYE = 0
KP_RIGHT_EYE = 1
KP_NOSE = 2
KP_LEFT_EAR_BASE = 3
KP_RIGHT_EAR_BASE = 4
KP_LEFT_EAR_TIP = 5
KP_RIGHT_EAR_TIP = 6
KP_FOREHEAD = 14


@dataclass
class HeadPose:
    """
    Результат оценки позы головы.

    Attributes:
        yaw: Поворот влево/вправо в градусах (-90 до +90)
        pitch: Наклон вверх/вниз в градусах (-90 до +90)
        roll: Наклон набок в градусах (-90 до +90)
        is_frontal: True если голова достаточно фронтальная
        confidence: Уверенность оценки (0-1)
    """
    yaw: float      # Поворот влево/вправо
    pitch: float    # Наклон вверх/вниз
    roll: float     # Наклон набок
    is_frontal: bool
    confidence: float

    def to_dict(self) -> dict:
        """Конвертирует в словарь."""
        return {
            "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1),
            "roll": round(self.roll, 1),
            "is_frontal": self.is_frontal,
            "confidence": round(self.confidence, 3),
        }


class HeadPoseEstimator:
    """
    Оценивает позу головы собаки из keypoints.

    Алгоритм:
    - YAW: угол между носом и центром глаз относительно ширины глаз
    - PITCH: угол между носом и центром ушей относительно высоты головы
    - ROLL: угол наклона линии между ушами

    Example:
        estimator = HeadPoseEstimator(frontal_threshold=30)
        keypoints = np.array([x0, y0, v0, x1, y1, v1, ...])  # 60 values
        pose = estimator.estimate(keypoints)
        if pose.is_frontal:
            # Можно вычислять AU
    """

    def __init__(self, frontal_threshold: float = 30.0) -> None:
        """
        Инициализирует estimator.

        Args:
            frontal_threshold: Максимальный угол для фронтальной позы (градусы)
        """
        self.frontal_threshold = frontal_threshold

    def estimate(self, keypoints_flat: np.ndarray) -> HeadPose:
        """
        Оценивает позу головы из keypoints.

        Args:
            keypoints_flat: Array [x0, y0, v0, x1, y1, v1, ...] (60 values)

        Returns:
            HeadPose с yaw, pitch, roll и флагом is_frontal
        """
        # Reshape to (20, 3)
        kp = keypoints_flat.reshape(NUM_KEYPOINTS, 3)
        coords = kp[:, :2]  # (20, 2)
        visibility = kp[:, 2]  # (20,)

        # Вычислить confidence как среднюю visibility ключевых точек
        key_indices = [KP_LEFT_EYE, KP_RIGHT_EYE, KP_NOSE,
                       KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE]
        confidence = float(np.mean([visibility[i] for i in key_indices]))

        # Получить координаты ключевых точек
        left_eye = coords[KP_LEFT_EYE]
        right_eye = coords[KP_RIGHT_EYE]
        nose = coords[KP_NOSE]
        left_ear = coords[KP_LEFT_EAR_BASE]
        right_ear = coords[KP_RIGHT_EAR_BASE]
        forehead = coords[KP_FOREHEAD]

        # Вычислить YAW (поворот влево/вправо)
        eye_center = (left_eye + right_eye) / 2
        eye_width = self._distance(left_eye, right_eye)

        if eye_width > 1e-6:
            # Смещение носа от центра глаз
            nose_offset = nose[0] - eye_center[0]
            yaw = math.degrees(math.atan2(nose_offset, eye_width))
        else:
            yaw = 0.0

        # Вычислить PITCH (наклон вверх/вниз)
        ear_center = (left_ear + right_ear) / 2
        head_height = self._distance(forehead, nose)

        if head_height > 1e-6:
            # Вертикальное смещение носа
            vertical_offset = nose[1] - ear_center[1]
            pitch = math.degrees(math.atan2(vertical_offset, head_height))
        else:
            pitch = 0.0

        # Вычислить ROLL (наклон набок)
        ear_dx = right_ear[0] - left_ear[0]
        ear_dy = right_ear[1] - left_ear[1]
        roll = math.degrees(math.atan2(ear_dy, ear_dx))

        # Определить is_frontal
        is_frontal = (
            abs(yaw) < self.frontal_threshold and
            abs(pitch) < self.frontal_threshold and
            abs(roll) < self.frontal_threshold
        )

        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            is_frontal=is_frontal,
            confidence=confidence,
        )

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Вычисляет евклидово расстояние."""
        return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def estimate_head_pose(
    keypoints_flat: np.ndarray,
    frontal_threshold: float = 30.0,
) -> HeadPose:
    """
    Функция-хелпер для оценки позы головы.

    Args:
        keypoints_flat: Array [x0, y0, v0, ...] (60 values)
        frontal_threshold: Порог для фронтальной позы (градусы)

    Returns:
        HeadPose
    """
    estimator = HeadPoseEstimator(frontal_threshold)
    return estimator.estimate(keypoints_flat)


def validate_head_pose(
    pose: HeadPose,
    max_angle: float = 30.0,
    min_confidence: float = 0.5,
) -> bool:
    """
    Валидирует позу головы для вычисления AU.

    Args:
        pose: HeadPose для валидации
        max_angle: Максимально допустимый угол
        min_confidence: Минимальная уверенность

    Returns:
        True если поза валидна для AU
    """
    return (
        pose.is_frontal and
        pose.confidence >= min_confidence and
        abs(pose.yaw) <= max_angle and
        abs(pose.pitch) <= max_angle and
        abs(pose.roll) <= max_angle
    )
