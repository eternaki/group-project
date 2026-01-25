"""
Temporal Processing для стабильных предсказаний эмоций.

Агрегирует Action Units по времени (30 кадров = 1 секунда @ 30 FPS)
для получения стабильных предсказаний вместо шумных покадровых.

Процесс:
1. Получить keypoints для кадра
2. Проверить head_pose (отфильтровать не-фронтальные)
3. Вычислить AU
4. Добавить в буфер
5. Вернуть усреднённые AU когда буфер стабилен
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from packages.models.head_pose import HeadPose, estimate_head_pose, validate_head_pose
from packages.models.action_units import (
    ActionUnitsExtractor,
    ActionUnitsPrediction,
    ACTION_UNIT_NAMES,
    NUM_ACTION_UNITS,
)


@dataclass
class TemporalAUResult:
    """
    Результат временной агрегации AU.

    Attributes:
        values: Усреднённые значения AU
        variance: Дисперсия каждого AU
        num_frames: Количество кадров в буфере
        is_stable: True если дисперсия ниже порога
    """
    values: dict[str, float]
    variance: dict[str, float]
    num_frames: int
    is_stable: bool

    def to_feature_vector(self) -> np.ndarray:
        """Конвертирует в feature vector для модели."""
        return np.array(
            [self.values[name] for name in ACTION_UNIT_NAMES],
            dtype=np.float32
        )


class TemporalAUBuffer:
    """
    Буфер для временной агрегации Action Units.

    Хранит историю AU за последние N кадров и вычисляет
    усреднённые значения и дисперсию.

    Example:
        buffer = TemporalAUBuffer(window_size=30)
        buffer.add_frame(au_values, confidence=0.8)
        if buffer.is_ready():
            result = buffer.get_aggregated()
            if result.is_stable:
                # Использовать result.values для эмоций
    """

    def __init__(
        self,
        window_size: int = 30,
        stability_threshold: float = 0.15,
        min_frames: int = 10,
    ) -> None:
        """
        Инициализирует буфер.

        Args:
            window_size: Размер окна (количество кадров)
            stability_threshold: Порог дисперсии для стабильности
            min_frames: Минимум кадров для начала агрегации
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.min_frames = min_frames

        # История: list of (au_dict, confidence)
        self._history: deque = deque(maxlen=window_size)

    def add_frame(
        self,
        au_values: dict[str, float],
        confidence: float = 1.0,
    ) -> None:
        """
        Добавляет кадр в буфер.

        Args:
            au_values: Словарь AU_name -> value
            confidence: Уверенность детекции (для взвешивания)
        """
        self._history.append((au_values.copy(), confidence))

    def clear(self) -> None:
        """Очищает буфер."""
        self._history.clear()

    def is_ready(self) -> bool:
        """Проверяет есть ли достаточно кадров."""
        return len(self._history) >= self.min_frames

    def get_aggregated(self) -> Optional[TemporalAUResult]:
        """
        Возвращает усреднённые AU.

        Returns:
            TemporalAUResult или None если недостаточно кадров
        """
        if not self.is_ready():
            return None

        # Собрать все значения AU
        au_arrays = {name: [] for name in ACTION_UNIT_NAMES}
        weights = []

        for au_values, confidence in self._history:
            weights.append(confidence)
            for name in ACTION_UNIT_NAMES:
                au_arrays[name].append(au_values.get(name, 0.0))

        # Вычислить взвешенное среднее и дисперсию
        weights = np.array(weights)
        weights = weights / weights.sum()  # Нормализовать

        mean_values = {}
        variance_values = {}

        for name in ACTION_UNIT_NAMES:
            values = np.array(au_arrays[name])
            mean_values[name] = float(np.average(values, weights=weights))
            variance_values[name] = float(np.var(values))

        # Проверить стабильность
        max_variance = max(variance_values.values())
        is_stable = max_variance < self.stability_threshold

        return TemporalAUResult(
            values=mean_values,
            variance=variance_values,
            num_frames=len(self._history),
            is_stable=is_stable,
        )


class TemporalProcessor:
    """
    Процессор для временной обработки видео.

    Объединяет:
    - Head pose estimation (фильтрация не-фронтальных)
    - Action Units extraction
    - Temporal aggregation

    Example:
        processor = TemporalProcessor()

        for frame_keypoints in video_keypoints:
            result = processor.process_frame(frame_keypoints)
            if result is not None and result.is_stable:
                # Классифицировать эмоцию из result.values
    """

    def __init__(
        self,
        window_size: int = 30,
        head_pose_threshold: float = 30.0,
        min_visibility: float = 0.5,
        stability_threshold: float = 0.15,
    ) -> None:
        """
        Инициализирует процессор.

        Args:
            window_size: Размер окна агрегации (кадры)
            head_pose_threshold: Максимальный угол для фронтальной позы
            min_visibility: Минимальная средняя visibility keypoints
            stability_threshold: Порог дисперсии AU для стабильности
        """
        self.head_pose_threshold = head_pose_threshold
        self.min_visibility = min_visibility

        self._au_extractor = ActionUnitsExtractor()
        self._buffer = TemporalAUBuffer(
            window_size=window_size,
            stability_threshold=stability_threshold,
        )

        # Статистика
        self._total_frames = 0
        self._accepted_frames = 0
        self._rejected_head_pose = 0
        self._rejected_visibility = 0

    def reset(self) -> None:
        """Сбрасывает состояние процессора."""
        self._buffer.clear()
        self._total_frames = 0
        self._accepted_frames = 0
        self._rejected_head_pose = 0
        self._rejected_visibility = 0

    def process_frame(
        self,
        keypoints_flat: np.ndarray,
    ) -> Optional[TemporalAUResult]:
        """
        Обрабатывает один кадр.

        Args:
            keypoints_flat: Keypoints [x0, y0, v0, ...] (60 values)

        Returns:
            TemporalAUResult если буфер готов, иначе None
        """
        self._total_frames += 1

        # 1. Проверить visibility
        kp = keypoints_flat.reshape(-1, 3)
        mean_visibility = float(np.mean(kp[:, 2]))

        if mean_visibility < self.min_visibility:
            self._rejected_visibility += 1
            return self._buffer.get_aggregated() if self._buffer.is_ready() else None

        # 2. Проверить head pose
        head_pose = estimate_head_pose(keypoints_flat, self.head_pose_threshold)

        if not validate_head_pose(head_pose, self.head_pose_threshold, self.min_visibility):
            self._rejected_head_pose += 1
            return self._buffer.get_aggregated() if self._buffer.is_ready() else None

        # 3. Извлечь AU
        au_prediction = self._au_extractor.extract(keypoints_flat)

        # 4. Добавить в буфер
        self._buffer.add_frame(au_prediction.values, head_pose.confidence)
        self._accepted_frames += 1

        # 5. Вернуть агрегированный результат
        return self._buffer.get_aggregated()

    def process_video_sequence(
        self,
        keypoints_sequence: list[np.ndarray],
    ) -> Optional[TemporalAUResult]:
        """
        Обрабатывает последовательность keypoints из видео.

        Args:
            keypoints_sequence: Список keypoints для каждого кадра

        Returns:
            Финальный TemporalAUResult или None
        """
        self.reset()
        result = None

        for keypoints in keypoints_sequence:
            result = self.process_frame(keypoints)

        return result

    def get_statistics(self) -> dict:
        """Возвращает статистику обработки."""
        return {
            "total_frames": self._total_frames,
            "accepted_frames": self._accepted_frames,
            "rejected_head_pose": self._rejected_head_pose,
            "rejected_visibility": self._rejected_visibility,
            "acceptance_rate": (
                self._accepted_frames / self._total_frames
                if self._total_frames > 0 else 0.0
            ),
        }
