"""
Схемы данных для проекта Dog FACS Dataset.

Содержит определения keypoints для детекции ключевых точек на морде собаки.
DogFLW dataset - 46 facial landmarks.
"""

from dataclasses import dataclass, field


# Количество keypoints из DogFLW dataset
NUM_KEYPOINTS: int = 46


# Названия keypoints (46 точек из DogFLW)
KEYPOINT_NAMES: list[str] = [f"landmark_{i}" for i in range(NUM_KEYPOINTS)]


# Основные соединения для визуализации (упрощённый skeleton)
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    # Глаза
    (0, 1),
    # Контур морды
    (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
    # Нос
    (14, 15), (15, 16),
]


@dataclass
class Keypoint:
    """Одна ключевая точка."""
    x: float
    y: float
    visibility: float = 1.0  # 0 = невидим, 0.5 = частично, 1.0 = видим


@dataclass
class KeypointsAnnotation:
    """Аннотация keypoints для одного изображения."""
    image_id: str
    keypoints: list[Keypoint] = field(default_factory=list)

    def to_coco_format(self) -> list[float]:
        """Конвертирует в формат COCO: [x1, y1, v1, x2, y2, v2, ...]"""
        result = []
        for kp in self.keypoints:
            result.extend([kp.x, kp.y, kp.visibility])
        return result

    @classmethod
    def from_coco_format(cls, image_id: str, keypoints_flat: list[float]) -> "KeypointsAnnotation":
        """Создаёт из формата COCO."""
        keypoints = []
        for i in range(0, len(keypoints_flat), 3):
            keypoints.append(Keypoint(
                x=keypoints_flat[i],
                y=keypoints_flat[i + 1],
                visibility=keypoints_flat[i + 2],
            ))
        return cls(image_id=image_id, keypoints=keypoints)


def get_keypoint_color(index: int) -> tuple[int, int, int]:
    """Возвращает цвет для keypoint по индексу."""
    # Разные цвета для разных групп
    colors = [
        (0, 255, 0),      # Зелёный - глаза (0-1)
        (0, 255, 0),
        (255, 0, 0),      # Красный - контур (2-13)
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (0, 0, 255),      # Синий - нос (14-19)
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (255, 255, 0),    # Жёлтый - рот (20-31)
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (255, 0, 255),    # Пурпурный - остальные (32-45)
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
        (255, 0, 255),
    ]
    if index < len(colors):
        return colors[index]
    return (128, 128, 128)
