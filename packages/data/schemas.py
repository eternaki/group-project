"""
Схемы данных для проекта Dog FACS Dataset.

Содержит определения keypoints для детекции ключевых точек на морде собаки.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple


class KeypointIndex(IntEnum):
    """Индексы ключевых точек на морде собаки (20 точек)."""

    LEFT_EYE = 0
    RIGHT_EYE = 1
    NOSE = 2
    LEFT_EAR_BASE = 3
    RIGHT_EAR_BASE = 4
    LEFT_EAR_TIP = 5
    RIGHT_EAR_TIP = 6
    LEFT_MOUTH_CORNER = 7
    RIGHT_MOUTH_CORNER = 8
    UPPER_LIP = 9
    LOWER_LIP = 10
    CHIN = 11
    LEFT_CHEEK = 12
    RIGHT_CHEEK = 13
    FOREHEAD = 14
    LEFT_EYEBROW = 15
    RIGHT_EYEBROW = 16
    MUZZLE_TOP = 17
    MUZZLE_LEFT = 18
    MUZZLE_RIGHT = 19


# Названия keypoints на английском
KEYPOINT_NAMES: list[str] = [
    "left_eye",
    "right_eye",
    "nose",
    "left_ear_base",
    "right_ear_base",
    "left_ear_tip",
    "right_ear_tip",
    "left_mouth_corner",
    "right_mouth_corner",
    "upper_lip",
    "lower_lip",
    "chin",
    "left_cheek",
    "right_cheek",
    "forehead",
    "left_eyebrow",
    "right_eyebrow",
    "muzzle_top",
    "muzzle_left",
    "muzzle_right",
]

# Названия keypoints на русском (для документации)
KEYPOINT_NAMES_RU: list[str] = [
    "левый глаз",
    "правый глаз",
    "нос",
    "основание левого уха",
    "основание правого уха",
    "кончик левого уха",
    "кончик правого уха",
    "левый угол рта",
    "правый угол рта",
    "верхняя губа",
    "нижняя губа",
    "подбородок",
    "левая щека",
    "правая щека",
    "лоб",
    "левая бровь",
    "правая бровь",
    "верх морды",
    "левая сторона морды",
    "правая сторона морды",
]

# Количество ключевых точек
NUM_KEYPOINTS: int = 20

# Соединения между точками для визуализации (skeleton)
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    # Глаза
    (KeypointIndex.LEFT_EYE, KeypointIndex.RIGHT_EYE),
    (KeypointIndex.LEFT_EYE, KeypointIndex.NOSE),
    (KeypointIndex.RIGHT_EYE, KeypointIndex.NOSE),
    (KeypointIndex.LEFT_EYE, KeypointIndex.LEFT_EYEBROW),
    (KeypointIndex.RIGHT_EYE, KeypointIndex.RIGHT_EYEBROW),

    # Уши
    (KeypointIndex.LEFT_EAR_BASE, KeypointIndex.LEFT_EAR_TIP),
    (KeypointIndex.RIGHT_EAR_BASE, KeypointIndex.RIGHT_EAR_TIP),
    (KeypointIndex.LEFT_EAR_BASE, KeypointIndex.FOREHEAD),
    (KeypointIndex.RIGHT_EAR_BASE, KeypointIndex.FOREHEAD),

    # Рот и морда
    (KeypointIndex.NOSE, KeypointIndex.UPPER_LIP),
    (KeypointIndex.UPPER_LIP, KeypointIndex.LOWER_LIP),
    (KeypointIndex.LOWER_LIP, KeypointIndex.CHIN),
    (KeypointIndex.LEFT_MOUTH_CORNER, KeypointIndex.UPPER_LIP),
    (KeypointIndex.RIGHT_MOUTH_CORNER, KeypointIndex.UPPER_LIP),

    # Морда (muzzle)
    (KeypointIndex.NOSE, KeypointIndex.MUZZLE_TOP),
    (KeypointIndex.MUZZLE_LEFT, KeypointIndex.NOSE),
    (KeypointIndex.MUZZLE_RIGHT, KeypointIndex.NOSE),

    # Щёки
    (KeypointIndex.LEFT_CHEEK, KeypointIndex.LEFT_MOUTH_CORNER),
    (KeypointIndex.RIGHT_CHEEK, KeypointIndex.RIGHT_MOUTH_CORNER),

    # Лоб
    (KeypointIndex.FOREHEAD, KeypointIndex.LEFT_EYEBROW),
    (KeypointIndex.FOREHEAD, KeypointIndex.RIGHT_EYEBROW),
]

# Цвета для визуализации групп keypoints (RGB)
KEYPOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "eyes": (0, 255, 0),       # Зелёный - глаза
    "ears": (255, 165, 0),     # Оранжевый - уши
    "nose": (255, 0, 0),       # Красный - нос
    "mouth": (255, 0, 255),    # Магента - рот
    "face": (0, 255, 255),     # Циан - лицо
}

# Группировка keypoints по областям
KEYPOINT_GROUPS: dict[str, list[int]] = {
    "eyes": [
        KeypointIndex.LEFT_EYE,
        KeypointIndex.RIGHT_EYE,
        KeypointIndex.LEFT_EYEBROW,
        KeypointIndex.RIGHT_EYEBROW,
    ],
    "ears": [
        KeypointIndex.LEFT_EAR_BASE,
        KeypointIndex.RIGHT_EAR_BASE,
        KeypointIndex.LEFT_EAR_TIP,
        KeypointIndex.RIGHT_EAR_TIP,
    ],
    "nose": [
        KeypointIndex.NOSE,
        KeypointIndex.MUZZLE_TOP,
        KeypointIndex.MUZZLE_LEFT,
        KeypointIndex.MUZZLE_RIGHT,
    ],
    "mouth": [
        KeypointIndex.LEFT_MOUTH_CORNER,
        KeypointIndex.RIGHT_MOUTH_CORNER,
        KeypointIndex.UPPER_LIP,
        KeypointIndex.LOWER_LIP,
    ],
    "face": [
        KeypointIndex.CHIN,
        KeypointIndex.LEFT_CHEEK,
        KeypointIndex.RIGHT_CHEEK,
        KeypointIndex.FOREHEAD,
    ],
}


class Keypoint(NamedTuple):
    """Одна ключевая точка с координатами и видимостью."""

    x: float
    y: float
    visibility: float  # 0 = не видна, 1 = видна, 0.5 = частично


@dataclass
class KeypointsAnnotation:
    """Аннотация keypoints для одного изображения."""

    image_id: str
    keypoints: list[Keypoint]
    bbox: tuple[float, float, float, float] | None = None  # x, y, w, h

    def __post_init__(self) -> None:
        """Проверка количества keypoints."""
        if len(self.keypoints) != NUM_KEYPOINTS:
            raise ValueError(
                f"Ожидается {NUM_KEYPOINTS} keypoints, получено {len(self.keypoints)}"
            )

    def to_flat_array(self) -> list[float]:
        """
        Преобразует keypoints в плоский массив [x1, y1, v1, x2, y2, v2, ...].

        Returns:
            Список из 60 значений (20 keypoints * 3)
        """
        result = []
        for kp in self.keypoints:
            result.extend([kp.x, kp.y, kp.visibility])
        return result

    @classmethod
    def from_flat_array(
        cls,
        image_id: str,
        flat_array: list[float],
        bbox: tuple[float, float, float, float] | None = None,
    ) -> "KeypointsAnnotation":
        """
        Создаёт KeypointsAnnotation из плоского массива.

        Args:
            image_id: Идентификатор изображения
            flat_array: Плоский массив [x1, y1, v1, x2, y2, v2, ...]
            bbox: Опциональный bounding box

        Returns:
            KeypointsAnnotation
        """
        if len(flat_array) != NUM_KEYPOINTS * 3:
            raise ValueError(
                f"Ожидается {NUM_KEYPOINTS * 3} значений, получено {len(flat_array)}"
            )

        keypoints = []
        for i in range(NUM_KEYPOINTS):
            x = flat_array[i * 3]
            y = flat_array[i * 3 + 1]
            v = flat_array[i * 3 + 2]
            keypoints.append(Keypoint(x, y, v))

        return cls(image_id=image_id, keypoints=keypoints, bbox=bbox)

    def get_visible_keypoints(self) -> list[tuple[int, Keypoint]]:
        """
        Возвращает только видимые keypoints.

        Returns:
            Список кортежей (индекс, Keypoint) для видимых точек
        """
        return [
            (i, kp) for i, kp in enumerate(self.keypoints)
            if kp.visibility > 0.5
        ]


def get_keypoint_name(index: int) -> str:
    """
    Возвращает название keypoint по индексу.

    Args:
        index: Индекс keypoint (0-19)

    Returns:
        Название keypoint
    """
    if 0 <= index < NUM_KEYPOINTS:
        return KEYPOINT_NAMES[index]
    raise ValueError(f"Неверный индекс keypoint: {index}")


def get_keypoint_color(index: int) -> tuple[int, int, int]:
    """
    Возвращает цвет для визуализации keypoint по индексу.

    Args:
        index: Индекс keypoint (0-19)

    Returns:
        RGB цвет
    """
    for group_name, indices in KEYPOINT_GROUPS.items():
        if index in indices:
            return KEYPOINT_COLORS[group_name]
    return (255, 255, 255)  # Белый по умолчанию
