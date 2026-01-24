"""
Схемы данных для проекта Dog FACS Dataset.

Содержит определения 20 keypoints для детекции ключевых точек на морде собаки.
Согласно спецификации проекта (docs/sprints/4-keypoint-detection/stories/4.2).
"""

from dataclasses import dataclass, field


# Количество keypoints согласно спецификации проекта
NUM_KEYPOINTS: int = 20

# Количество keypoints в DogFLW dataset (для mapping)
NUM_KEYPOINTS_DOGFLW: int = 46


# Названия 20 keypoints согласно спецификации
KEYPOINT_NAMES: list[str] = [
    "left_eye",           # 0 - Left eye center
    "right_eye",          # 1 - Right eye center
    "nose",               # 2 - Nose tip
    "left_ear_base",      # 3 - Base of left ear
    "right_ear_base",     # 4 - Base of right ear
    "left_ear_tip",       # 5 - Tip of left ear
    "right_ear_tip",      # 6 - Tip of right ear
    "left_mouth_corner",  # 7 - Left corner of mouth
    "right_mouth_corner", # 8 - Right corner of mouth
    "upper_lip",          # 9 - Center of upper lip
    "lower_lip",          # 10 - Center of lower lip
    "chin",               # 11 - Chin point
    "left_cheek",         # 12 - Left cheek
    "right_cheek",        # 13 - Right cheek
    "forehead",           # 14 - Center forehead
    "left_eyebrow",       # 15 - Left eyebrow (inner)
    "right_eyebrow",      # 16 - Right eyebrow (inner)
    "muzzle_top",         # 17 - Top of muzzle
    "muzzle_left",        # 18 - Left side of muzzle
    "muzzle_right",       # 19 - Right side of muzzle
]


# Mapping z DogFLW 46 landmarks do 20 keypoints projektu
# Indeksy odpowiadają przybliżonym pozycjom anatomicznym
# Na podstawie grupowania w DogFLW:
# - 0-1: oczy, 2-13: kontur, 14-19: nos, 20-31: usta, 32-45: pozostałe
DOGFLW_TO_PROJECT_MAPPING: dict[int, int] = {
    # left_eye (0) - DogFLW landmark dla lewego oka
    0: 0,
    # right_eye (1) - DogFLW landmark dla prawego oka
    1: 1,
    # nose (2) - DogFLW landmark dla nosa
    14: 2,
    # left_ear_base (3) - podstawa lewego ucha
    32: 3,
    # right_ear_base (4) - podstawa prawego ucha
    36: 4,
    # left_ear_tip (5) - czubek lewego ucha
    34: 5,
    # right_ear_tip (6) - czubek prawego ucha
    38: 6,
    # left_mouth_corner (7) - lewy kącik ust
    20: 7,
    # right_mouth_corner (8) - prawy kącik ust
    24: 8,
    # upper_lip (9) - górna warga
    22: 9,
    # lower_lip (10) - dolna warga
    26: 10,
    # chin (11) - podbródek
    28: 11,
    # left_cheek (12) - lewy policzek
    4: 12,
    # right_cheek (13) - prawy policzek
    8: 13,
    # forehead (14) - czoło
    40: 14,
    # left_eyebrow (15) - lewa brew
    42: 15,
    # right_eyebrow (16) - prawa brew
    44: 16,
    # muzzle_top (17) - góra pyska
    16: 17,
    # muzzle_left (18) - lewa strona pyska
    6: 18,
    # muzzle_right (19) - prawa strona pyska
    10: 19,
}

# Odwrotny mapping: project index -> DogFLW index
PROJECT_TO_DOGFLW_MAPPING: dict[int, int] = {
    v: k for k, v in DOGFLW_TO_PROJECT_MAPPING.items()
}


# Skeleton connections dla wizualizacji - pełna forma pyska psa
# Struktura tworzy rozpoznawalny kształt twarzy
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    # === Oczy ===
    (0, 1),   # left_eye - right_eye (linia oczu)
    (0, 15),  # left_eye - left_eyebrow
    (1, 16),  # right_eye - right_eyebrow

    # === Brwi i czoło ===
    (15, 14),  # left_eyebrow - forehead
    (16, 14),  # right_eyebrow - forehead
    (15, 16),  # left_eyebrow - right_eyebrow (linia brwi)

    # === Uszy ===
    (3, 5),   # left_ear_base - left_ear_tip
    (4, 6),   # right_ear_base - right_ear_tip
    (3, 15),  # left_ear_base - left_eyebrow
    (4, 16),  # right_ear_base - right_eyebrow

    # === Nos i pysk ===
    (0, 2),   # left_eye - nose
    (1, 2),   # right_eye - nose
    (14, 17), # forehead - muzzle_top (linia środkowa)
    (17, 2),  # muzzle_top - nose
    (2, 18),  # nose - muzzle_left
    (2, 19),  # nose - muzzle_right

    # === Policzki i kontur ===
    (0, 12),  # left_eye - left_cheek
    (1, 13),  # right_eye - right_cheek
    (12, 18), # left_cheek - muzzle_left
    (13, 19), # right_cheek - muzzle_right
    (12, 7),  # left_cheek - left_mouth_corner
    (13, 8),  # right_cheek - right_mouth_corner

    # === Usta ===
    (7, 9),   # left_mouth_corner - upper_lip
    (8, 9),   # right_mouth_corner - upper_lip
    (7, 10),  # left_mouth_corner - lower_lip
    (8, 10),  # right_mouth_corner - lower_lip
    (2, 9),   # nose - upper_lip
    (9, 10),  # upper_lip - lower_lip

    # === Podbródek ===
    (10, 11), # lower_lip - chin
    (7, 11),  # left_mouth_corner - chin
    (8, 11),  # right_mouth_corner - chin
]


@dataclass
class Keypoint:
    """Jedna kluczowa punkta."""
    x: float
    y: float
    visibility: float = 1.0  # 0 = niewidoczny, 0.5 = częściowo, 1.0 = widoczny


@dataclass
class KeypointsAnnotation:
    """Anotacja keypoints dla jednego obrazu."""
    image_id: str
    keypoints: list[Keypoint] = field(default_factory=list)

    def to_coco_format(self) -> list[float]:
        """Konwertuje do formatu COCO: [x1, y1, v1, x2, y2, v2, ...]"""
        result = []
        for kp in self.keypoints:
            result.extend([kp.x, kp.y, kp.visibility])
        return result

    @classmethod
    def from_coco_format(cls, image_id: str, keypoints_flat: list[float]) -> "KeypointsAnnotation":
        """Tworzy z formatu COCO."""
        keypoints = []
        for i in range(0, len(keypoints_flat), 3):
            keypoints.append(Keypoint(
                x=keypoints_flat[i],
                y=keypoints_flat[i + 1],
                visibility=keypoints_flat[i + 2],
            ))
        return cls(image_id=image_id, keypoints=keypoints)


def map_dogflw_to_project(dogflw_keypoints: list[Keypoint]) -> list[Keypoint]:
    """
    Mapuje 46 keypoints z DogFLW do 20 keypoints projektu.

    Args:
        dogflw_keypoints: Lista 46 keypoints z modelu DogFLW

    Returns:
        Lista 20 keypoints zgodnie ze specyfikacją projektu
    """
    if len(dogflw_keypoints) != NUM_KEYPOINTS_DOGFLW:
        raise ValueError(
            f"Oczekiwano {NUM_KEYPOINTS_DOGFLW} keypoints z DogFLW, "
            f"otrzymano: {len(dogflw_keypoints)}"
        )

    project_keypoints = []
    for project_idx in range(NUM_KEYPOINTS):
        dogflw_idx = PROJECT_TO_DOGFLW_MAPPING[project_idx]
        project_keypoints.append(dogflw_keypoints[dogflw_idx])

    return project_keypoints


def get_keypoint_color(index: int) -> tuple[int, int, int]:
    """Zwraca kolor dla keypoint po indeksie."""
    # Różne kolory dla różnych grup anatomicznych
    if index in [0, 1]:  # Oczy
        return (0, 255, 0)  # Zielony
    elif index == 2:  # Nos
        return (0, 0, 255)  # Niebieski
    elif index in [3, 4, 5, 6]:  # Uszy
        return (255, 165, 0)  # Pomarańczowy
    elif index in [7, 8, 9, 10, 11]:  # Usta/podbródek
        return (255, 255, 0)  # Żółty
    elif index in [12, 13]:  # Policzki
        return (255, 0, 0)  # Czerwony
    elif index in [14, 15, 16]:  # Czoło/brwi
        return (128, 0, 255)  # Fioletowy
    elif index in [17, 18, 19]:  # Pysk
        return (255, 0, 255)  # Magenta
    else:
        return (128, 128, 128)  # Szary (domyślny)


def get_keypoint_name(index: int) -> str:
    """Zwraca nazwę keypoint po indeksie."""
    if 0 <= index < NUM_KEYPOINTS:
        return KEYPOINT_NAMES[index]
    return f"unknown_{index}"
