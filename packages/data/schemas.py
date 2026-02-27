"""
Schematy danych dla projektu Dog FACS Dataset.

Zawiera definicje 46 keypoints zgodnie ze schematem DogFLW (Dog Facial Landmarks),
który jest podstawą do obliczania Action Units (AU) w systemie DogFACS.

Źródło: DogFLW — psi odpowiednik punktów kluczowych twarzy człowieka.
"""

from dataclasses import dataclass, field

# Liczba keypoints zgodnie ze schematem DogFLW
NUM_KEYPOINTS: int = 46


# Nazwy 46 keypoints według schematu DogFLW
# Grupowanie anatomiczne: oczy, brwi, uszy, nos, usta, pysk, kontur
KEYPOINT_NAMES: list[str] = [
    # === Oczy (0-7) ===
    "left_eye_inner",        # 0  - wewnętrzny kąt lewego oka
    "left_eye_top",          # 1  - górna powieka lewego oka
    "left_eye_outer",        # 2  - zewnętrzny kąt lewego oka
    "left_eye_bottom",       # 3  - dolna powieka lewego oka
    "right_eye_inner",       # 4  - wewnętrzny kąt prawego oka
    "right_eye_top",         # 5  - górna powieka prawego oka
    "right_eye_outer",       # 6  - zewnętrzny kąt prawego oka
    "right_eye_bottom",      # 7  - dolna powieka prawego oka

    # === Brwi (8-13) ===
    "left_brow_inner",       # 8  - wewnętrzny punkt lewej brwi
    "left_brow_center",      # 9  - środek lewej brwi
    "left_brow_outer",       # 10 - zewnętrzny punkt lewej brwi
    "right_brow_inner",      # 11 - wewnętrzny punkt prawej brwi
    "right_brow_center",     # 12 - środek prawej brwi
    "right_brow_outer",      # 13 - zewnętrzny punkt prawej brwi

    # === Uszy (14-21) ===
    "left_ear_base_front",   # 14 - przednia podstawa lewego ucha
    "left_ear_base_back",    # 15 - tylna podstawa lewego ucha
    "left_ear_mid",          # 16 - środek lewego ucha
    "left_ear_tip",          # 17 - czubek lewego ucha
    "right_ear_base_front",  # 18 - przednia podstawa prawego ucha
    "right_ear_base_back",   # 19 - tylna podstawa prawego ucha
    "right_ear_mid",         # 20 - środek prawego ucha
    "right_ear_tip",         # 21 - czubek prawego ucha

    # === Nos (22-25) ===
    "nose_tip",              # 22 - czubek nosa
    "nose_left_wing",        # 23 - lewe skrzydło nosa
    "nose_right_wing",       # 24 - prawe skrzydło nosa
    "nose_bridge",           # 25 - grzbiet nosa

    # === Usta i wargi (26-33) ===
    "mouth_left_corner",     # 26 - lewy kącik ust
    "upper_lip_left",        # 27 - górna warga lewa
    "upper_lip_center",      # 28 - środek górnej wargi
    "upper_lip_right",       # 29 - górna warga prawa
    "mouth_right_corner",    # 30 - prawy kącik ust
    "lower_lip_right",       # 31 - dolna warga prawa
    "lower_lip_center",      # 32 - środek dolnej wargi
    "lower_lip_left",        # 33 - dolna warga lewa

    # === Pysk / muzzle (34-37) ===
    "muzzle_top",            # 34 - góra pyska
    "muzzle_left",           # 35 - lewa strona pyska
    "muzzle_right",          # 36 - prawa strona pyska
    "chin",                  # 37 - podbródek

    # === Kontur twarzy (38-45) ===
    "forehead_center",       # 38 - środek czoła
    "forehead_left",         # 39 - lewe czoło
    "forehead_right",        # 40 - prawe czoło
    "left_cheek_upper",      # 41 - górny lewy policzek
    "left_cheek_lower",      # 42 - dolny lewy policzek
    "right_cheek_upper",     # 43 - górny prawy policzek
    "right_cheek_lower",     # 44 - dolny prawy policzek
    "jaw_center",            # 45 - środek żuchwy
]

assert len(KEYPOINT_NAMES) == NUM_KEYPOINTS, (
    f"Liczba nazw keypoints ({len(KEYPOINT_NAMES)}) "
    f"musi być równa NUM_KEYPOINTS ({NUM_KEYPOINTS})"
)


# Indeksy kluczowych punktów dla czytelności kodu
# Używane przez DeltaActionUnitsExtractor
class KP:
    """Indeksy keypoints dla łatwego dostępu w obliczeniach AU."""

    # Oczy — centrum (dla AU)
    LEFT_EYE_INNER: int = 0
    LEFT_EYE_TOP: int = 1
    LEFT_EYE_OUTER: int = 2
    LEFT_EYE_BOTTOM: int = 3
    RIGHT_EYE_INNER: int = 4
    RIGHT_EYE_TOP: int = 5
    RIGHT_EYE_OUTER: int = 6
    RIGHT_EYE_BOTTOM: int = 7

    # Brwi
    LEFT_BROW_INNER: int = 8
    LEFT_BROW_CENTER: int = 9
    LEFT_BROW_OUTER: int = 10
    RIGHT_BROW_INNER: int = 11
    RIGHT_BROW_CENTER: int = 12
    RIGHT_BROW_OUTER: int = 13

    # Uszy
    LEFT_EAR_BASE_FRONT: int = 14
    LEFT_EAR_BASE_BACK: int = 15
    LEFT_EAR_MID: int = 16
    LEFT_EAR_TIP: int = 17
    RIGHT_EAR_BASE_FRONT: int = 18
    RIGHT_EAR_BASE_BACK: int = 19
    RIGHT_EAR_MID: int = 20
    RIGHT_EAR_TIP: int = 21

    # Nos
    NOSE_TIP: int = 22
    NOSE_LEFT_WING: int = 23
    NOSE_RIGHT_WING: int = 24
    NOSE_BRIDGE: int = 25

    # Usta
    MOUTH_LEFT_CORNER: int = 26
    UPPER_LIP_LEFT: int = 27
    UPPER_LIP_CENTER: int = 28
    UPPER_LIP_RIGHT: int = 29
    MOUTH_RIGHT_CORNER: int = 30
    LOWER_LIP_RIGHT: int = 31
    LOWER_LIP_CENTER: int = 32
    LOWER_LIP_LEFT: int = 33

    # Pysk
    MUZZLE_TOP: int = 34
    MUZZLE_LEFT: int = 35
    MUZZLE_RIGHT: int = 36
    CHIN: int = 37

    # Kontur
    FOREHEAD_CENTER: int = 38
    FOREHEAD_LEFT: int = 39
    FOREHEAD_RIGHT: int = 40
    LEFT_CHEEK_UPPER: int = 41
    LEFT_CHEEK_LOWER: int = 42
    RIGHT_CHEEK_UPPER: int = 43
    RIGHT_CHEEK_LOWER: int = 44
    JAW_CENTER: int = 45


# Połączenia szkieletu do wizualizacji
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    # === Oczy ===
    (KP.LEFT_EYE_INNER, KP.LEFT_EYE_TOP),
    (KP.LEFT_EYE_TOP, KP.LEFT_EYE_OUTER),
    (KP.LEFT_EYE_OUTER, KP.LEFT_EYE_BOTTOM),
    (KP.LEFT_EYE_BOTTOM, KP.LEFT_EYE_INNER),
    (KP.RIGHT_EYE_INNER, KP.RIGHT_EYE_TOP),
    (KP.RIGHT_EYE_TOP, KP.RIGHT_EYE_OUTER),
    (KP.RIGHT_EYE_OUTER, KP.RIGHT_EYE_BOTTOM),
    (KP.RIGHT_EYE_BOTTOM, KP.RIGHT_EYE_INNER),

    # === Brwi ===
    (KP.LEFT_BROW_INNER, KP.LEFT_BROW_CENTER),
    (KP.LEFT_BROW_CENTER, KP.LEFT_BROW_OUTER),
    (KP.RIGHT_BROW_INNER, KP.RIGHT_BROW_CENTER),
    (KP.RIGHT_BROW_CENTER, KP.RIGHT_BROW_OUTER),

    # === Uszy ===
    (KP.LEFT_EAR_BASE_FRONT, KP.LEFT_EAR_MID),
    (KP.LEFT_EAR_MID, KP.LEFT_EAR_TIP),
    (KP.RIGHT_EAR_BASE_FRONT, KP.RIGHT_EAR_MID),
    (KP.RIGHT_EAR_MID, KP.RIGHT_EAR_TIP),

    # === Nos ===
    (KP.NOSE_BRIDGE, KP.NOSE_TIP),
    (KP.NOSE_LEFT_WING, KP.NOSE_TIP),
    (KP.NOSE_RIGHT_WING, KP.NOSE_TIP),

    # === Usta ===
    (KP.MOUTH_LEFT_CORNER, KP.UPPER_LIP_LEFT),
    (KP.UPPER_LIP_LEFT, KP.UPPER_LIP_CENTER),
    (KP.UPPER_LIP_CENTER, KP.UPPER_LIP_RIGHT),
    (KP.UPPER_LIP_RIGHT, KP.MOUTH_RIGHT_CORNER),
    (KP.MOUTH_LEFT_CORNER, KP.LOWER_LIP_LEFT),
    (KP.LOWER_LIP_LEFT, KP.LOWER_LIP_CENTER),
    (KP.LOWER_LIP_CENTER, KP.LOWER_LIP_RIGHT),
    (KP.LOWER_LIP_RIGHT, KP.MOUTH_RIGHT_CORNER),

    # === Pysk ===
    (KP.NOSE_TIP, KP.MUZZLE_TOP),
    (KP.MUZZLE_TOP, KP.UPPER_LIP_CENTER),
    (KP.MUZZLE_LEFT, KP.MOUTH_LEFT_CORNER),
    (KP.MUZZLE_RIGHT, KP.MOUTH_RIGHT_CORNER),
    (KP.CHIN, KP.LOWER_LIP_CENTER),

    # === Kontur twarzy ===
    (KP.FOREHEAD_CENTER, KP.FOREHEAD_LEFT),
    (KP.FOREHEAD_CENTER, KP.FOREHEAD_RIGHT),
    (KP.FOREHEAD_LEFT, KP.LEFT_BROW_OUTER),
    (KP.FOREHEAD_RIGHT, KP.RIGHT_BROW_OUTER),
    (KP.LEFT_CHEEK_UPPER, KP.LEFT_CHEEK_LOWER),
    (KP.RIGHT_CHEEK_UPPER, KP.RIGHT_CHEEK_LOWER),
    (KP.LEFT_CHEEK_LOWER, KP.JAW_CENTER),
    (KP.RIGHT_CHEEK_LOWER, KP.JAW_CENTER),
    (KP.JAW_CENTER, KP.CHIN),
]


# 9 klas emocji zgodnie z Mota-Rojas et al. 2021
EMOTION_CLASSES: list[str] = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "relaxed",
    "neutral",
    "surprise",
    "pain",
    "submission",
]


@dataclass
class Keypoint:
    """Jeden punkt kluczowy twarzy psa."""

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
    def from_coco_format(
        cls,
        image_id: str,
        keypoints_flat: list[float],
    ) -> "KeypointsAnnotation":
        """Tworzy z formatu COCO."""
        keypoints = [
            Keypoint(
                x=keypoints_flat[i],
                y=keypoints_flat[i + 1],
                visibility=keypoints_flat[i + 2],
            )
            for i in range(0, len(keypoints_flat), 3)
        ]
        return cls(image_id=image_id, keypoints=keypoints)


def get_keypoint_color(index: int) -> tuple[int, int, int]:
    """
    Zwraca kolor BGR dla keypoint według grupy anatomicznej.

    Args:
        index: Indeks keypoint (0-45)

    Returns:
        Kolor w formacie BGR (Blue, Green, Red)
    """
    if 0 <= index <= 7:    # Oczy
        return (0, 255, 0)
    elif 8 <= index <= 13:  # Brwi
        return (128, 0, 255)
    elif 14 <= index <= 21: # Uszy
        return (255, 165, 0)
    elif 22 <= index <= 25: # Nos
        return (0, 0, 255)
    elif 26 <= index <= 33: # Usta
        return (255, 255, 0)
    elif 34 <= index <= 37: # Pysk
        return (255, 0, 255)
    elif 38 <= index <= 45: # Kontur
        return (255, 0, 0)
    return (128, 128, 128)  # Domyślny szary


def get_keypoint_name(index: int) -> str:
    """
    Zwraca nazwę keypoint według indeksu.

    Args:
        index: Indeks keypoint (0-45)

    Returns:
        Nazwa keypoint lub 'unknown_N' jeśli poza zakresem
    """
    if 0 <= index < NUM_KEYPOINTS:
        return KEYPOINT_NAMES[index]
    return f"unknown_{index}"
