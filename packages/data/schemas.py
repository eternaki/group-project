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
# Kolejność ZGODNA z oficjalnym schematem DogFLW (martvelge/DogFLW, arXiv:2405.11501).
# To jest kolejność kanałów wyjściowych modelu keypoints_dogflw.pt — zweryfikowana
# empirycznie (uśrednienie pozycji kanałów po 66 frontalnych mordach).
KEYPOINT_NAMES: list[str] = [
    # === Uszy (0-13) ===
    "ear_left_top_base",     # 0  - górna (medialna) podstawa lewego ucha
    "ear_right_top_base",    # 1  - górna (medialna) podstawa prawego ucha
    "ear_left_upper_bend",   # 2  - górne zagięcie lewego ucha
    "ear_right_upper_bend",  # 3  - górne zagięcie prawego ucha
    "ear_left_upper_mid",    # 4  - środek górnej krawędzi lewego ucha
    "ear_right_upper_mid",   # 5  - środek górnej krawędzi prawego ucha
    "ear_left_tip",          # 6  - czubek lewego ucha
    "ear_right_tip",         # 7  - czubek prawego ucha
    "ear_left_lower_mid",    # 8  - środek dolnej krawędzi lewego ucha
    "ear_right_lower_mid",   # 9  - środek dolnej krawędzi prawego ucha
    "ear_left_lower_23",     # 10 - dolne 2/3 lewego ucha
    "ear_right_lower_23",    # 11 - dolne 2/3 prawego ucha
    "ear_left_bottom_base",  # 12 - dolna (lateralna) podstawa lewego ucha
    "ear_right_bottom_base", # 13 - dolna (lateralna) podstawa prawego ucha

    # === Brwi (14-15) ===
    "brow_left",             # 14 - lewa brew (poduszka wąsów brwi)
    "brow_right",            # 15 - prawa brew

    # === Oczy (16-23) ===
    "eye_left_inner",        # 16 - wewnętrzny kąt lewego oka
    "eye_right_inner",       # 17 - wewnętrzny kąt prawego oka
    "eye_left_outer",        # 18 - zewnętrzny kąt lewego oka
    "eye_right_outer",       # 19 - zewnętrzny kąt prawego oka
    "eye_left_upper",        # 20 - środek górnej powieki lewego oka
    "eye_right_upper",       # 21 - środek górnej powieki prawego oka
    "eye_left_lower",        # 22 - środek dolnej powieki lewego oka
    "eye_right_lower",       # 23 - środek dolnej powieki prawego oka

    # === Nos / pysk (24-37) ===
    "nose_pad_mid",          # 24 - środek między stopem a noskiem
    "nose_upper",            # 25 - górny środek nosa
    "nose_left_upper_edge",  # 26 - lewa górna krawędź nosa
    "nose_right_upper_edge", # 27 - prawa górna krawędź nosa
    "snout_left",            # 28 - lewa strona pyska
    "snout_right",           # 29 - prawa strona pyska
    "zygoma_left",           # 30 - lewa kość jarzmowa (policzek)
    "zygoma_right",          # 31 - prawa kość jarzmowa (policzek)
    "nostril_mid",           # 32 - między nozdrzami
    "nostril_left_outer",    # 33 - zewnętrzny kąt lewego nozdrza
    "nostril_right_outer",   # 34 - zewnętrzny kąt prawego nozdrza
    "nose_bottom",           # 35 - dolny środek nosa (czubek)
    "whisker_pad_left",      # 36 - środek lewej poduszki wąsów
    "whisker_pad_right",     # 37 - środek prawej poduszki wąsów

    # === Pasek / wargi / podbródek / język (38-45) ===
    "lip_upper_mid",         # 38 - środek górnej wargi (pod nosem)
    "lip_left_corner",       # 39 - lewy kącik ust
    "lip_right_corner",      # 40 - prawy kącik ust
    "lip_lower_mid",         # 41 - środek dolnej wargi
    "chin",                  # 42 - środek podbródka
    "lip_left_upper_mid",    # 43 - lewy punkt między podbródkiem a kącikiem
    "lip_right_upper_mid",   # 44 - prawy punkt między podbródkiem a kącikiem
    "tongue_tip",            # 45 - czubek języka
]

assert len(KEYPOINT_NAMES) == NUM_KEYPOINTS, (
    f"Liczba nazw keypoints ({len(KEYPOINT_NAMES)}) "
    f"musi być równa NUM_KEYPOINTS ({NUM_KEYPOINTS})"
)


# Indeksy kluczowych punktów dla czytelności kodu
# Używane przez DeltaActionUnitsExtractor
class KP:
    """Indeksy keypoints dla obliczeń AU.

    UWAGA: wartości odpowiadają KANONICZNEJ kolejności DogFLW (kolejność kanałów
    modelu). Nazwy atrybutów są zachowane dla zgodności z kodem AU, a wskazują
    najbliższy anatomicznie punkt DogFLW. Niektóre nazwy (np. brwi inner/center/
    outer) wskazują ten sam punkt, bo DogFLW ma jeden punkt brwi na stronę.
    """

    # Oczy (DogFLW 16-23)
    LEFT_EYE_INNER: int = 16
    LEFT_EYE_TOP: int = 20
    LEFT_EYE_OUTER: int = 18
    LEFT_EYE_BOTTOM: int = 22
    RIGHT_EYE_INNER: int = 17
    RIGHT_EYE_TOP: int = 21
    RIGHT_EYE_OUTER: int = 19
    RIGHT_EYE_BOTTOM: int = 23

    # Brwi (DogFLW 14-15 — jeden punkt na stronę)
    LEFT_BROW_INNER: int = 14
    LEFT_BROW_CENTER: int = 14
    LEFT_BROW_OUTER: int = 14
    RIGHT_BROW_INNER: int = 15
    RIGHT_BROW_CENTER: int = 15
    RIGHT_BROW_OUTER: int = 15

    # Uszy (DogFLW 0-13)
    LEFT_EAR_BASE_FRONT: int = 0
    LEFT_EAR_BASE_BACK: int = 12
    LEFT_EAR_MID: int = 4
    LEFT_EAR_TIP: int = 6
    RIGHT_EAR_BASE_FRONT: int = 1
    RIGHT_EAR_BASE_BACK: int = 13
    RIGHT_EAR_MID: int = 5
    RIGHT_EAR_TIP: int = 7

    # Nos (DogFLW): czubek = nose_bottom(35), skrzydła = nozdrza(33/34), grzbiet = nose_upper(25)
    NOSE_TIP: int = 35
    NOSE_LEFT_WING: int = 33
    NOSE_RIGHT_WING: int = 34
    NOSE_BRIDGE: int = 25

    # Usta (DogFLW 38-44)
    MOUTH_LEFT_CORNER: int = 39
    UPPER_LIP_LEFT: int = 43
    UPPER_LIP_CENTER: int = 38
    UPPER_LIP_RIGHT: int = 44
    MOUTH_RIGHT_CORNER: int = 40
    LOWER_LIP_RIGHT: int = 44
    LOWER_LIP_CENTER: int = 41
    LOWER_LIP_LEFT: int = 43

    # Pysk (DogFLW): górny stop = nose_pad_mid(24), boki = snout(28/29), podbródek = chin(42)
    MUZZLE_TOP: int = 24
    MUZZLE_LEFT: int = 28
    MUZZLE_RIGHT: int = 29
    CHIN: int = 42

    # Kontur (DogFLW): policzki = zygoma(30/31), czoło proxy = nose_pad_mid/brwi
    FOREHEAD_CENTER: int = 24
    FOREHEAD_LEFT: int = 14
    FOREHEAD_RIGHT: int = 15
    LEFT_CHEEK_UPPER: int = 30
    LEFT_CHEEK_LOWER: int = 30
    RIGHT_CHEEK_UPPER: int = 31
    RIGHT_CHEEK_LOWER: int = 31
    JAW_CENTER: int = 42


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
    # Grupy zgodne z kolejnością DogFLW (zwracane jako RGB; konsumenci konwertują na BGR)
    if 0 <= index <= 13:    # Uszy
        return (255, 165, 0)    # pomarańczowy
    elif 14 <= index <= 15:  # Brwi
        return (255, 0, 128)    # różowy
    elif 16 <= index <= 23:  # Oczy
        return (0, 255, 0)      # zielony
    elif 24 <= index <= 37:  # Nos / pysk / policzki
        return (255, 48, 48)    # czerwony
    elif 38 <= index <= 45:  # Pasek / wargi / podbródek / język
        return (0, 255, 255)    # cyjan
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
