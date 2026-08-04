"""
Estymacja pozy głowy psa z keypoints.

Oblicza orientację głowy (obrót lewo/prawo, przechylenie) do filtrowania
niefrontalnych klatek. Tylko frontalne mordy dają wiarygodne AU.

Uwaga: klasyczna miara "pitch" (kąt nos–linia oczu) NIE jest tu używana.
U psa nos jest zawsze poniżej oczu — to anatomia pyska, nie poza głowy.
Na ręcznie anotowanym DogFLW (480 obrazów) taka miara dawała medianę +47.5°
i 91.5% wartości powyżej progu 30°, mimo że mordy są frontalne. Zastąpiona
asymetrią odległości kąciki oczu <-> nos, niezależną od długości pyska
i od skali obrazu.

Używane do:
- Filtrowania klatek w wideo (pomijanie obróconej/przechylonej głowy)
- Oceny jakości detekcji keypoints
- Walidacji przed obliczaniem emocji
"""

import math
from dataclasses import dataclass

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS

# Domyślne progi frontalności (patrz uzasadnienie w docstringu modułu)
DEFAULT_MAX_YAW_ASYMMETRY = 0.35
DEFAULT_MAX_ROLL = 30.0


@dataclass
class HeadPose:
    """
    Wynik estymacji pozy głowy.

    Attributes:
        yaw_asymmetry: Obrót lewo/prawo jako asymetria odległości kącik oka ↔ nos,
            zakres [-1, 1], 0 = morda frontalna. Ujemna wartość = nos bliżej
            LEWEGO oka (głowa obrócona w stronę lewego oka), dodatnia = nos
            bliżej PRAWEGO oka (obrót w stronę prawego oka). Metryka
            bezwymiarowa, niezależna od długości pyska i od skali obrazu.
        roll: Przechylenie w stopniach (kąt linii wewnętrznych kącików oczu do osi X)
        is_frontal: True gdy oba kąty mieszczą się w limitach
        confidence: Pewność estymacji (0-1)
    """

    yaw_asymmetry: float
    roll: float
    is_frontal: bool
    confidence: float

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "yaw_asymmetry": round(self.yaw_asymmetry, 3),
            "roll": round(self.roll, 1),
            "is_frontal": self.is_frontal,
            "confidence": round(self.confidence, 3),
        }


class HeadPoseEstimator:
    """
    Estymuje pozę głowy psa z keypoints (46 punktów DogFLW).

    Algorytm:
    - yaw_asymmetry: asymetria odległości kącik oka (wewnętrzny) <-> nos
    - roll: kąt przechylenia linii między wewnętrznymi kącikami oczu

    Przykład:
        >>> estimator = HeadPoseEstimator(max_yaw_asymmetry=0.35, max_roll=30.0)
        >>> keypoints = np.zeros(138)  # 46 × 3 wartości
        >>> pose = estimator.estimate(keypoints)
        >>> if pose.is_frontal:
        ...     pass  # można obliczać AU
    """

    def __init__(
        self,
        max_yaw_asymmetry: float = DEFAULT_MAX_YAW_ASYMMETRY,
        max_roll: float = DEFAULT_MAX_ROLL,
    ) -> None:
        """
        Inicjalizuje estimator.

        Args:
            max_yaw_asymmetry: Maksymalna asymetria kącik oka <-> nos dla frontalnej pozy
            max_roll: Maksymalny kąt przechylenia dla frontalnej pozy (stopnie)
        """
        self.max_yaw_asymmetry = max_yaw_asymmetry
        self.max_roll = max_roll

    def estimate(self, keypoints_flat: np.ndarray) -> HeadPose:
        """
        Estymuje pozę głowy z keypoints.

        Args:
            keypoints_flat: Array [x0, y0, v0, ...] (138 wartości = 46×3)

        Returns:
            HeadPose z yaw_asymmetry, roll i flagą is_frontal

        Raises:
            ValueError: Gdy liczba wartości keypoints jest nieprawidłowa
        """
        expected = NUM_KEYPOINTS * 3
        if len(keypoints_flat) != expected:
            raise ValueError(
                f"Oczekiwano {expected} wartości keypoints, "
                f"otrzymano {len(keypoints_flat)}"
            )

        kp = keypoints_flat.reshape(NUM_KEYPOINTS, 3)
        coords = kp[:, :2]
        visibility = kp[:, 2]

        left_eye_inner = coords[KP.LEFT_EYE_INNER]
        right_eye_inner = coords[KP.RIGHT_EYE_INNER]
        nose = coords[KP.NOSE_TIP]

        yaw_asymmetry = _compute_yaw_asymmetry(left_eye_inner, right_eye_inner, nose)
        roll = _compute_roll(left_eye_inner, right_eye_inner)

        is_frontal = (
            abs(yaw_asymmetry) <= self.max_yaw_asymmetry and abs(roll) <= self.max_roll
        )

        # Pewność = średnia widoczność kluczowych punktów
        key_indices = [
            KP.LEFT_EYE_INNER, KP.RIGHT_EYE_INNER,
            KP.NOSE_TIP,
            KP.LEFT_EAR_BASE_FRONT, KP.RIGHT_EAR_BASE_FRONT,
        ]
        confidence = float(np.mean([visibility[i] for i in key_indices]))

        return HeadPose(
            yaw_asymmetry=yaw_asymmetry,
            roll=roll,
            is_frontal=is_frontal,
            confidence=confidence,
        )


def estimate_head_pose(
    keypoints_flat: np.ndarray,
    max_yaw_asymmetry: float = DEFAULT_MAX_YAW_ASYMMETRY,
    max_roll: float = DEFAULT_MAX_ROLL,
) -> HeadPose:
    """
    Funkcja pomocnicza do estymacji pozy głowy.

    Args:
        keypoints_flat: Array [x0, y0, v0, ...] (138 wartości)
        max_yaw_asymmetry: Próg asymetrii kącik oka <-> nos dla frontalnej pozy
        max_roll: Próg przechylenia dla frontalnej pozy (stopnie)

    Returns:
        HeadPose
    """
    estimator = HeadPoseEstimator(max_yaw_asymmetry, max_roll)
    return estimator.estimate(keypoints_flat)


def validate_head_pose(
    pose: HeadPose,
    max_yaw_asymmetry: float = DEFAULT_MAX_YAW_ASYMMETRY,
    min_confidence: float = 0.5,
) -> bool:
    """
    Waliduje pozę głowy dla obliczania AU.

    Args:
        pose: HeadPose do walidacji
        max_yaw_asymmetry: Maksymalnie dopuszczalna asymetria kącik oka <-> nos
        min_confidence: Minimalna pewność

    Returns:
        True jeśli poza jest prawidłowa dla AU
    """
    return (
        pose.is_frontal
        and pose.confidence >= min_confidence
        and abs(pose.yaw_asymmetry) <= max_yaw_asymmetry
    )


# =============================================================================
# Funkcje pomocnicze (prywatne)
# =============================================================================

def _euclidean_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Odległość euklidesowa między dwoma punktami."""
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def _compute_yaw_asymmetry(
    left_eye_inner: np.ndarray,
    right_eye_inner: np.ndarray,
    nose: np.ndarray,
) -> float:
    """
    Liczy obrót głowy jako asymetrię odległości od kącików oczu do nosa.

    Miara „nos poniżej oczu" nie nadaje się dla psów: nos jest poniżej oczu zawsze,
    niezależnie od pozy (na referencji DogFLW mediana takiej miary to +47.5°).
    Asymetria lewo/prawo jest zerowa dla mordy frontalnej przy dowolnej długości pyska.
    """
    left_distance = _euclidean_dist(left_eye_inner, nose)
    right_distance = _euclidean_dist(right_eye_inner, nose)
    total = left_distance + right_distance
    if total < 1e-6:
        return 0.0
    return float((left_distance - right_distance) / total)


def _compute_roll(left_eye_inner: np.ndarray, right_eye_inner: np.ndarray) -> float:
    """Liczy przechylenie z linii wewnętrznych kącików oczu."""
    dx = right_eye_inner[0] - left_eye_inner[0]
    dy = right_eye_inner[1] - left_eye_inner[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return float(np.clip(math.degrees(math.atan2(dy, dx)), -90, 90))
