"""
Estymacja pozy głowy psa z keypoints.

Oblicza orientację głowy (yaw, pitch, roll) do filtrowania
niefrونtalnych klatek. Tylko frontalne mordy dają wiarygodne AU.

Używane do:
- Filtrowania klatek w wideo (pomijanie obróconej głowy)
- Oceny jakości detekcji keypoints
- Walidacji przed obliczaniem emocji
"""

import math
from dataclasses import dataclass

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS


@dataclass
class HeadPose:
    """
    Wynik estymacji pozy głowy.

    Attributes:
        yaw: Obrót lewo/prawo w stopniach (-90 do +90)
        pitch: Nachylenie góra/dół w stopniach (-90 do +90)
        roll: Przechylenie na bok w stopniach (-90 do +90)
        is_frontal: True jeśli głowa jest wystarczająco frontalna
        confidence: Pewność estymacji (0-1)
    """

    yaw: float
    pitch: float
    roll: float
    is_frontal: bool
    confidence: float

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1),
            "roll": round(self.roll, 1),
            "is_frontal": self.is_frontal,
            "confidence": round(self.confidence, 3),
        }


class HeadPoseEstimator:
    """
    Estymuje pozę głowy psa z keypoints (46 punktów DogFLW).

    Algorytm:
    - YAW: kąt między nosem a centrum oczu względem szerokości oczu
    - PITCH: kąt między nosem a centrum uszu względem wysokości głowy
    - ROLL: kąt przechylenia linii między centrami oczu

    Przykład:
        >>> estimator = HeadPoseEstimator(frontal_threshold=30)
        >>> keypoints = np.zeros(138)  # 46 × 3 wartości
        >>> pose = estimator.estimate(keypoints)
        >>> if pose.is_frontal:
        ...     pass  # można obliczać AU
    """

    def __init__(self, frontal_threshold: float = 30.0) -> None:
        """
        Inicjalizuje estimator.

        Args:
            frontal_threshold: Maksymalny kąt dla frontalnej pozy (stopnie)
        """
        self.frontal_threshold = frontal_threshold

    def estimate(self, keypoints_flat: np.ndarray) -> HeadPose:
        """
        Estymuje pozę głowy z keypoints.

        Args:
            keypoints_flat: Array [x0, y0, v0, ...] (138 wartości = 46×3)

        Returns:
            HeadPose z yaw, pitch, roll i flagą is_frontal

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

        # Centrum oka = średnia kąta wewnętrznego i zewnętrznego
        left_eye = (coords[KP.LEFT_EYE_INNER] + coords[KP.LEFT_EYE_OUTER]) / 2
        right_eye = (coords[KP.RIGHT_EYE_INNER] + coords[KP.RIGHT_EYE_OUTER]) / 2
        nose = coords[KP.NOSE_TIP]

        eye_width = _euclidean_dist(left_eye, right_eye)

        # YAW: przesunięcie nosa od centrum oczu (+ = obrót w lewo)
        yaw = _compute_yaw(nose, left_eye, right_eye, eye_width)

        # PITCH: nachylenie góra/dół (nos względem centrum oczu)
        pitch = _compute_pitch(nose, left_eye, right_eye, eye_width)

        # ROLL: przechylenie na bok (oczy względem poziomej osi)
        roll = _compute_roll(left_eye, right_eye)

        is_frontal = (
            abs(yaw) < self.frontal_threshold
            and abs(pitch) < self.frontal_threshold
            and abs(roll) < self.frontal_threshold
        )

        # Pewność = średnia widoczność kluczowych punktów
        key_indices = [
            KP.LEFT_EYE_INNER, KP.RIGHT_EYE_INNER,
            KP.NOSE_TIP,
            KP.LEFT_EAR_BASE_FRONT, KP.RIGHT_EAR_BASE_FRONT,
        ]
        confidence = float(np.mean([visibility[i] for i in key_indices]))

        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            is_frontal=is_frontal,
            confidence=confidence,
        )


def estimate_head_pose(
    keypoints_flat: np.ndarray,
    frontal_threshold: float = 30.0,
) -> HeadPose:
    """
    Funkcja pomocnicza do estymacji pozy głowy.

    Args:
        keypoints_flat: Array [x0, y0, v0, ...] (138 wartości)
        frontal_threshold: Próg dla frontalnej pozy (stopnie)

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
    Waliduje pozę głowy dla obliczania AU.

    Args:
        pose: HeadPose do walidacji
        max_angle: Maksymalnie dopuszczalny kąt
        min_confidence: Minimalna pewność

    Returns:
        True jeśli poza jest prawidłowa dla AU
    """
    return (
        pose.is_frontal
        and pose.confidence >= min_confidence
        and abs(pose.yaw) <= max_angle
        and abs(pose.pitch) <= max_angle
        and abs(pose.roll) <= max_angle
    )


# =============================================================================
# Funkcje pomocnicze (prywatne)
# =============================================================================

def _euclidean_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Odległość euklidesowa między dwoma punktami."""
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def _compute_yaw(
    nose: np.ndarray,
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    eye_width: float,
) -> float:
    """
    Oblicza kąt obrotu lewo/prawo (yaw).

    Konwencja: yaw > 0 = obrót w LEWO (nos przesuwa się w lewo),
               yaw < 0 = obrót w PRAWO.
    """
    if eye_width < 1e-6:
        return 0.0
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    # Ujemny sign: nos w lewo → offset ujemny → yaw dodatni
    nose_offset = eye_center_x - nose[0]
    return float(np.clip(math.degrees(math.atan2(nose_offset, eye_width)), -90, 90))


def _compute_pitch(
    nose: np.ndarray,
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    eye_width: float,
) -> float:
    """
    Oblicza kąt nachylenia góra/dół (pitch).

    Mierzy jak bardzo nos jest poniżej centrum oczu względem szerokości oczu.
    Konwencja: pitch > 0 = nos poniżej oczu (normalna poza psa).
    """
    if eye_width < 1e-6:
        return 0.0
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    vertical_offset = nose[1] - eye_center_y
    return float(np.clip(math.degrees(math.atan2(vertical_offset, eye_width)), -90, 90))


def _compute_roll(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """Oblicza kąt przechylenia na bok (roll)."""
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    if abs(dx) < 1e-6:
        return 0.0
    return float(np.clip(math.degrees(math.atan2(dy, dx)), -90, 90))
