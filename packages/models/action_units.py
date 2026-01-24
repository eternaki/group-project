"""
Action Units (AU) computed from keypoints.

Вычисляет DogFACS Action Units на основе геометрии 20 keypoints.
Это промежуточный слой между keypoints и эмоциями.

DogFACS AU для собак (основные):
- EAD101: Inner Brow Raiser
- EAD102: Outer Brow Raiser / Ears Adductor
- EAD103: Ears Flattener
- EAD104: Ears Rotator
- AU101: Inner Brow Raiser
- AU102: Outer Brow Raiser
- AU143: Eyes Closure
- AU145: Blink
- AU201: Ears Forward
- AU202: Ears Back
- AU25: Lips Part
- AU26: Jaw Drop
- AU27: Lip Corner Puller (smile)

Źródło: https://www.animalfacs.com/dogfacs
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import math

from packages.data.schemas import NUM_KEYPOINTS, KEYPOINT_NAMES


# Indeksy keypoints (dla czytelności)
KP_LEFT_EYE = 0
KP_RIGHT_EYE = 1
KP_NOSE = 2
KP_LEFT_EAR_BASE = 3
KP_RIGHT_EAR_BASE = 4
KP_LEFT_EAR_TIP = 5
KP_RIGHT_EAR_TIP = 6
KP_LEFT_MOUTH = 7
KP_RIGHT_MOUTH = 8
KP_UPPER_LIP = 9
KP_LOWER_LIP = 10
KP_CHIN = 11
KP_LEFT_CHEEK = 12
KP_RIGHT_CHEEK = 13
KP_FOREHEAD = 14
KP_LEFT_BROW = 15
KP_RIGHT_BROW = 16
KP_MUZZLE_TOP = 17
KP_MUZZLE_LEFT = 18
KP_MUZZLE_RIGHT = 19


# Nazwy Action Units
ACTION_UNIT_NAMES = [
    "AU_brow_raise",        # 0 - Podniesienie brwi (AU101/102)
    "AU_ear_forward",       # 1 - Uszy do przodu (EAD102/AU201)
    "AU_ear_back",          # 2 - Uszy do tyłu (EAD103/AU202)
    "AU_ear_asymmetry",     # 3 - Asymetria uszu
    "AU_eye_opening",       # 4 - Otwarcie oczu (przeciwne do AU143/145)
    "AU_mouth_open",        # 5 - Otwarcie pyska (AU25/26)
    "AU_lip_corner_pull",   # 6 - Pociągnięcie kącików ust (AU27 - uśmiech)
    "AU_jaw_drop",          # 7 - Opadnięcie szczęki (AU26)
    "AU_nose_wrinkle",      # 8 - Zmarszczenie nosa
    "AU_muzzle_width",      # 9 - Szerokość pyska
    "AU_face_elongation",   # 10 - Wydłużenie twarzy
    "AU_eye_distance",      # 11 - Odległość między oczami (znormalizowana)
]

NUM_ACTION_UNITS = len(ACTION_UNIT_NAMES)


@dataclass
class ActionUnitsPrediction:
    """Wynik ekstrakcji Action Units z keypoints."""

    values: dict[str, float]  # AU_name -> wartość (0.0 - 1.0)
    features: np.ndarray      # Wektor cech dla modelu emocji

    def to_dict(self) -> dict:
        return {"action_units": self.values}


class ActionUnitsExtractor:
    """
    Ekstrahuje Action Units z keypoints.

    Używa geometrycznych relacji między keypoints do oszacowania
    aktywacji poszczególnych Action Units zgodnie z DogFACS.

    Example:
        extractor = ActionUnitsExtractor()
        keypoints = np.array([x0, y0, v0, x1, y1, v1, ...])  # 60 wartości
        au_prediction = extractor.extract(keypoints)
        print(au_prediction.values)
    """

    def __init__(self) -> None:
        """Inicjalizuje ekstraktor AU."""
        pass

    def extract(self, keypoints_flat: np.ndarray) -> ActionUnitsPrediction:
        """
        Ekstrahuje Action Units z keypoints.

        Args:
            keypoints_flat: Array [x0, y0, v0, x1, y1, v1, ...] (60 wartości)

        Returns:
            ActionUnitsPrediction z wartościami AU
        """
        # Reshape do (20, 3)
        kp = keypoints_flat.reshape(NUM_KEYPOINTS, 3)

        # Wyodrębnij współrzędne i visibility
        coords = kp[:, :2]  # (20, 2)
        visibility = kp[:, 2]  # (20,)

        # Oblicz odległość referencyjną (między oczami) do normalizacji
        eye_distance = self._distance(coords[KP_LEFT_EYE], coords[KP_RIGHT_EYE])
        if eye_distance < 1e-6:
            eye_distance = 1.0  # Fallback

        # Oblicz wszystkie AU
        au_values = {}

        # AU_brow_raise: Odległość brew od oczu (znormalizowana)
        au_values["AU_brow_raise"] = self._compute_brow_raise(coords, eye_distance)

        # AU_ear_forward: Kąt uszu względem pionu
        au_values["AU_ear_forward"] = self._compute_ear_forward(coords)

        # AU_ear_back: Odwrotność ear_forward
        au_values["AU_ear_back"] = 1.0 - au_values["AU_ear_forward"]

        # AU_ear_asymmetry: Różnica kątów uszu
        au_values["AU_ear_asymmetry"] = self._compute_ear_asymmetry(coords)

        # AU_eye_opening: Bazowane na visibility oczu
        au_values["AU_eye_opening"] = self._compute_eye_opening(visibility)

        # AU_mouth_open: Odległość między wargami
        au_values["AU_mouth_open"] = self._compute_mouth_open(coords, eye_distance)

        # AU_lip_corner_pull: Kąt kącików ust (uśmiech)
        au_values["AU_lip_corner_pull"] = self._compute_lip_corner_pull(coords)

        # AU_jaw_drop: Odległość podbródka od nosa
        au_values["AU_jaw_drop"] = self._compute_jaw_drop(coords, eye_distance)

        # AU_nose_wrinkle: Aproksymacja przez odległość nosa od górnej wargi
        au_values["AU_nose_wrinkle"] = self._compute_nose_wrinkle(coords, eye_distance)

        # AU_muzzle_width: Szerokość pyska
        au_values["AU_muzzle_width"] = self._compute_muzzle_width(coords, eye_distance)

        # AU_face_elongation: Proporcje twarzy
        au_values["AU_face_elongation"] = self._compute_face_elongation(coords, eye_distance)

        # AU_eye_distance: Znormalizowana odległość oczu
        au_values["AU_eye_distance"] = min(eye_distance / 100.0, 1.0)

        # Konwertuj do feature vector
        features = np.array([au_values[name] for name in ACTION_UNIT_NAMES], dtype=np.float32)

        return ActionUnitsPrediction(
            values=au_values,
            features=features,
        )

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Oblicza odległość euklidesową."""
        return float(np.sqrt(np.sum((p1 - p2) ** 2)))

    def _angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Oblicza kąt linii względem poziomu (w radianach)."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.atan2(dy, dx)

    def _compute_brow_raise(self, coords: np.ndarray, ref_dist: float) -> float:
        """Oblicza podniesienie brwi."""
        left_brow_eye_dist = self._distance(coords[KP_LEFT_BROW], coords[KP_LEFT_EYE])
        right_brow_eye_dist = self._distance(coords[KP_RIGHT_BROW], coords[KP_RIGHT_EYE])
        avg_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2

        # Normalizuj względem odległości oczu
        normalized = avg_dist / ref_dist

        # Typowy zakres: 0.2 - 0.5
        return float(np.clip((normalized - 0.2) / 0.3, 0.0, 1.0))

    def _compute_ear_forward(self, coords: np.ndarray) -> float:
        """Oblicza czy uszy są do przodu."""
        # Kąt między bazą a czubkiem ucha względem pionu
        left_angle = self._angle(coords[KP_LEFT_EAR_BASE], coords[KP_LEFT_EAR_TIP])
        right_angle = self._angle(coords[KP_RIGHT_EAR_BASE], coords[KP_RIGHT_EAR_TIP])

        # Uszy do przodu: kąty bliskie -π/2 (w górę i do środka)
        # Uszy do tyłu: kąty bliskie π/2 lub rozproszone
        avg_angle = (abs(left_angle) + abs(right_angle)) / 2

        # Normalizuj: mały kąt = uszy do przodu = wysokie AU
        return float(np.clip(1.0 - avg_angle / math.pi, 0.0, 1.0))

    def _compute_ear_asymmetry(self, coords: np.ndarray) -> float:
        """Oblicza asymetrię uszu."""
        left_angle = self._angle(coords[KP_LEFT_EAR_BASE], coords[KP_LEFT_EAR_TIP])
        right_angle = self._angle(coords[KP_RIGHT_EAR_BASE], coords[KP_RIGHT_EAR_TIP])

        # Różnica kątów znormalizowana
        diff = abs(left_angle - right_angle) / math.pi
        return float(np.clip(diff, 0.0, 1.0))

    def _compute_eye_opening(self, visibility: np.ndarray) -> float:
        """Oblicza otwarcie oczu na podstawie visibility."""
        left_eye_vis = visibility[KP_LEFT_EYE]
        right_eye_vis = visibility[KP_RIGHT_EYE]

        # Wysoka visibility = otwarte oczy
        return float((left_eye_vis + right_eye_vis) / 2)

    def _compute_mouth_open(self, coords: np.ndarray, ref_dist: float) -> float:
        """Oblicza otwarcie pyska."""
        lip_dist = self._distance(coords[KP_UPPER_LIP], coords[KP_LOWER_LIP])
        normalized = lip_dist / ref_dist

        # Typowy zakres: 0.0 - 0.3
        return float(np.clip(normalized / 0.3, 0.0, 1.0))

    def _compute_lip_corner_pull(self, coords: np.ndarray) -> float:
        """Oblicza pociągnięcie kącików ust (uśmiech)."""
        # Kąt między kącikami ust a górną wargą
        left_angle = self._angle(coords[KP_UPPER_LIP], coords[KP_LEFT_MOUTH])
        right_angle = self._angle(coords[KP_UPPER_LIP], coords[KP_RIGHT_MOUTH])

        # Kąciki w górę = uśmiech
        # left_angle powinien być dodatni, right_angle ujemny przy uśmiechu
        smile_indicator = (left_angle - right_angle) / math.pi

        return float(np.clip((smile_indicator + 1) / 2, 0.0, 1.0))

    def _compute_jaw_drop(self, coords: np.ndarray, ref_dist: float) -> float:
        """Oblicza opadnięcie szczęki."""
        nose_chin_dist = self._distance(coords[KP_NOSE], coords[KP_CHIN])
        normalized = nose_chin_dist / ref_dist

        # Typowy zakres: 0.5 - 1.5
        return float(np.clip((normalized - 0.5) / 1.0, 0.0, 1.0))

    def _compute_nose_wrinkle(self, coords: np.ndarray, ref_dist: float) -> float:
        """Oblicza zmarszczenie nosa."""
        nose_lip_dist = self._distance(coords[KP_NOSE], coords[KP_UPPER_LIP])
        normalized = nose_lip_dist / ref_dist

        # Mała odległość = zmarszczony nos
        return float(np.clip(1.0 - normalized / 0.5, 0.0, 1.0))

    def _compute_muzzle_width(self, coords: np.ndarray, ref_dist: float) -> float:
        """Oblicza szerokość pyska."""
        muzzle_width = self._distance(coords[KP_MUZZLE_LEFT], coords[KP_MUZZLE_RIGHT])
        normalized = muzzle_width / ref_dist

        # Typowy zakres: 0.3 - 0.8
        return float(np.clip((normalized - 0.3) / 0.5, 0.0, 1.0))

    def _compute_face_elongation(self, coords: np.ndarray, ref_dist: float) -> float:
        """Oblicza wydłużenie twarzy."""
        # Pionowa odległość: czoło - podbródek
        face_height = self._distance(coords[KP_FOREHEAD], coords[KP_CHIN])
        normalized = face_height / ref_dist

        # Typowy zakres: 1.0 - 2.5
        return float(np.clip((normalized - 1.0) / 1.5, 0.0, 1.0))


def extract_action_units(keypoints_flat: np.ndarray) -> np.ndarray:
    """
    Funkcja pomocnicza do ekstrakcji AU jako feature vector.

    Args:
        keypoints_flat: Array [x0, y0, v0, ...] (60 wartości)

    Returns:
        Array z wartościami AU (12 wartości)
    """
    extractor = ActionUnitsExtractor()
    prediction = extractor.extract(keypoints_flat)
    return prediction.features
