"""
Ekstraktor Action Units (AU) oparty na deltach dla systemu DogFACS.

Oblicza Action Units jako **delta ratio** względem bazowej klatki neutralnej.
Podejście to jest niezależne od morfologii rasy — mierzymy ZMIANY, nie wartości absolutne.

Zasada: Delta_AU = (odległość_cel / odległość_neutral) - 1.0

Oficjalne kody DogFACS (21 AU łącznie):
  Górna twarz:  AU101, AU143, AU145
  Dolna twarz:  AU109, AU110, AU12, AU116, AU118, AU25, AU26, AU27
  Deskryptory:  AD19, AD33, AD35, AD37, AD137
  Uszy:         EAD101, EAD102, EAD103, EAD104, EAD105

Źródło naukowe: Waller et al. 2013 (DogFACS), Mota-Rojas et al. 2021,
                https://www.animalfacs.com/dogfacs
"""

import math
from dataclasses import dataclass

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS

# Oficjalne kody AU zgodnie z DogFACS (21 łącznie)
ACTION_UNIT_NAMES: list[str] = [
    # --- Górna twarz ---
    "AU101",   # Inner Brow Raiser — unoszenie wewnętrznej części brwi
    "AU143",   # Lid Tightener — napięcie powieki
    "AU145",   # Blink / Eye Closure — mrugnięcie / zamknięcie oka

    # --- Dolna twarz ---
    "AU109",   # Nose Wrinkler Left — zmarszczenie nosa (lewa strona)
    "AU110",   # Nose Wrinkler Right — zmarszczenie nosa (prawa strona)
    "AU12",    # Lip Corner Puller — ciągnięcie kącika ust (uśmiech)
    "AU116",   # Lower Lip Depressor — opuszczanie dolnej wargi
    "AU118",   # Lip Stretcher — rozciąganie wargi
    "AU25",    # Lips Part — rozwarcie warg
    "AU26",    # Jaw Drop — opuszczenie żuchwy
    "AU27",    # Mouth Stretch — szerokie otwarcie pyska

    # --- Action Descriptors ---
    "AD19",    # Tongue Show — pokazanie języka
    "AD33",    # Blow — dmuchnięcie / wydech przez usta
    "AD35",    # Lip Bite — przygryzienie wargi
    "AD37",    # Lip Wipe — oblizanie warg (nie nosa!)
    "AD137",   # Nose Lick — lizanie nosa (nie warg!)

    # --- Ear Action Descriptors ---
    "EAD101",  # Ears Forward — uszy do przodu
    "EAD102",  # Ears Adductor — uszy przyciągnięte do siebie
    "EAD103",  # Ears Flattener — uszy przyciśnięte do głowy
    "EAD104",  # Ears Rotator — obrót uszu
    "EAD105",  # Ears Apart — uszy rozstawione
]

NUM_ACTION_UNITS: int = len(ACTION_UNIT_NAMES)

# Próg aktywacji AU (15% zmiana względem stanu neutralnego)
DEFAULT_ACTIVATION_THRESHOLD: float = 1.15

# Próg dla AU aktywowanych przez ZMNIEJSZENIE (ratio < próg)
DECREASE_ACTIVATION_THRESHOLD: float = 0.85

# Minimalna pewność keypoints, by AU mógł być uznany za aktywny.
# Chroni przed fałszywymi AU z szumu (np. uszy, które model lokalizuje niepewnie).
MIN_AU_CONFIDENCE: float = 0.30

# Zakres przycięcia stosunku — zapobiega "wybuchowi" przy małym mianowniku
# (gdy w klatce neutralnej punkty się prawie pokrywają → ogromne ratio → wszystkie AU).
RATIO_CLAMP_MIN: float = 0.2
RATIO_CLAMP_MAX: float = 3.0


@dataclass
class DeltaActionUnit:
    """
    Pojedynczy Action Unit z pomiarem opartym na delcie.

    Attributes:
        name: Oficjalny kod DogFACS (np. "AU101", "EAD101")
        ratio: Stosunek cel/neutral (1.0 = brak zmiany, 1.3 = wzrost o 30%)
        delta: Ratio - 1.0 (0.0 = brak, 0.3 = wzrost o 30%)
        is_active: Binarna aktywacja na podstawie progu
        confidence: Pewność pomiaru z widoczności keypoints (0.0-1.0)
    """

    name: str
    ratio: float
    delta: float
    is_active: bool
    confidence: float

    def to_dict(self) -> dict:
        """Konwertuje do słownika dla serializacji JSON."""
        return {
            "name": self.name,
            "ratio": round(self.ratio, 4),
            "delta": round(self.delta, 4),
            "is_active": self.is_active,
            "confidence": round(self.confidence, 4),
        }


class DeltaActionUnitsExtractor:
    """
    Ekstraktor Action Units oparty na deltach względem klatki neutralnej.

    Oblicza AU jako stosunki odległości anatomicznych, co zapewnia
    niezależność od morfologii rasy psa.

    Przykład:
        >>> neutral_kp = np.zeros(NUM_KEYPOINTS * 3)
        >>> extractor = DeltaActionUnitsExtractor(neutral_kp)
        >>> target_kp = np.zeros(NUM_KEYPOINTS * 3)
        >>> delta_aus = extractor.extract(target_kp)
        >>> print(delta_aus["AU101"].ratio)
    """

    def __init__(
        self,
        neutral_keypoints: np.ndarray,
        activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    ) -> None:
        """
        Inicjalizuje ekstraktor z klatką neutralną jako bazą.

        Args:
            neutral_keypoints: Keypoints klatki neutralnej [x0,y0,v0,...] (138 wartości)
            activation_threshold: Próg stosunku dla aktywacji AU (domyślnie 1.15 = +15%)

        Raises:
            ValueError: Gdy liczba wartości keypoints jest nieprawidłowa
        """
        expected = NUM_KEYPOINTS * 3
        if len(neutral_keypoints) != expected:
            raise ValueError(
                f"Oczekiwano {expected} wartości keypoints (46×3), "
                f"otrzymano {len(neutral_keypoints)}"
            )

        self.neutral_kp = neutral_keypoints.reshape(NUM_KEYPOINTS, 3)
        self.activation_threshold = activation_threshold
        self.neutral_distances = self._compute_measurements(self.neutral_kp)

    def extract(self, target_keypoints: np.ndarray) -> dict[str, DeltaActionUnit]:
        """
        Ekstrahuje delta AU z klatki docelowej.

        Args:
            target_keypoints: Keypoints klatki docelowej [x0,y0,v0,...] (138 wartości)

        Returns:
            Słownik AU_name → DeltaActionUnit

        Raises:
            ValueError: Gdy liczba wartości keypoints jest nieprawidłowa
        """
        expected = NUM_KEYPOINTS * 3
        if len(target_keypoints) != expected:
            raise ValueError(
                f"Oczekiwano {expected} wartości keypoints, "
                f"otrzymano {len(target_keypoints)}"
            )

        target_kp = target_keypoints.reshape(NUM_KEYPOINTS, 3)
        target_distances = self._compute_measurements(target_kp)

        return {
            au_name: self._compute_delta_au(au_name, target_distances, target_kp)
            for au_name in ACTION_UNIT_NAMES
        }

    def _compute_delta_au(
        self,
        au_name: str,
        target_distances: dict[str, float],
        target_kp: np.ndarray,
    ) -> DeltaActionUnit:
        """
        Oblicza delta dla pojedynczego AU.

        Args:
            au_name: Kod AU (np. "AU101")
            target_distances: Pomiary klatki docelowej
            target_kp: Keypoints klatki docelowej (46, 3)

        Returns:
            DeltaActionUnit z obliczonymi wartościami
        """
        measurement_key = _AU_MEASUREMENT_MAP[au_name]
        neutral_val = self.neutral_distances[measurement_key]
        target_val = target_distances[measurement_key]

        if abs(neutral_val) > 1e-6:
            ratio = target_val / neutral_val
        else:
            ratio = 1.0
        # Przytnij stosunek, by szum/mały mianownik nie "wysadzał" AU do maksimum
        ratio = float(np.clip(ratio, RATIO_CLAMP_MIN, RATIO_CLAMP_MAX))
        delta = ratio - 1.0

        confidence = _compute_au_confidence(au_name, target_kp)
        is_active = _is_au_activated(au_name, ratio, self.activation_threshold)
        # AU uznajemy za aktywny tylko gdy keypointy są wystarczająco pewne —
        # inaczej "pokazywalibyśmy" ruchy, których nie ma (np. szum uszu).
        if confidence < MIN_AU_CONFIDENCE:
            is_active = False

        return DeltaActionUnit(
            name=au_name,
            ratio=ratio,
            delta=delta,
            is_active=is_active,
            confidence=confidence,
        )

    def _compute_measurements(self, keypoints: np.ndarray) -> dict[str, float]:
        """
        Oblicza wszystkie pomiary anatomiczne znormalizowane przez odległość oczu.

        Normalizacja przez odległość oczu zapewnia niezależność od rozmiaru głowy.

        Args:
            keypoints: Tablica keypoints (46, 3)

        Returns:
            Słownik klucz_pomiaru → wartość znormalizowana
        """
        coords = keypoints[:, :2]   # (46, 2)
        vis = keypoints[:, 2]       # (46,)

        eye_dist = _eye_distance(coords)

        return {
            # AU101: Inner Brow Raiser — odległość wewnętrznej brwi od wewnętrznego kąta oka
            "au101_brow_eye": (
                _dist(coords[KP.LEFT_BROW_INNER], coords[KP.LEFT_EYE_INNER])
                + _dist(coords[KP.RIGHT_BROW_INNER], coords[KP.RIGHT_EYE_INNER])
            ) / (2 * eye_dist),

            # AU143: Lid Tightener — pionowe otwarcie oka (mała wartość = zmrużenie)
            "au143_eye_opening": (
                _dist(coords[KP.LEFT_EYE_TOP], coords[KP.LEFT_EYE_BOTTOM])
                + _dist(coords[KP.RIGHT_EYE_TOP], coords[KP.RIGHT_EYE_BOTTOM])
            ) / (2 * eye_dist),

            # AU145: Blink — widoczność oka (mała = zamknięte)
            "au145_eye_visibility": float(
                (vis[KP.LEFT_EYE_TOP] + vis[KP.LEFT_EYE_BOTTOM]
                 + vis[KP.RIGHT_EYE_TOP] + vis[KP.RIGHT_EYE_BOTTOM]) / 4
            ),

            # AU109/110: Nose Wrinkler — unoszenie skrzydełek nosa ku brwiom
            "au109_nose_wrinkle_left": (
                _dist(coords[KP.NOSE_LEFT_WING], coords[KP.LEFT_EYE_BOTTOM])
                / eye_dist
            ),
            "au110_nose_wrinkle_right": (
                _dist(coords[KP.NOSE_RIGHT_WING], coords[KP.RIGHT_EYE_BOTTOM])
                / eye_dist
            ),

            # AU12: Lip Corner Puller — kąt kącika ust (uśmiech)
            "au12_lip_corner": _lip_corner_ratio(coords, eye_dist),

            # AU116: Lower Lip Depressor — opuszczanie dolnej wargi od nosa
            "au116_lower_lip": (
                _dist(coords[KP.LOWER_LIP_CENTER], coords[KP.NOSE_TIP])
                / eye_dist
            ),

            # AU118: Lip Stretcher — szerokość ust znormalizowana
            "au118_lip_width": (
                _dist(coords[KP.MOUTH_LEFT_CORNER], coords[KP.MOUTH_RIGHT_CORNER])
                / eye_dist
            ),

            # AU25: Lips Part — pionowa szczelina między wargami
            "au25_lips_gap": (
                _dist(coords[KP.UPPER_LIP_CENTER], coords[KP.LOWER_LIP_CENTER])
                / eye_dist
            ),

            # AU26: Jaw Drop — odległość nos-podbródek
            "au26_jaw_drop": (
                _dist(coords[KP.NOSE_TIP], coords[KP.CHIN])
                / eye_dist
            ),

            # AU27: Mouth Stretch — pełne otwarcie pyska (kąciki do podbródka)
            "au27_mouth_stretch": (
                _dist(coords[KP.UPPER_LIP_CENTER], coords[KP.CHIN])
                / eye_dist
            ),

            # AD19: Tongue Show — proxy: otwarcie ust + szczelina pionowa
            "ad19_tongue": _tongue_proxy(coords, eye_dist),

            # AD33: Blow — szczelina mała, wargi zaokrąglone (podobne do AU25)
            "ad33_blow": (
                _dist(coords[KP.UPPER_LIP_CENTER], coords[KP.LOWER_LIP_CENTER])
                / eye_dist
            ),

            # AD35: Lip Bite — zbliżenie górnej wargi do dolnej (mała szczelina)
            "ad35_lip_bite": (
                _dist(coords[KP.UPPER_LIP_CENTER], coords[KP.LOWER_LIP_CENTER])
                / eye_dist
            ),

            # AD37: Lip Wipe — odległość języka/wargi od kącika ust
            "ad37_lip_wipe": (
                _dist(coords[KP.LOWER_LIP_CENTER], coords[KP.MOUTH_LEFT_CORNER])
                + _dist(coords[KP.LOWER_LIP_CENTER], coords[KP.MOUTH_RIGHT_CORNER])
            ) / (2 * eye_dist),

            # AD137: Nose Lick — odległość wargi od nosa (mała = język przy nosie)
            "ad137_nose_lick": (
                _dist(coords[KP.UPPER_LIP_CENTER], coords[KP.NOSE_TIP])
                / eye_dist
            ),

            # EAD101: Ears Forward — uszy przesunięte do przodu (małe x względem oka)
            "ead101_ears_forward": _ear_forward_ratio(coords, eye_dist),

            # EAD102: Ears Adductor — uszy zbliżone do siebie
            "ead102_ears_adduct": (
                _dist(coords[KP.LEFT_EAR_BASE_FRONT], coords[KP.RIGHT_EAR_BASE_FRONT])
                / eye_dist
            ),

            # EAD103: Ears Flattener — czubki uszu blisko głowy (nisko)
            "ead103_ears_flat": _ear_flatten_ratio(coords, eye_dist),

            # EAD104: Ears Rotator — asymetria uszu (kąt obrotu)
            "ead104_ears_rotate": _ear_rotation_asymmetry(coords),

            # EAD105: Ears Apart — uszy daleko od siebie
            "ead105_ears_apart": (
                _dist(coords[KP.LEFT_EAR_TIP], coords[KP.RIGHT_EAR_TIP])
                / eye_dist
            ),
        }


# =============================================================================
# Mapowanie AU → klucz pomiaru (prywatne)
# =============================================================================

_AU_MEASUREMENT_MAP: dict[str, str] = {
    "AU101":  "au101_brow_eye",
    "AU143":  "au143_eye_opening",
    "AU145":  "au145_eye_visibility",
    "AU109":  "au109_nose_wrinkle_left",
    "AU110":  "au110_nose_wrinkle_right",
    "AU12":   "au12_lip_corner",
    "AU116":  "au116_lower_lip",
    "AU118":  "au118_lip_width",
    "AU25":   "au25_lips_gap",
    "AU26":   "au26_jaw_drop",
    "AU27":   "au27_mouth_stretch",
    "AD19":   "ad19_tongue",
    "AD33":   "ad33_blow",
    "AD35":   "ad35_lip_bite",
    "AD37":   "ad37_lip_wipe",
    "AD137":  "ad137_nose_lick",
    "EAD101": "ead101_ears_forward",
    "EAD102": "ead102_ears_adduct",
    "EAD103": "ead103_ears_flat",
    "EAD104": "ead104_ears_rotate",
    "EAD105": "ead105_ears_apart",
}

# AU aktywowane przez ZMNIEJSZENIE (np. zmrużenie = mniejsze otwarcie oka)
_DECREASE_ACTIVATED_AUS: frozenset[str] = frozenset({
    "AU143",   # Lid Tightener: mniejsze otwarcie oka
    "AU145",   # Blink: mniejsza widoczność
    "AD35",    # Lip Bite: mniejsza szczelina
    "AD137",   # Nose Lick: wargi bliżej nosa
    "EAD102",  # Ears Adductor: uszy bliżej siebie
    "EAD103",  # Ears Flattener: uszy nisko
})

# AU aktywowane przez znaczną ZMIANĘ w dowolnym kierunku (bidirektywne)
_BIDIRECTIONAL_AUS: frozenset[str] = frozenset({
    "EAD104",  # Ears Rotator: asymetria może iść w obu kierunkach
})

# Grupy keypoints dla obliczania pewności każdego AU
_AU_KEYPOINT_GROUPS: dict[str, list[int]] = {
    "AU101":  [KP.LEFT_BROW_INNER, KP.RIGHT_BROW_INNER,
               KP.LEFT_EYE_INNER, KP.RIGHT_EYE_INNER],
    "AU143":  [KP.LEFT_EYE_TOP, KP.LEFT_EYE_BOTTOM,
               KP.RIGHT_EYE_TOP, KP.RIGHT_EYE_BOTTOM],
    "AU145":  [KP.LEFT_EYE_TOP, KP.LEFT_EYE_BOTTOM,
               KP.RIGHT_EYE_TOP, KP.RIGHT_EYE_BOTTOM],
    "AU109":  [KP.NOSE_LEFT_WING, KP.LEFT_EYE_BOTTOM],
    "AU110":  [KP.NOSE_RIGHT_WING, KP.RIGHT_EYE_BOTTOM],
    "AU12":   [KP.MOUTH_LEFT_CORNER, KP.MOUTH_RIGHT_CORNER, KP.UPPER_LIP_CENTER],
    "AU116":  [KP.LOWER_LIP_CENTER, KP.NOSE_TIP],
    "AU118":  [KP.MOUTH_LEFT_CORNER, KP.MOUTH_RIGHT_CORNER],
    "AU25":   [KP.UPPER_LIP_CENTER, KP.LOWER_LIP_CENTER],
    "AU26":   [KP.NOSE_TIP, KP.CHIN],
    "AU27":   [KP.UPPER_LIP_CENTER, KP.LOWER_LIP_CENTER, KP.CHIN],
    "AD19":   [KP.UPPER_LIP_CENTER, KP.LOWER_LIP_CENTER, KP.NOSE_TIP],
    "AD33":   [KP.UPPER_LIP_CENTER, KP.LOWER_LIP_CENTER],
    "AD35":   [KP.UPPER_LIP_CENTER, KP.LOWER_LIP_CENTER],
    "AD37":   [KP.LOWER_LIP_CENTER, KP.MOUTH_LEFT_CORNER, KP.MOUTH_RIGHT_CORNER],
    "AD137":  [KP.UPPER_LIP_CENTER, KP.NOSE_TIP],
    "EAD101": [KP.LEFT_EAR_BASE_FRONT, KP.RIGHT_EAR_BASE_FRONT,
               KP.LEFT_EAR_TIP, KP.RIGHT_EAR_TIP],
    "EAD102": [KP.LEFT_EAR_BASE_FRONT, KP.RIGHT_EAR_BASE_FRONT],
    "EAD103": [KP.LEFT_EAR_TIP, KP.RIGHT_EAR_TIP,
               KP.LEFT_EAR_BASE_FRONT, KP.RIGHT_EAR_BASE_FRONT],
    "EAD104": [KP.LEFT_EAR_TIP, KP.RIGHT_EAR_TIP],
    "EAD105": [KP.LEFT_EAR_TIP, KP.RIGHT_EAR_TIP],
}


# =============================================================================
# Funkcje pomocnicze (prywatne)
# =============================================================================

def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Odległość euklidesowa między dwoma punktami."""
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def _eye_distance(coords: np.ndarray) -> float:
    """
    Odległość między centrami oczu używana do normalizacji.

    Centrum oka = średnia wewnętrznego i zewnętrznego kąta.
    """
    left_center = (coords[KP.LEFT_EYE_INNER] + coords[KP.LEFT_EYE_OUTER]) / 2
    right_center = (coords[KP.RIGHT_EYE_INNER] + coords[KP.RIGHT_EYE_OUTER]) / 2
    dist = _dist(left_center, right_center)
    return dist if dist > 1e-6 else 1.0  # Zabezpieczenie przed dzieleniem przez zero


def _lip_corner_ratio(coords: np.ndarray, eye_dist: float) -> float:
    """
    Oblicza ratio kącika ust dla AU12 (Lip Corner Puller).

    Wyższy wynik = bardziej uniesione kąciki (uśmiech).
    """
    upper_to_corner_left = _dist(coords[KP.UPPER_LIP_CENTER], coords[KP.MOUTH_LEFT_CORNER])
    upper_to_corner_right = _dist(coords[KP.UPPER_LIP_CENTER], coords[KP.MOUTH_RIGHT_CORNER])
    avg_corner_dist = (upper_to_corner_left + upper_to_corner_right) / 2
    return avg_corner_dist / eye_dist if eye_dist > 1e-6 else 0.0


def _tongue_proxy(coords: np.ndarray, eye_dist: float) -> float:
    """
    Proxy dla AD19 (Tongue Show).

    Język widoczny gdy: usta otwarte + dolna warga nisko + szczelina pionowa duża.
    """
    vertical_gap = _dist(coords[KP.UPPER_LIP_CENTER], coords[KP.LOWER_LIP_CENTER])
    jaw_drop = _dist(coords[KP.NOSE_TIP], coords[KP.CHIN])
    return (vertical_gap / eye_dist) * min(1.0, jaw_drop / eye_dist)


def _ear_forward_ratio(coords: np.ndarray, eye_dist: float) -> float:
    """
    Oblicza ratio przesunięcia uszu do przodu dla EAD101.

    Uszy do przodu = czubki uszu blisko czubka nosa (małe x w profilu).
    """
    nose_x = coords[KP.NOSE_TIP][0]
    left_tip_dx = abs(coords[KP.LEFT_EAR_TIP][0] - nose_x)
    right_tip_dx = abs(coords[KP.RIGHT_EAR_TIP][0] - nose_x)
    return (left_tip_dx + right_tip_dx) / (2 * eye_dist)


def _ear_flatten_ratio(coords: np.ndarray, eye_dist: float) -> float:
    """
    Oblicza ratio przyciśnięcia uszu do głowy dla EAD103.

    Uszy płaskie = czubki uszu nisko (wysokie y) względem podstawy.
    """
    left_drop = coords[KP.LEFT_EAR_TIP][1] - coords[KP.LEFT_EAR_BASE_FRONT][1]
    right_drop = coords[KP.RIGHT_EAR_TIP][1] - coords[KP.RIGHT_EAR_BASE_FRONT][1]
    return (left_drop + right_drop) / (2 * eye_dist)


def _ear_rotation_asymmetry(coords: np.ndarray) -> float:
    """
    Oblicza asymetrię rotacji uszu dla EAD104.

    Zwraca wartość bezwzględną różnicy kątów uszu (0 = symetryczne).
    """
    left_angle = math.atan2(
        coords[KP.LEFT_EAR_TIP][1] - coords[KP.LEFT_EAR_BASE_FRONT][1],
        coords[KP.LEFT_EAR_TIP][0] - coords[KP.LEFT_EAR_BASE_FRONT][0],
    )
    right_angle = math.atan2(
        coords[KP.RIGHT_EAR_TIP][1] - coords[KP.RIGHT_EAR_BASE_FRONT][1],
        coords[KP.RIGHT_EAR_TIP][0] - coords[KP.RIGHT_EAR_BASE_FRONT][0],
    )
    return abs(left_angle - right_angle) / math.pi


def _is_au_activated(
    au_name: str,
    ratio: float,
    threshold: float,
) -> bool:
    """
    Określa czy AU jest aktywowany na podstawie stosunku i progu.

    Args:
        au_name: Kod AU
        ratio: Stosunek cel/neutral
        threshold: Próg aktywacji (np. 1.15 = +15%)

    Returns:
        True jeśli AU jest aktywowany
    """
    if au_name in _BIDIRECTIONAL_AUS:
        return abs(ratio - 1.0) > (threshold - 1.0)
    if au_name in _DECREASE_ACTIVATED_AUS:
        return ratio < DECREASE_ACTIVATION_THRESHOLD
    return ratio > threshold


def _compute_au_confidence(au_name: str, keypoints: np.ndarray) -> float:
    """
    Oblicza pewność AU na podstawie widoczności keypoints.

    Args:
        au_name: Kod AU
        keypoints: Tablica keypoints (46, 3)

    Returns:
        Pewność (0.0-1.0)
    """
    relevant_indices = _AU_KEYPOINT_GROUPS.get(au_name, list(range(NUM_KEYPOINTS)))
    visibilities = [keypoints[idx, 2] for idx in relevant_indices]
    return float(np.mean(visibilities)) if visibilities else 0.0


# =============================================================================
# Publiczna funkcja pomocnicza
# =============================================================================

def extract_delta_action_units(
    neutral_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> dict[str, DeltaActionUnit]:
    """
    Ekstrahuje delta AU z pary klatek (neutral + cel).

    Args:
        neutral_keypoints: Keypoints klatki neutralnej (138 wartości)
        target_keypoints: Keypoints klatki docelowej (138 wartości)
        activation_threshold: Próg stosunku dla aktywacji (domyślnie 1.15)

    Returns:
        Słownik AU_name → DeltaActionUnit
    """
    extractor = DeltaActionUnitsExtractor(neutral_keypoints, activation_threshold)
    return extractor.extract(target_keypoints)
