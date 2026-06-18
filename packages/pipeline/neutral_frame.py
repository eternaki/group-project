"""
Detekcja klatki neutralnej dla obliczeń delta AU.

Wykrywa "neutralną klatkę bazową" w sekwencji wideo, gdzie wyraz twarzy
psa jest najbardziej rozluźniony i stabilny (minimalne ruchy).

Neutralna klatka służy jako punkt odniesienia do obliczania delta AU.

Znane ograniczenia (data-free):
- Stabilność ≠ neutralność: detektor mierzy brak ruchu, nie rozluźnienie wyrazu.
  Łagodzone heurystyką "typowej konfiguracji" (_select_most_typical), ale to proxy 2D.
- Medianowy baseline zakłada ~stałą skalę twarzy w obrębie okna (frontalny, krótki
  odcinek czasu). Per-AU progi i kalibracja wyrazu — poza zakresem (wariant C).
"""

import math
from typing import Optional

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS
from packages.models.head_pose import HeadPose, estimate_head_pose

# Skala wariancji w jednostkach znormalizowanych (po podziale przez eye-distance).
# Dobrana tak, by ~0.05 (5% odległości oczu) jitteru dawało wyraźnie niższy score.
VARIANCE_SCALE: float = 1000.0

# Próg widoczności keypointa, by wszedł do medianowej bazy.
BASELINE_VIS_THRESHOLD: float = 0.3

# Frakcja najstabilniejszych kandydatów branych pod uwagę przy wyborze "typowej" klatki.
TOP_STABLE_FRACTION: float = 0.34


class NeutralFrameDetector:
    """
    Wykrywa neutralną klatkę bazową z sekwencji wideo.

    Neutralna klatka powinna mieć:
    1. Minimalną wariancję keypoints (stabilna, bez ruchu)
    2. Frontalną pozę głowy
    3. Wysoką pewność keypoints
    4. Krytyczne keypoints widoczne

    Przykład:
        >>> detector = NeutralFrameDetector()
        >>> frames = [frame1, frame2, ...]
        >>> keypoints_list = [kp1, kp2, ...]   # 138 wartości każdy
        >>> neutral_idx = detector.detect_auto(frames, keypoints_list)
        >>> print(f"Neutralna klatka: {neutral_idx}")
    """

    def __init__(
        self,
        window_size: int = 10,
        min_keypoint_conf: float = 0.5,
        max_yaw: float = 35.0,
        max_pitch: float = 40.0,
        max_roll: float = 20.0,
    ) -> None:
        """
        Inicjalizuje detektor.

        Args:
            window_size: Rozmiar okna do obliczania stabilności
            min_keypoint_conf: Minimalna pewność keypoints (domyślnie 0.5)
            max_yaw: Maks. kąt yaw dla frontalnej pozy (stopnie)
            max_pitch: Maks. kąt pitch dla frontalnej pozy (stopnie)
            max_roll: Maks. kąt roll dla frontalnej pozy (stopnie)
        """
        self.window_size = window_size
        self.min_keypoint_conf = min_keypoint_conf
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.max_roll = max_roll

    def detect_auto(
        self,
        frames: list[np.ndarray],
        keypoints_list: list[Optional[np.ndarray]],
        head_poses: Optional[list[Optional[HeadPose]]] = None,
        debug: bool = False,
        frame_indices: Optional[list[int]] = None,
    ) -> int:
        """
        Automatycznie wykrywa neutralną klatkę z sekwencji wideo.

        Args:
            frames: Lista klatek wideo
            keypoints_list: Lista tablic keypoints (138 wartości każda)
            head_poses: Opcjonalna lista HeadPose (obliczana jeśli None)
            debug: Włącz logowanie debugowania
            frame_indices: Oryginalne indeksy klatek w wideo (opcjonalne)

        Returns:
            Indeks neutralnej klatki

        Raises:
            ValueError: Gdy sekwencja jest pusta lub brak kandydatów
        """
        if not frames:
            raise ValueError("Sekwencja klatek jest pusta")

        if len(frames) == 1:
            return 0

        if head_poses is None:
            head_poses = [
                estimate_head_pose(kp) if kp is not None else None
                for kp in keypoints_list
            ]

        candidates = self._find_candidates(keypoints_list, head_poses, debug)

        if not candidates:
            candidates = self._find_relaxed_candidates(keypoints_list, head_poses)

        if not candidates:
            # Ostatnia deska ratunku: jakakolwiek klatka z keypoints
            for i, kp in enumerate(keypoints_list):
                if kp is not None and np.mean(kp.reshape(NUM_KEYPOINTS, 3)[:, 2]) > 0.3:
                    return i

            raise ValueError(
                "Brak kandydatów na neutralną klatkę. "
                "Wideo może mieć za mało wykrytych keypoints."
            )

        score_map = {
            idx: self._compute_stability_score(keypoints_list, idx, frame_indices)
            for idx in candidates
        }
        return _select_most_typical(candidates, score_map, keypoints_list)

    def detect_manual(self, frame_idx: int) -> int:
        """
        Manualne wskazanie neutralnej klatki.

        Args:
            frame_idx: Indeks klatki wybrany przez użytkownika

        Returns:
            Ten sam frame_idx (dla spójności z detect_auto)
        """
        return frame_idx

    def _find_candidates(
        self,
        keypoints_list: list[Optional[np.ndarray]],
        head_poses: list[Optional[HeadPose]],
        debug: bool,
    ) -> list[int]:
        """Filtruje kandydatów według ścisłych kryteriów."""
        return [
            i
            for i in range(len(keypoints_list))
            if self._is_valid_candidate(keypoints_list[i], head_poses[i], i, debug)
        ]

    def _is_valid_candidate(
        self,
        keypoints: Optional[np.ndarray],
        head_pose: Optional[HeadPose],
        frame_idx: int = -1,
        debug: bool = False,
    ) -> bool:
        """
        Sprawdza czy klatka jest kandydatem na neutralną (ścisłe kryteria).

        Args:
            keypoints: Tablica keypoints (138 wartości) lub None
            head_pose: Estymacja pozy głowy lub None
            frame_idx: Indeks klatki do logowania
            debug: Włącz logowanie

        Returns:
            True jeśli klatka jest prawidłowym kandydatem
        """
        if keypoints is None or head_pose is None:
            return False

        kp = keypoints.reshape(NUM_KEYPOINTS, 3)

        # Sprawdź frontalność
        if not _is_frontal_pose(head_pose, self.max_yaw, self.max_pitch, self.max_roll):
            if debug:
                print(f"  Klatka {frame_idx}: odrzucona — nie frontalna "
                      f"(yaw={head_pose.yaw:.1f}, pitch={head_pose.pitch:.1f})")
            return False

        # Sprawdź ogólną widoczność
        mean_visibility = float(np.mean(kp[:, 2]))
        if mean_visibility < self.min_keypoint_conf:
            if debug:
                print(f"  Klatka {frame_idx}: odrzucona — niska widoczność "
                      f"({mean_visibility:.2f} < {self.min_keypoint_conf})")
            return False

        # Sprawdź krytyczne keypoints
        if not _critical_keypoints_visible(kp, threshold=0.3):
            if debug:
                print(f"  Klatka {frame_idx}: odrzucona — brak krytycznych keypoints")
            return False

        return True

    def _find_relaxed_candidates(
        self,
        keypoints_list: list[Optional[np.ndarray]],
        head_poses: list[Optional[HeadPose]],
    ) -> list[int]:
        """
        Szuka kandydatów z poluzowanymi kryteriami (fallback).

        Poluzowane kryteria:
        - Kąt do 60° (zamiast max_yaw/max_pitch)
        - Pewność 0.4 (zamiast min_keypoint_conf)
        - Co najmniej 3 krytyczne keypoints widoczne

        Args:
            keypoints_list: Lista keypoints (może zawierać None)
            head_poses: Lista head poses (może zawierać None)

        Returns:
            Lista indeksów kandydatów
        """
        relaxed_angle = 60.0
        relaxed_conf = 0.4

        candidates = []
        for i in range(len(keypoints_list)):
            if keypoints_list[i] is None or head_poses[i] is None:
                continue

            kp = keypoints_list[i].reshape(NUM_KEYPOINTS, 3)
            pose = head_poses[i]

            if abs(pose.yaw) > relaxed_angle or abs(pose.pitch) > relaxed_angle:
                continue

            if float(np.mean(kp[:, 2])) < relaxed_conf:
                continue

            if _count_visible_critical_kps(kp, threshold=0.4) >= 3:
                candidates.append(i)

        return candidates

    def _compute_stability_score(
        self,
        keypoints_list: list[Optional[np.ndarray]],
        center_idx: int,
        frame_indices: Optional[list[int]] = None,
    ) -> float:
        """
        Oblicza wynik stabilności klatki na znormalizowanych współrzędnych.

        Stabilność = 1 / (1 + wariancja * VARIANCE_SCALE). Wyższy = bardziej neutralna.
        Gdy frame_indices podane, okno obejmuje klatki o oryginalnym indeksie w zasięgu
        ±window_size//2 od center (poprawne sąsiedztwo czasowe mimo luk detekcji).

        Args:
            keypoints_list: Lista wszystkich keypoints (może zawierać None)
            center_idx: Indeks klatki do oceny
            frame_indices: Oryginalne indeksy klatek w wideo (opcjonalne)

        Returns:
            Wynik stabilności (wyższy = bardziej stabilna)
        """
        half = self.window_size // 2
        if frame_indices is not None:
            center_frame = frame_indices[center_idx]
            members = [
                keypoints_list[j]
                for j in range(len(keypoints_list))
                if abs(frame_indices[j] - center_frame) <= half
            ]
        else:
            start = max(0, center_idx - half)
            end = min(len(keypoints_list), center_idx + half + 1)
            members = keypoints_list[start:end]

        window_coords = [
            _normalize_shape(kp.reshape(NUM_KEYPOINTS, 3)[:, :2])
            for kp in members
            if kp is not None
        ]

        if len(window_coords) < 2:
            return 0.0

        coords_array = np.array(window_coords)   # (window, 46, 2)
        mean_variance = float(np.mean(np.var(coords_array, axis=0)))
        return 1.0 / (1.0 + mean_variance * VARIANCE_SCALE)


# =============================================================================
# Funkcje pomocnicze (prywatne)
# =============================================================================


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Odległość euklidesowa między dwoma punktami."""
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def _normalize_shape(coords: np.ndarray) -> np.ndarray:
    """
    Normalizuje kształt twarzy: centruje na punkcie środkowym oczu i skaluje
    przez odległość między oczami. Usuwa translację i skalę (blisko/daleko),
    zostawiając samą zmianę kształtu wyrazu.

    Args:
        coords: Współrzędne keypoints (46, 2)

    Returns:
        Znormalizowane współrzędne (46, 2)
    """
    left_center = (coords[KP.LEFT_EYE_INNER] + coords[KP.LEFT_EYE_OUTER]) / 2
    right_center = (coords[KP.RIGHT_EYE_INNER] + coords[KP.RIGHT_EYE_OUTER]) / 2
    mid_eye = (left_center + right_center) / 2
    eye_dist = _dist(left_center, right_center)
    scale = eye_dist if eye_dist > 1e-6 else 1.0
    return (coords - mid_eye) / scale


# Indeksy krytycznych keypoints (oczy, nos, uszy ORAZ usta/wargi — kluczowe dla
# baseline AU dolnej twarzy: bez nich mouth-AU liczone od śmiecia).
_CRITICAL_KP_INDICES: list[int] = [
    KP.LEFT_EYE_INNER,
    KP.RIGHT_EYE_INNER,
    KP.NOSE_TIP,
    KP.LEFT_EAR_BASE_FRONT,
    KP.RIGHT_EAR_BASE_FRONT,
    KP.MOUTH_LEFT_CORNER,
    KP.MOUTH_RIGHT_CORNER,
    KP.UPPER_LIP_CENTER,
    KP.LOWER_LIP_CENTER,
]


def _is_frontal_pose(
    pose: HeadPose,
    max_yaw: float,
    max_pitch: float,
    max_roll: float,
) -> bool:
    """Sprawdza czy poza głowy jest frontalna."""
    return (
        abs(pose.yaw) <= max_yaw
        and abs(pose.pitch) <= max_pitch
        and abs(pose.roll) <= max_roll
    )


def _critical_keypoints_visible(kp: np.ndarray, threshold: float) -> bool:
    """Sprawdza czy wszystkie krytyczne keypoints są widoczne."""
    return all(kp[idx, 2] >= threshold for idx in _CRITICAL_KP_INDICES)


def _count_visible_critical_kps(kp: np.ndarray, threshold: float) -> int:
    """Liczy widoczne krytyczne keypoints."""
    return sum(1 for idx in _CRITICAL_KP_INDICES if kp[idx, 2] >= threshold)


def _select_most_typical(
    candidates: list[int],
    scores: dict[int, float],
    keypoints_list: list[Optional[np.ndarray]],
) -> int:
    """
    Wybiera klatkę o konfiguracji najbliższej globalnej medianie kształtu.

    Łagodzi fakt, że "stabilna" ≠ "neutralna": wśród najstabilniejszych kandydatów
    preferuje tego najbliższego typowej (modalnej) konfiguracji po wszystkich kandydatach.
    Założenie heurystyczne: typowe = rozluźnione. To proxy, nie twardy fakt.

    Args:
        candidates: Indeksy kandydatów
        scores: Mapa indeks → stability score
        keypoints_list: Lista keypoints (None dozwolone)

    Returns:
        Indeks wybranej klatki
    """
    if len(candidates) == 1:
        return candidates[0]

    shapes = {
        idx: _normalize_shape(keypoints_list[idx].reshape(NUM_KEYPOINTS, 3)[:, :2])
        for idx in candidates
    }
    median_shape = np.median(np.array(list(shapes.values())), axis=0)

    ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
    top_n = max(1, math.ceil(len(ranked) * TOP_STABLE_FRACTION))
    shortlist = ranked[:top_n]

    return min(
        shortlist,
        key=lambda i: float(np.sum((shapes[i] - median_shape) ** 2)),
    )


# =============================================================================
# Funkcje publiczne
# =============================================================================


def compute_neutral_baseline(
    keypoints_list: list[Optional[np.ndarray]],
    neutral_idx: int,
    head_poses: list[Optional[HeadPose]],
    window_size: int = 10,
    max_yaw: float = 35.0,
    max_pitch: float = 40.0,
) -> np.ndarray:
    """
    Buduje odporny baseline neutralny jako per-keypoint medianę po oknie klatek.

    Zamiast jednej (szumnej) klatki neutralnej bierze medianę x,y po valid+frontalnych
    klatkach w oknie ±window_size//2 wokół neutral_idx (po realnym indeksie). Gasi szum
    lokalizacji keypoints (±piksele). Punkt bez widocznych próbek → wartość z neutral_idx.

    Args:
        keypoints_list: Lista keypoints (138 wartości) lub None, indeksowana po klatkach
        neutral_idx: Indeks wybranej klatki neutralnej (w tej samej liście)
        head_poses: Lista HeadPose lub None (równoległa do keypoints_list)
        window_size: Rozmiar okna czasowego
        max_yaw: Maks. yaw klatki wchodzącej do mediany (frontalność)
        max_pitch: Maks. pitch klatki wchodzącej do mediany

    Returns:
        Wektor (138,) medianowej bazy neutralnej
    """
    neutral = keypoints_list[neutral_idx].reshape(NUM_KEYPOINTS, 3)
    half = window_size // 2
    lo, hi = neutral_idx - half, neutral_idx + half

    members = [
        keypoints_list[j].reshape(NUM_KEYPOINTS, 3)
        for j in range(max(0, lo), min(len(keypoints_list), hi + 1))
        if keypoints_list[j] is not None
        and head_poses[j] is not None
        and abs(head_poses[j].yaw) <= max_yaw
        and abs(head_poses[j].pitch) <= max_pitch
    ]
    if not members:
        return neutral.flatten()

    stack = np.array(members)  # (M, 46, 3)
    baseline = neutral.copy()
    for k in range(NUM_KEYPOINTS):
        visible = stack[stack[:, k, 2] >= BASELINE_VIS_THRESHOLD, k, :]
        if len(visible) > 0:
            baseline[k, 0] = float(np.median(visible[:, 0]))
            baseline[k, 1] = float(np.median(visible[:, 1]))
            baseline[k, 2] = float(np.median(visible[:, 2]))
        # else: zostaw wartość z klatki neutral (fallback)
    return baseline.flatten()
