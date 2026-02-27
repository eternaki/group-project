"""
Detekcja klatki neutralnej dla obliczeń delta AU.

Wykrywa "neutralną klatkę bazową" w sekwencji wideo, gdzie wyraz twarzy
psa jest najbardziej rozluźniony i stabilny (minimalne ruchy).

Neutralna klatka służy jako punkt odniesienia do obliczania delta AU.
"""

from typing import Optional

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS
from packages.models.head_pose import HeadPose, estimate_head_pose


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
    ) -> int:
        """
        Automatycznie wykrywa neutralną klatkę z sekwencji wideo.

        Args:
            frames: Lista klatek wideo
            keypoints_list: Lista tablic keypoints (138 wartości każda)
            head_poses: Opcjonalna lista HeadPose (obliczana jeśli None)
            debug: Włącz logowanie debugowania

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

        scores = [
            (idx, self._compute_stability_score(keypoints_list, idx))
            for idx in candidates
        ]
        best_idx, _ = max(scores, key=lambda x: x[1])
        return best_idx

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
    ) -> float:
        """
        Oblicza wynik stabilności klatki.

        Stabilność = 1 / (1 + wariancja). Wyższy = bardziej neutralna.

        Args:
            keypoints_list: Lista wszystkich keypoints (może zawierać None)
            center_idx: Indeks klatki do oceny

        Returns:
            Wynik stabilności (wyższy = bardziej stabilna)
        """
        start = max(0, center_idx - self.window_size // 2)
        end = min(len(keypoints_list), center_idx + self.window_size // 2 + 1)

        window_coords = [
            kp.reshape(NUM_KEYPOINTS, 3)[:, :2]
            for kp in keypoints_list[start:end]
            if kp is not None
        ]

        if len(window_coords) < 2:
            return 0.0

        coords_array = np.array(window_coords)   # (window, 46, 2)
        mean_variance = float(np.mean(np.var(coords_array, axis=0)))
        return 1.0 / (1.0 + mean_variance * 10)


# =============================================================================
# Funkcje pomocnicze (prywatne)
# =============================================================================

# Indeksy krytycznych keypoints (oczy, nos, uszy)
_CRITICAL_KP_INDICES: list[int] = [
    KP.LEFT_EYE_INNER,
    KP.RIGHT_EYE_INNER,
    KP.NOSE_TIP,
    KP.LEFT_EAR_BASE_FRONT,
    KP.RIGHT_EAR_BASE_FRONT,
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
