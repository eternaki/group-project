"""
Temporalna agregacja Delta AU dla stabilnych predykcji emocji.

Aggruje delta Action Units (AU) w czasie (okno N klatek) dla uzyskania
stabilnych predykcji emocji zamiast szumnych predykcji klatka-po-klatce.

Proces:
1. Wymagana neutralna klatka bazowa (neutral keypoints)
2. Dla każdej klatki: oblicz delta AU względem neutralnej
3. Filtruj klatki z nieodpowiednią pozą głowy lub niską widocznością
4. Agreguj delta AU w oknie czasowym
5. Zwróć uśrednione delta AU gdy bufor jest gotowy
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from packages.models.delta_action_units import (
    ACTION_UNIT_NAMES,
    DeltaActionUnit,
    DeltaActionUnitsExtractor,
)
from packages.models.head_pose import (
    DEFAULT_MAX_ROLL,
    DEFAULT_MAX_YAW_ASYMMETRY,
    estimate_head_pose,
    validate_head_pose,
)


@dataclass
class TemporalAUResult:
    """
    Wynik czasowej agregacji delta AU.

    Attributes:
        values: Uśrednione wartości ratio dla każdego AU
        variance: Wariancja każdego AU w oknie czasowym
        num_frames: Liczba klatek w buforze
        is_stable: True jeśli wariancja poniżej progu stabilności
    """

    values: dict[str, float]
    variance: dict[str, float]
    num_frames: int
    is_stable: bool

    def to_feature_vector(self) -> np.ndarray:
        """Konwertuje do wektora cech (ratio dla każdego AU)."""
        return np.array(
            [self.values[name] for name in ACTION_UNIT_NAMES],
            dtype=np.float32,
        )


class TemporalAUBuffer:
    """
    Bufor do czasowej agregacji delta Action Units.

    Przechowuje historię ratio AU z ostatnich N klatek i oblicza
    uśrednione wartości oraz wariancję.

    Przykład:
        buffer = TemporalAUBuffer(window_size=30)
        buffer.add_frame(au_ratios, confidence=0.8)
        if buffer.is_ready():
            result = buffer.get_aggregated()
            if result.is_stable:
                # Użyj result.values do klasyfikacji emocji
    """

    def __init__(
        self,
        window_size: int = 30,
        stability_threshold: float = 0.05,
        min_frames: int = 10,
    ) -> None:
        """
        Inicjalizuje bufor.

        Args:
            window_size: Rozmiar okna (liczba klatek)
            stability_threshold: Próg wariancji dla stabilności ratio
            min_frames: Minimalna liczba klatek do rozpoczęcia agregacji
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.min_frames = min_frames

        # Historia: list of (au_ratios_dict, confidence)
        self._history: deque = deque(maxlen=window_size)

    def add_frame(
        self,
        au_ratios: dict[str, float],
        confidence: float = 1.0,
    ) -> None:
        """
        Dodaje klatkę do bufora.

        Args:
            au_ratios: Słownik AU_name -> ratio (np. {"AU12": 1.25, ...})
            confidence: Pewność detekcji (do ważenia)
        """
        self._history.append((au_ratios.copy(), confidence))

    def clear(self) -> None:
        """Czyści bufor."""
        self._history.clear()

    def is_ready(self) -> bool:
        """Sprawdza czy bufor ma wystarczająco klatek."""
        return len(self._history) >= self.min_frames

    def get_aggregated(self) -> Optional[TemporalAUResult]:
        """
        Zwraca uśrednione delta AU.

        Returns:
            TemporalAUResult lub None jeśli za mało klatek
        """
        if not self.is_ready():
            return None

        au_arrays: dict[str, list[float]] = {name: [] for name in ACTION_UNIT_NAMES}
        weights: list[float] = []

        for au_ratios, confidence in self._history:
            weights.append(confidence)
            for name in ACTION_UNIT_NAMES:
                au_arrays[name].append(au_ratios.get(name, 1.0))  # 1.0 = brak zmiany

        weights_arr = np.array(weights)
        weights_arr = weights_arr / weights_arr.sum()

        mean_values: dict[str, float] = {}
        variance_values: dict[str, float] = {}

        for name in ACTION_UNIT_NAMES:
            values = np.array(au_arrays[name])
            mean_values[name] = float(np.average(values, weights=weights_arr))
            variance_values[name] = float(np.var(values))

        max_variance = max(variance_values.values()) if variance_values else 0.0
        is_stable = max_variance < self.stability_threshold

        return TemporalAUResult(
            values=mean_values,
            variance=variance_values,
            num_frames=len(self._history),
            is_stable=is_stable,
        )


class TemporalProcessor:
    """
    Procesor do czasowej obróbki wideo z delta AU.

    Łączy:
    - Estymację pozy głowy (filtrowanie nie-frontalnych klatek)
    - Ekstrakcję delta Action Units względem klatki neutralnej
    - Temporalną agregację w oknie czasowym

    Przykład:
        neutral_kp = keypoints_list[neutral_idx]
        processor = TemporalProcessor(neutral_keypoints=neutral_kp)

        for frame_keypoints in video_keypoints:
            result = processor.process_frame(frame_keypoints)
            if result is not None and result.is_stable:
                # Klasyfikuj emocję z result.values
    """

    def __init__(
        self,
        neutral_keypoints: np.ndarray,
        window_size: int = 30,
        max_yaw_asymmetry: float = DEFAULT_MAX_YAW_ASYMMETRY,
        max_roll: float = DEFAULT_MAX_ROLL,
        min_visibility: float = 0.5,
        stability_threshold: float = 0.05,
    ) -> None:
        """
        Inicjalizuje procesor.

        Args:
            neutral_keypoints: Keypoints neutralnej klatki (138 wartości)
            window_size: Rozmiar okna agregacji (klatki)
            max_yaw_asymmetry: Maks. asymetria kącik oka <-> nos dla frontalnej pozy
            max_roll: Maksymalne przechylenie (stopnie) dla frontalnej pozy
            min_visibility: Minimalna średnia widoczność keypoints
            stability_threshold: Próg wariancji ratio AU dla stabilności
        """
        self.neutral_keypoints = neutral_keypoints
        self.max_yaw_asymmetry = max_yaw_asymmetry
        self.max_roll = max_roll
        self.min_visibility = min_visibility

        # Baza neutralna jest stała dla całego wideo — miary liczone raz.
        self._extractor = DeltaActionUnitsExtractor(neutral_keypoints)

        self._buffer = TemporalAUBuffer(
            window_size=window_size,
            stability_threshold=stability_threshold,
        )

        # Statystyki
        self._total_frames: int = 0
        self._accepted_frames: int = 0
        self._rejected_head_pose: int = 0
        self._rejected_visibility: int = 0

    def reset(self) -> None:
        """Resetuje stan procesora."""
        self._buffer.clear()
        self._total_frames = 0
        self._accepted_frames = 0
        self._rejected_head_pose = 0
        self._rejected_visibility = 0

    def process_frame(
        self,
        keypoints_flat: np.ndarray,
    ) -> Optional[TemporalAUResult]:
        """
        Przetwarza jedną klatkę.

        Args:
            keypoints_flat: Keypoints klatki [x0, y0, v0, ...] (138 wartości)

        Returns:
            TemporalAUResult jeśli bufor gotowy, inaczej None
        """
        self._total_frames += 1

        # 1. Sprawdź widoczność
        kp = keypoints_flat.reshape(-1, 3)
        mean_visibility = float(np.mean(kp[:, 2]))

        if mean_visibility < self.min_visibility:
            self._rejected_visibility += 1
            return self._buffer.get_aggregated() if self._buffer.is_ready() else None

        # 2. Sprawdź pozę głowy
        head_pose = estimate_head_pose(
            keypoints_flat,
            max_yaw_asymmetry=self.max_yaw_asymmetry,
            max_roll=self.max_roll,
        )
        if not validate_head_pose(
            head_pose,
            max_yaw_asymmetry=self.max_yaw_asymmetry,
            min_confidence=self.min_visibility,
        ):
            self._rejected_head_pose += 1
            return self._buffer.get_aggregated() if self._buffer.is_ready() else None

        # 3. Oblicz delta AU względem klatki neutralnej
        delta_aus: dict[str, DeltaActionUnit] = self._extractor.extract(keypoints_flat)

        # 4. Konwertuj do dict[str, float] (ratio) i dodaj do bufora
        au_ratios = {name: au.ratio for name, au in delta_aus.items()}
        self._buffer.add_frame(au_ratios, confidence=head_pose.confidence)
        self._accepted_frames += 1

        # 5. Zwróć zagregowany wynik
        return self._buffer.get_aggregated()

    def process_video_sequence(
        self,
        keypoints_sequence: list[np.ndarray],
    ) -> Optional[TemporalAUResult]:
        """
        Przetwarza sekwencję keypoints z wideo.

        Args:
            keypoints_sequence: Lista keypoints dla każdej klatki

        Returns:
            Finalny TemporalAUResult lub None
        """
        self.reset()
        result = None

        for keypoints in keypoints_sequence:
            result = self.process_frame(keypoints)

        return result

    def get_statistics(self) -> dict:
        """Zwraca statystyki przetwarzania."""
        return {
            "total_frames": self._total_frames,
            "accepted_frames": self._accepted_frames,
            "rejected_head_pose": self._rejected_head_pose,
            "rejected_visibility": self._rejected_visibility,
            "acceptance_rate": (
                self._accepted_frames / self._total_frames
                if self._total_frames > 0
                else 0.0
            ),
        }
