"""
Trekowanie psów między klatkami wideo.

Delta AU liczy się względem klatki neutralnej JEDNEGO psa, więc bez trwałej
tożsamości pies może zostać podmieniony między klatkami. Skojarzenie opiera się
na dwóch sygnałach: pokryciu bboxów (IoU) i podobieństwie koloru (histogram HSV).

Zasada nadrzędna: lepiej rozerwać trek niż zmieszać dwa psy — zmieszanie po cichu
psuje bazę AU, rozerwanie tylko skraca serię.
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from packages.models.bbox import Detection

HISTOGRAM_BINS: tuple[int, int] = (8, 8)
HISTOGRAM_RANGES: tuple[float, ...] = (0.0, 180.0, 0.0, 256.0)
MIN_APPEARANCE_SIMILARITY: float = 0.1


@dataclass
class _Track:
    """Stan pojedynczego treku."""

    track_id: int
    bbox: tuple[int, int, int, int]
    histogram: np.ndarray
    frames_since_seen: int = 0


@dataclass
class DogTracker:
    """
    Przypisuje trwałe track_id detekcjom psów w kolejnych klatkach.

    Attributes:
        iou_weight: Waga pokrycia bboxów w koszcie dopasowania
        appearance_weight: Waga podobieństwa histogramu koloru
        max_gap_frames: Po ilu klatkach bez dopasowania trek wygasa
        min_match_score: Minimalna jakość dopasowania; poniżej powstaje nowy trek
    """

    iou_weight: float = 0.6
    appearance_weight: float = 0.4
    max_gap_frames: int = 3
    min_match_score: float = 0.35

    _tracks: list[_Track] = field(default_factory=list, init=False)
    _next_id: int = field(default=0, init=False)

    def update(self, frame: np.ndarray, detections: list[Detection]) -> list[int]:
        """
        Przypisuje track_id detekcjom z bieżącej klatki.

        Args:
            frame: Pełna klatka wideo (BGR)
            detections: Detekcje psów w tej klatce

        Returns:
            Lista track_id w kolejności zgodnej z `detections`
        """
        histograms = [self._histogram(frame, det.bbox) for det in detections]
        assigned: list[Optional[int]] = [None] * len(detections)
        used_tracks: set[int] = set()

        pairs = self._score_pairs(detections, histograms, used_tracks)
        for score, det_idx, track in pairs:
            if assigned[det_idx] is not None or track.track_id in used_tracks:
                continue
            if score < self.min_match_score:
                continue
            track.bbox = detections[det_idx].bbox
            track.histogram = histograms[det_idx]
            track.frames_since_seen = 0
            assigned[det_idx] = track.track_id
            used_tracks.add(track.track_id)

        for det_idx, track_id in enumerate(assigned):
            if track_id is None:
                assigned[det_idx] = self._start_track(
                    detections[det_idx].bbox, histograms[det_idx]
                )

        self._age_tracks(used_tracks)
        return [track_id for track_id in assigned if track_id is not None]

    def _score_pairs(
        self,
        detections: list[Detection],
        histograms: list[np.ndarray],
        used_tracks: set[int],
    ) -> list[tuple[float, int, _Track]]:
        """Zwraca pary (jakość, indeks detekcji, trek) posortowane malejąco."""
        pairs: list[tuple[float, int, _Track]] = []
        for det_idx, detection in enumerate(detections):
            for track in self._tracks:
                if track.track_id in used_tracks:
                    continue
                appearance = _histogram_similarity(histograms[det_idx], track.histogram)
                if appearance < MIN_APPEARANCE_SIMILARITY:
                    # kolor jednoznacznie inny - to na pewno inny pies,
                    # nawet przy idealnym pokryciu bboxów (patrz docstring modułu)
                    continue
                iou = _iou(detection.bbox, track.bbox)
                score = self.iou_weight * iou + self.appearance_weight * appearance
                pairs.append((score, det_idx, track))
        pairs.sort(key=lambda item: item[0], reverse=True)
        return pairs

    def _start_track(
        self, bbox: tuple[int, int, int, int], histogram: np.ndarray
    ) -> int:
        """Zakłada nowy trek i zwraca jego id."""
        track = _Track(track_id=self._next_id, bbox=bbox, histogram=histogram)
        self._tracks.append(track)
        self._next_id += 1
        return track.track_id

    def _age_tracks(self, used_tracks: set[int]) -> None:
        """Postarza treki bez dopasowania i usuwa wygasłe."""
        for track in self._tracks:
            if track.track_id not in used_tracks:
                track.frames_since_seen += 1
        self._tracks = [
            track for track in self._tracks if track.frames_since_seen <= self.max_gap_frames
        ]

    @staticmethod
    def _histogram(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Liczy znormalizowany histogram HSV (odcień + nasycenie) wnętrza bboxa."""
        x, y, w, h = bbox
        crop = frame[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0:
            return np.zeros(HISTOGRAM_BINS[0] * HISTOGRAM_BINS[1], dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, list(HISTOGRAM_BINS), list(HISTOGRAM_RANGES))
        cv2.normalize(hist, hist)
        return hist.flatten()


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    """Pokrycie dwóch bboxów (x, y, w, h) w zakresie [0, 1]."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    inter_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_y = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = inter_x * inter_y
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def _histogram_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Podobieństwo histogramów w zakresie [0, 1] (korelacja obcięta do zera)."""
    if first.size == 0 or second.size == 0:
        return 0.0
    correlation = float(
        cv2.compareHist(first.astype(np.float32), second.astype(np.float32), cv2.HISTCMP_CORREL)
    )
    return max(0.0, correlation)
