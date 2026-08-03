# Pipeline wielu psów — plan implementacji

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Przetwarzać każdego psa na wideo jako osobny trek z własną klatką neutralną, własnymi AU i wygładzonymi keypoints, a w COCO zapisywać dane nadające się do trenowania własnej sieci AU.

**Architecture:** Trwały `track_id` z detekcji (IoU + histogram koloru) daje każdemu psu własny układ odniesienia. Na tym opierają się: klatka neutralna per trek, filtr One Euro na keypoints i poza głowy liczona metrykami niezależnymi od długości pyska. Kanoniczny kształt (Prokrustes) trafia do anotacji jako wejście dla przyszłej sieci.

**Tech Stack:** Python 3.12, NumPy, OpenCV, PyTorch (istniejące modele), pytest, ruff.

## Global Constraints

- Dokumentacja i komentarze **po polsku**, nazwy zmiennych i funkcji **po angielsku**.
- Type hints zawsze; format `ruff` (PEP 8); f-strings.
- Funkcja robi jedną rzecz (≤50 linii, złożoność ≤10), max 5–6 parametrów.
- Bez mutable default args, bez bare `except`, bez hardcodowanych ścieżek i magic numbers.
- Commity: `[SPRINT-14][TASK] Krótki opis po polsku` + stopka
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Praca na gałęzi `feature/pipeline-audit` (już istnieje, odgałęziona od `develop`).
- Po każdym zadaniu: `pytest -q` i `ruff check .` muszą być czyste.
- Uruchamianie: `.venv/Scripts/python.exe -m pytest ...` (Windows, Python 3.12).
- Liczba keypoints: `NUM_KEYPOINTS = 46`, format płaski `[x0, y0, v0, ...]` = 138 wartości.
- Spec: `docs/superpowers/specs/2026-08-03-multi-dog-pipeline-design.md`.

---

## Struktura plików

**Nowe:**
| Plik | Odpowiedzialność |
|------|------------------|
| `packages/pipeline/dog_tracker.py` | Przypisanie `track_id` detekcjom między klatkami |
| `packages/pipeline/landmark_smoothing.py` | Filtr One Euro na keypoints w obrębie treku |
| `packages/models/shape_normalization.py` | Superpozycja Prokrustesa (kanoniczny kształt) |
| `packages/pipeline/track_processing.py` | Przetwarzanie jednego treku: neutralna → AU → peaki |

**Modyfikowane:**
| Plik | Zmiana |
|------|--------|
| `packages/models/head_pose.py` | Nowe metryki pozy, usunięcie `pitch` |
| `packages/pipeline/neutral_frame.py` | `_frontal_factor` i progi bez `pitch` |
| `packages/pipeline/peak_selector.py` | Filtr pozy na nowej metryce, odstęp w sekundach |
| `packages/pipeline/inference.py` | `process_video_for_dataset` zwraca treki |
| `packages/data/coco.py` | Nowe pola anotacji |
| `scripts/annotation/batch_annotate.py` | Anotacja na (klatka, trek) |
| `scripts/annotation/tag_dataset_quality.py` | Tiery na nowej metryce pozy |
| `scripts/debug/audit_pipeline.py` | Pomiar nowej metryki + wielu treków |
| `apps/webapp/backend/main.py` | Sesja z listą psów |

---

## Task 1: Trekowanie psów między klatkami

**Files:**
- Create: `packages/pipeline/dog_tracker.py`
- Test: `tests/test_pipeline/test_dog_tracker.py`

**Interfaces:**
- Consumes: `packages.models.bbox.Detection` (pola: `bbox: tuple[int,int,int,int]` jako `(x, y, w, h)`, `confidence: float`, `class_id: int`, `class_name: str`).
- Produces: `DogTracker(iou_weight, appearance_weight, max_gap_frames, min_match_score)` z metodą `update(frame: np.ndarray, detections: list[Detection]) -> list[int]` zwracającą `track_id` dla każdej detekcji w tej samej kolejności.

- [ ] **Step 1: Napisz test na zachowanie tożsamości i rozdzielanie psów**

```python
"""Testy trekowania psów między klatkami."""

import numpy as np

from packages.models.bbox import Detection
from packages.pipeline.dog_tracker import DogTracker


def _frame_with_patch(color: tuple[int, int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """Buduje czarną klatkę 400x400 z jednokolorowym prostokątem w miejscu psa."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    x, y, w, h = box
    frame[y : y + h, x : x + w] = color
    return frame


def _detection(box: tuple[int, int, int, int]) -> Detection:
    """Buduje detekcję psa o zadanym bboxie."""
    return Detection(bbox=box, confidence=0.9, class_id=0, class_name="dog")


class TestDogTracker:
    """Przypisywanie stabilnych track_id."""

    def test_ten_sam_pies_zachowuje_id_miedzy_klatkami(self):
        box_a = (100, 100, 60, 60)
        box_b = (108, 104, 60, 60)  # niewielkie przesunięcie
        tracker = DogTracker()

        ids_first = tracker.update(_frame_with_patch((0, 0, 200), box_a), [_detection(box_a)])
        ids_second = tracker.update(_frame_with_patch((0, 0, 200), box_b), [_detection(box_b)])

        assert ids_first == ids_second

    def test_dwa_psy_dostaja_rozne_id(self):
        box_left = (30, 30, 50, 50)
        box_right = (300, 300, 50, 50)
        frame = _frame_with_patch((0, 0, 200), box_left)
        frame[300:350, 300:350] = (0, 200, 0)
        tracker = DogTracker()

        ids = tracker.update(frame, [_detection(box_left), _detection(box_right)])

        assert ids[0] != ids[1]

    def test_psy_o_tym_samym_polozeniu_ale_innym_kolorze_nie_zlewaja_sie(self):
        """Kolor rozstrzyga, gdy geometria jest myląca (pies zniknął, inny wszedł w to miejsce)."""
        box = (100, 100, 60, 60)
        tracker = DogTracker()

        first = tracker.update(_frame_with_patch((0, 0, 200), box), [_detection(box)])
        second = tracker.update(_frame_with_patch((0, 200, 0), box), [_detection(box)])

        assert first != second

    def test_krotka_przerwa_wraca_do_tego_samego_treku(self):
        box = (100, 100, 60, 60)
        frame = _frame_with_patch((0, 0, 200), box)
        tracker = DogTracker(max_gap_frames=3)

        first = tracker.update(frame, [_detection(box)])
        tracker.update(np.zeros((400, 400, 3), dtype=np.uint8), [])
        again = tracker.update(frame, [_detection(box)])

        assert first == again

    def test_dluga_przerwa_tworzy_nowy_trek(self):
        box = (100, 100, 60, 60)
        frame = _frame_with_patch((0, 0, 200), box)
        tracker = DogTracker(max_gap_frames=2)

        first = tracker.update(frame, [_detection(box)])
        for _ in range(3):
            tracker.update(np.zeros((400, 400, 3), dtype=np.uint8), [])
        again = tracker.update(frame, [_detection(box)])

        assert first != again
```

- [ ] **Step 2: Uruchom test i potwierdź czerwony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline/test_dog_tracker.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.pipeline.dog_tracker'`

- [ ] **Step 3: Zaimplementuj tracker**

```python
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
                iou = _iou(detection.bbox, track.bbox)
                appearance = _histogram_similarity(histograms[det_idx], track.histogram)
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
```

- [ ] **Step 4: Uruchom testy i potwierdź zielony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline/test_dog_tracker.py -q`
Expected: PASS (5 testów)

- [ ] **Step 5: Sprawdź linter i zacommituj**

```bash
.venv/Scripts/python.exe -m ruff check packages/pipeline/dog_tracker.py tests/test_pipeline/test_dog_tracker.py
git add packages/pipeline/dog_tracker.py tests/test_pipeline/test_dog_tracker.py
git commit -m "[SPRINT-14][TASK] Trekowanie psów: IoU + histogram koloru

Delta AU liczy się względem klatki neutralnej jednego psa — bez trwałej tożsamości
pies bywał podmieniany między klatkami (audyt: 12.5% klatek ma wiele psów).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Poza głowy niezależna od długości pyska

**Files:**
- Modify: `packages/models/head_pose.py` (całość — `pitch` znika)
- Modify: `packages/pipeline/neutral_frame.py:325-341` (`_frontal_factor`), progi `max_pitch`
- Modify: `packages/pipeline/peak_selector.py:299-310` (`_is_valid_peak`)
- Modify: `scripts/annotation/tag_dataset_quality.py:57-84`
- Modify: `scripts/debug/audit_pipeline.py` (pomiar nowej metryki)
- Modify: `tests/test_pipeline/test_neutral_frame.py:153-215, 286-294`
- Test: `tests/test_models/test_head_pose.py`

**Interfaces:**
- Produces: `HeadPose(yaw_asymmetry: float, roll: float, is_frontal: bool, confidence: float)`
  — pole `pitch` i pole `yaw` (w stopniach) **nie istnieją**.
  `estimate_head_pose(keypoints_flat: np.ndarray, max_yaw_asymmetry: float = 0.35, max_roll: float = 30.0) -> HeadPose`.
  `yaw_asymmetry` ∈ [−1, 1]: 0 = front, znak wskazuje kierunek obrotu.

**Uzasadnienie (zmierzone):** stary `pitch` = kąt nos–linia oczu. U psa nos zawsze jest
poniżej oczu, więc to miara anatomii, nie pozy. Na ręcznie anotowanym DogFLW (480 obrazów):
mediana `pitch` **+47.5°**, 100% wartości dodatnich, **91.5%** przekracza 30°.
Nowa metryka na tym samym zbiorze: mediana `yaw_asymmetry` **+0.003**, 52.1% dodatnich,
**0.0%** przekracza 0.35. `roll`: mediana 0.0°, 1.9% przekracza 30°.

- [ ] **Step 1: Napisz test nowej metryki**

```python
"""Testy pozy głowy psa (metryki niezależne od długości pyska)."""

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS
from packages.models.head_pose import estimate_head_pose


def _keypoints(points: dict[int, tuple[float, float]]) -> np.ndarray:
    """Buduje płaską tablicę keypoints; punkty niepodane są w (0,0) z widocznością 1."""
    flat = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
    flat[2::3] = 1.0
    for index, (x, y) in points.items():
        flat[index * 3] = x
        flat[index * 3 + 1] = y
    return flat


class TestYawAsymmetry:
    """Obrót lewo/prawo mierzony asymetrią odległości kąciki oczu ↔ nos."""

    def test_symetryczna_morda_daje_zero(self):
        pose = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (50.0, 120.0),
            })
        )

        assert abs(pose.yaw_asymmetry) < 1e-6

    def test_dlugi_pysk_nie_wplywa_na_wynik(self):
        """Nos przesunięty w dół (długi pysk) nadal daje pozę frontalną."""
        pose = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (50.0, 300.0),
            })
        )

        assert abs(pose.yaw_asymmetry) < 1e-6
        assert pose.is_frontal is True

    def test_obrot_w_bok_daje_przeciwne_znaki(self):
        left_turn = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (20.0, 120.0),
            })
        )
        right_turn = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (80.0, 120.0),
            })
        )

        assert left_turn.yaw_asymmetry * right_turn.yaw_asymmetry < 0

    def test_metryka_niezalezna_od_skali_obrazu(self):
        small = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (20.0, 120.0),
            })
        )
        large = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (400.0, 500.0),
                KP.RIGHT_EYE_INNER: (600.0, 500.0),
                KP.NOSE_TIP: (200.0, 1200.0),
            })
        )

        assert abs(small.yaw_asymmetry - large.yaw_asymmetry) < 1e-6


class TestRoll:
    """Przechylenie liczone z linii wewnętrznych kącików oczu."""

    def test_pozioma_linia_oczu_daje_zero(self):
        pose = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (50.0, 120.0),
            })
        )

        assert abs(pose.roll) < 1e-6

    def test_przechylona_glowa_nie_jest_frontalna(self):
        pose = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 20.0),
                KP.RIGHT_EYE_INNER: (60.0, 80.0),
                KP.NOSE_TIP: (50.0, 120.0),
            }),
            max_roll=30.0,
        )

        assert pose.is_frontal is False


class TestNaReferencjiDogFLW:
    """Metryka sprawdzona na ręcznie anotowanych landmarkach (nie na predykcjach)."""

    def test_rozklad_na_zbiorze_testowym_jest_wysrodkowany(self, tmp_path):
        import glob
        import json

        files = sorted(glob.glob("data/dogflw_raw/DogFLW/test/labels/*.json"))
        if not files:
            import pytest

            pytest.skip("Brak lokalnej kopii DogFLW")

        values = []
        for path in files:
            landmarks = json.load(open(path, encoding="utf-8"))["landmarks"]
            if len(landmarks) != NUM_KEYPOINTS:
                continue
            flat = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
            for index, (x, y) in enumerate(landmarks):
                flat[index * 3] = x
                flat[index * 3 + 1] = y
                flat[index * 3 + 2] = 1.0
            values.append(estimate_head_pose(flat).yaw_asymmetry)

        median_abs = float(np.median(np.abs(values)))
        share_rejected = float(np.mean(np.abs(values) > 0.35))

        assert median_abs < 0.10, "Fronty muszą dawać wartości bliskie zeru"
        assert share_rejected < 0.05, "Twardy limit nie może odrzucać frontalnych mord"
```

- [ ] **Step 2: Uruchom test i potwierdź czerwony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_head_pose.py -q`
Expected: FAIL — `AttributeError: 'HeadPose' object has no attribute 'yaw_asymmetry'`

- [ ] **Step 3: Przepisz `packages/models/head_pose.py`**

Zastąp `HeadPose`, `HeadPoseEstimator.estimate` i funkcje pomocnicze:

```python
@dataclass
class HeadPose:
    """
    Wynik estymacji pozy głowy.

    Attributes:
        yaw_asymmetry: Obrót lewo/prawo jako asymetria odległości kącik oka ↔ nos,
            zakres [-1, 1], 0 = morda frontalna. Metryka bezwymiarowa, niezależna
            od długości pyska i od skali obrazu.
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
```

W `HeadPoseEstimator.__init__` zamień `frontal_threshold` na `max_yaw_asymmetry: float = 0.35`
i `max_roll: float = 30.0`. W `estimate()` licz obie metryki na `KP.LEFT_EYE_INNER`,
`KP.RIGHT_EYE_INNER`, `KP.NOSE_TIP`, ustaw
`is_frontal = abs(yaw_asymmetry) <= self.max_yaw_asymmetry and abs(roll) <= self.max_roll`.
Usuń `_compute_yaw`, `_compute_pitch` i parametr `max_angle` z `validate_head_pose`
(zastąp go `max_yaw_asymmetry`).

- [ ] **Step 4: Zaktualizuj konsumentów**

`packages/pipeline/neutral_frame.py` — `_frontal_factor`:

```python
def _frontal_factor(pose: Optional[HeadPose]) -> float:
    """
    Współczynnik frontalności kandydata na klatkę neutralną.

    Liczony z obrotu i przechylenia; miara „nos poniżej oczu" nie jest używana,
    bo odzwierciedla długość pyska, nie pozę.
    """
    if pose is None:
        return 0.5
    deviation = abs(pose.yaw_asymmetry) / 0.35 + abs(pose.roll) / 30.0
    return 1.0 / (1.0 + deviation)
```

W `NeutralFrameDetector.__init__` zamień `max_pitch` na `max_roll: float = 30.0`,
a `max_yaw` (stopnie) na `max_yaw_asymmetry: float = 0.35`; w `_is_frontal_pose`
porównuj `abs(pose.yaw_asymmetry)` i `abs(pose.roll)`.

`packages/pipeline/peak_selector.py` — `_is_valid_peak`, blok „2. Head pose check":

```python
        if self.frontal_only:
            if not head_pose.is_frontal:
                return False
        else:
            if abs(head_pose.yaw_asymmetry) > self.max_yaw_asymmetry:
                return False
            if abs(head_pose.roll) > self.max_roll:
                return False
```

W `PeakFrameSelector.__init__` zamień `max_head_angle: float = 40.0` na
`max_yaw_asymmetry: float = 0.35` i `max_roll: float = 30.0`.

`scripts/annotation/tag_dataset_quality.py` — progi tierów:

```python
    strict = (
        anat and abs(hp.yaw_asymmetry) <= 0.15 and abs(hp.roll) <= 20 and ...
    )
    good = (
        anat and mean_conf >= 0.6 and abs(hp.yaw_asymmetry) <= 0.25
        and abs(hp.roll) <= 30 and ...
    )
```

oraz w zapisie: `"yaw_asymmetry": round(hp.yaw_asymmetry, 3), "roll": round(hp.roll, 1)`
zamiast `yaw`/`pitch`.

`scripts/debug/audit_pipeline.py` — zamień pola `yaw`/`pitch` w `AuditStats` na
`yaw_asymmetry`/`roll`, w `audit_frame_data` zbieraj `abs(pose.yaw_asymmetry)` i
`abs(pose.roll)`, w `build_report` zmień `etap_3_poza_glowy` na te dwie metryki
z progami 0.35 i 30. W `audit_video` frontalność klatki neutralnej licz jako
`abs(pose.yaw_asymmetry) / 0.35 + abs(pose.roll) / 30.0`.

`tests/test_pipeline/test_neutral_frame.py` — w `hp()` i konstrukcjach `HeadPose`
zamień argumenty na `yaw_asymmetry=` i usuń `pitch=`; usuń test
`assert abs(pose.pitch) < 30  # nos naturalnie poniżej oczu ~26°` (dokumentował błąd);
w teście progów zamień `detector.max_pitch == 40.0` na `detector.max_roll == 30.0`.

- [ ] **Step 5: Uruchom pełne testy**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (wszystkie, w tym nowe testy pozy)

- [ ] **Step 6: Sprawdź linter i zacommituj**

```bash
.venv/Scripts/python.exe -m ruff check .
git add -A
git commit -m "[SPRINT-14][TASK] Poza głowy: metryki niezależne od długości pyska

Stary pitch (nos poniżej oczu) mierzył anatomię, nie pozę: na ręcznie anotowanym
DogFLW mediana +47.5 st., 100% wartości dodatnich, 91.5% powyżej progu 30 st.
Zastąpiony asymetrią odległości kąciki oczu <-> nos (mediana +0.003, 0% odrzuceń)
i przechyleniem z linii oczu. Metryki wzorowane na pracy o morfometrii geometrycznej
psów (Sci Rep 2025).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wygładzanie keypoints filtrem One Euro

**Files:**
- Create: `packages/pipeline/landmark_smoothing.py`
- Test: `tests/test_pipeline/test_landmark_smoothing.py`

**Interfaces:**
- Produces: `KeypointSmoother(min_cutoff: float = 1.0, beta: float = 0.3, d_cutoff: float = 1.0)`
  z metodą `smooth(keypoints_flat: np.ndarray, face_box: tuple[float, float, float, float], timestamp: float) -> np.ndarray`.
  Zwraca tablicę 138 wartości w układzie obrazu. Jedna instancja = jeden trek.

**Uzasadnienie (zmierzone):** próg aktywacji AU to `ratio > 1.15` (sygnał 0.15), a zmierzone
σ ratio w obrębie wideo wynosi 0.35–0.76 — szum przewyższa próg 2.3–5×. Filtr 1€ (Casiez i in.
2012, stosowany w MediaPipe) jest adaptacyjny: tłumi drgania przy wolnym ruchu i przepuszcza
ruch szybki, więc nie zatrze samego szczytu mimiki. Zwykła mediana po oknie by go zatarła.

- [ ] **Step 1: Napisz test tłumienia szumu i przepuszczania skoku**

```python
"""Testy wygładzania keypoints w obrębie treku."""

import numpy as np

from packages.data.schemas import NUM_KEYPOINTS
from packages.pipeline.landmark_smoothing import KeypointSmoother

FACE_BOX = (100.0, 100.0, 200.0, 200.0)


def _noisy_series(rng: np.random.Generator, steps: int, noise: float) -> list[np.ndarray]:
    """Nieruchoma morda z szumem gaussowskim na każdej współrzędnej."""
    base = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
    base[0::3] = 150.0
    base[1::3] = 150.0
    base[2::3] = 1.0

    series = []
    for _ in range(steps):
        frame = base.copy()
        frame[0::3] += rng.normal(0, noise, NUM_KEYPOINTS)
        frame[1::3] += rng.normal(0, noise, NUM_KEYPOINTS)
        series.append(frame)
    return series


class TestKeypointSmoother:
    """Filtr One Euro na trek."""

    def test_tlumi_drganie_nieruchomej_mordy(self):
        rng = np.random.default_rng(0)
        series = _noisy_series(rng, steps=30, noise=3.0)
        smoother = KeypointSmoother()

        smoothed = [
            smoother.smooth(frame, FACE_BOX, timestamp=i / 5.0)
            for i, frame in enumerate(series)
        ]

        raw_std = float(np.std([frame[0] for frame in series]))
        smooth_std = float(np.std([frame[0] for frame in smoothed[5:]]))

        assert smooth_std < raw_std / 2, f"szum ma spaść, było {raw_std}, jest {smooth_std}"

    def test_przepuszcza_szybki_ruch_mimiki(self):
        """Skok (szczyt mimiki) nie może zostać zatarty przez filtr."""
        smoother = KeypointSmoother()
        base = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
        base[0::3] = 150.0
        base[1::3] = 150.0
        base[2::3] = 1.0

        for i in range(10):
            smoother.smooth(base, FACE_BOX, timestamp=i / 5.0)

        jumped = base.copy()
        jumped[1::3] = 180.0  # wyraźne otwarcie pyska
        result = smoother.smooth(jumped, FACE_BOX, timestamp=10 / 5.0)

        assert result[1] > 170.0, "szybka zmiana ma zostać przepuszczona"

    def test_zachowuje_widocznosc_punktow(self):
        smoother = KeypointSmoother()
        frame = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
        frame[2::3] = 0.7

        result = smoother.smooth(frame, FACE_BOX, timestamp=0.0)

        assert np.allclose(result[2::3], 0.7)

    def test_ruch_psa_po_kadrze_nie_rozmywa_punktow(self):
        """Wygładzanie działa w układzie mordy, więc przesunięcie boksu nie szkodzi."""
        smoother = KeypointSmoother()
        first = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
        first[0::3] = 150.0
        first[1::3] = 150.0
        first[2::3] = 1.0

        smoother.smooth(first, (100.0, 100.0, 200.0, 200.0), timestamp=0.0)

        moved = first.copy()
        moved[0::3] = 350.0  # ta sama morda, przesunięta o 200 px w prawo
        result = smoother.smooth(moved, (300.0, 100.0, 200.0, 200.0), timestamp=0.2)

        assert result[0] > 330.0, "punkt ma podążyć za mordą, a nie zostać w tyle"
```

- [ ] **Step 2: Uruchom test i potwierdź czerwony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline/test_landmark_smoothing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.pipeline.landmark_smoothing'`

- [ ] **Step 3: Zaimplementuj filtr**

```python
"""
Wygładzanie keypoints w obrębie treku filtrem One Euro.

Zmierzony szum ratio AU (σ 0.35–0.76) przewyższa próg aktywacji (0.15) kilkukrotnie,
więc AU zapalają się od drgania punktów, nie od mimiki. Filtr 1€ (Casiez i in. 2012,
używany w MediaPipe) jest adaptacyjny: przy wolnym ruchu mocno tłumi, przy szybkim
przepuszcza — dzięki temu nie zaciera szczytu mimiki, którego szukamy.

Wygładzanie działa we współrzędnych względem boksu mordy: gdyby liczyć w pikselach
kadru, ruch psa po kadrze rozmyłby punkty.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from packages.data.schemas import NUM_KEYPOINTS


class _OneEuroFilter:
    """Filtr 1€ dla pojedynczego sygnału skalarnego."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._previous_value: Optional[float] = None
        self._previous_derivative: float = 0.0
        self._previous_timestamp: Optional[float] = None

    def filter(self, value: float, timestamp: float) -> float:
        """Zwraca wygładzoną wartość dla podanej chwili."""
        if self._previous_value is None or self._previous_timestamp is None:
            self._previous_value = value
            self._previous_timestamp = timestamp
            return value

        elapsed = timestamp - self._previous_timestamp
        if elapsed <= 0:
            return self._previous_value

        rate = 1.0 / elapsed
        derivative = (value - self._previous_value) * rate
        derivative = _exponential_smoothing(
            _alpha(rate, self.d_cutoff), derivative, self._previous_derivative
        )

        cutoff = self.min_cutoff + self.beta * abs(derivative)
        smoothed = _exponential_smoothing(
            _alpha(rate, cutoff), value, self._previous_value
        )

        self._previous_value = smoothed
        self._previous_derivative = derivative
        self._previous_timestamp = timestamp
        return smoothed


@dataclass
class KeypointSmoother:
    """
    Wygładza 46 keypoints jednego treku w czasie.

    Attributes:
        min_cutoff: Częstotliwość odcięcia przy braku ruchu (mniejsza = mocniejsze tłumienie)
        beta: Wpływ prędkości na odcięcie (większa = szybsza reakcja na ruch)
        d_cutoff: Odcięcie dla estymaty prędkości
    """

    min_cutoff: float = 1.0
    beta: float = 0.3
    d_cutoff: float = 1.0

    _filters: dict[int, _OneEuroFilter] = field(default_factory=dict, init=False)

    def smooth(
        self,
        keypoints_flat: np.ndarray,
        face_box: tuple[float, float, float, float],
        timestamp: float,
    ) -> np.ndarray:
        """
        Wygładza keypoints jednej klatki treku.

        Args:
            keypoints_flat: Keypoints [x0, y0, v0, ...] w układzie obrazu (138 wartości)
            face_box: Boks mordy (x, y, w, h) w układzie obrazu
            timestamp: Czas klatki w sekundach (od początku wideo)

        Returns:
            Wygładzone keypoints w układzie obrazu (138 wartości)

        Raises:
            ValueError: Gdy liczba wartości keypoints jest nieprawidłowa
        """
        expected = NUM_KEYPOINTS * 3
        if len(keypoints_flat) != expected:
            raise ValueError(f"Oczekiwano {expected} wartości, otrzymano {len(keypoints_flat)}")

        face_x, face_y, face_w, face_h = face_box
        if face_w < 1e-6 or face_h < 1e-6:
            return np.asarray(keypoints_flat, dtype=float).copy()

        result = np.asarray(keypoints_flat, dtype=float).copy()
        for index in range(NUM_KEYPOINTS):
            local_x = (result[index * 3] - face_x) / face_w
            local_y = (result[index * 3 + 1] - face_y) / face_h

            smoothed_x = self._filter_for(index * 2).filter(local_x, timestamp)
            smoothed_y = self._filter_for(index * 2 + 1).filter(local_y, timestamp)

            result[index * 3] = smoothed_x * face_w + face_x
            result[index * 3 + 1] = smoothed_y * face_h + face_y
        return result

    def _filter_for(self, key: int) -> _OneEuroFilter:
        """Zwraca (lub tworzy) filtr dla jednej współrzędnej."""
        if key not in self._filters:
            self._filters[key] = _OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)
        return self._filters[key]


def _alpha(rate: float, cutoff: float) -> float:
    """Współczynnik wygładzania wykładniczego dla zadanego odcięcia."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * rate)


def _exponential_smoothing(alpha: float, value: float, previous: float) -> float:
    """Wygładzanie wykładnicze."""
    return alpha * value + (1.0 - alpha) * previous
```

- [ ] **Step 4: Uruchom testy i potwierdź zielony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline/test_landmark_smoothing.py -q`
Expected: PASS (4 testy). Jeśli `test_przepuszcza_szybki_ruch_mimiki` nie przechodzi,
zwiększ `beta` (domyślnie 0.3) — filtr reaguje wtedy szybciej na ruch.

- [ ] **Step 5: Sprawdź linter i zacommituj**

```bash
.venv/Scripts/python.exe -m ruff check packages/pipeline/landmark_smoothing.py tests/test_pipeline/test_landmark_smoothing.py
git add packages/pipeline/landmark_smoothing.py tests/test_pipeline/test_landmark_smoothing.py
git commit -m "[SPRINT-14][TASK] Wygładzanie keypoints filtrem One Euro w obrębie treku

Zmierzony szum ratio AU (sigma 0.35-0.76) przewyższa próg aktywacji 0.15 nawet 5x.
Filtr 1€ (Casiez 2012, MediaPipe) tłumi drgania przy wolnym ruchu i przepuszcza
szybki, więc nie zaciera szczytu mimiki. Działa w układzie mordy, nie kadru.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Kanoniczny kształt (superpozycja Prokrustesa)

**Files:**
- Create: `packages/models/shape_normalization.py`
- Test: `tests/test_models/test_shape_normalization.py`

**Interfaces:**
- Produces:
  `procrustes_align(keypoints_flat: np.ndarray, reference_shape: np.ndarray) -> np.ndarray`
  — zwraca 138 wartości `[x, y, v, ...]` w przestrzeni kształtu (widoczność bez zmian);
  `mean_shape(shapes: list[np.ndarray], iterations: int = 3) -> np.ndarray`
  — kształt referencyjny metodą GPA, zwraca tablicę `(46, 2)`.

**Uzasadnienie:** superpozycja Prokrustesa usuwa przesunięcie, obrót i skalę naraz — to
standard geometrycznej morfometrii, użyty przez autorów DogFLW do analizy mimiki psów
(Sci Rep 2025). Obecne dzielenie odległości przez rozstaw oczu usuwa tylko skalę i sprzęga
AU z pozą (przy obrocie rozstaw oczu maleje perspektywicznie, więc wszystkie ratio rosną).

- [ ] **Step 1: Napisz test niezmienniczości**

```python
"""Testy normalizacji kształtu metodą Prokrustesa."""

import numpy as np

from packages.data.schemas import NUM_KEYPOINTS
from packages.models.shape_normalization import mean_shape, procrustes_align


def _shape_to_flat(coords: np.ndarray) -> np.ndarray:
    """Zamienia tablicę (46, 2) na płaską [x, y, v, ...] z widocznością 1."""
    flat = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
    flat[0::3] = coords[:, 0]
    flat[1::3] = coords[:, 1]
    flat[2::3] = 1.0
    return flat


def _random_shape(seed: int) -> np.ndarray:
    """Losowy, ale powtarzalny kształt (46, 2)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 50, (NUM_KEYPOINTS, 2)) + 200.0


class TestProcrustesAlign:
    """Niezmienniczość na przesunięcie, skalę i obrót."""

    def test_przesuniecie_nie_zmienia_wyniku(self):
        coords = _random_shape(1)
        reference = _random_shape(2)

        first = procrustes_align(_shape_to_flat(coords), reference)
        second = procrustes_align(_shape_to_flat(coords + 137.0), reference)

        assert np.allclose(first, second, atol=1e-6)

    def test_skala_nie_zmienia_wyniku(self):
        coords = _random_shape(3)
        reference = _random_shape(2)

        first = procrustes_align(_shape_to_flat(coords), reference)
        second = procrustes_align(_shape_to_flat(coords * 2.5), reference)

        assert np.allclose(first, second, atol=1e-6)

    def test_obrot_nie_zmienia_wyniku(self):
        coords = _random_shape(4)
        reference = _random_shape(2)
        angle = np.radians(37.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        first = procrustes_align(_shape_to_flat(coords), reference)
        second = procrustes_align(_shape_to_flat(coords @ rotation.T), reference)

        assert np.allclose(first, second, atol=1e-6)

    def test_zachowuje_widocznosc(self):
        coords = _random_shape(5)
        flat = _shape_to_flat(coords)
        flat[2::3] = 0.42

        result = procrustes_align(flat, _random_shape(2))

        assert np.allclose(result[2::3], 0.42)

    def test_rozne_ksztalty_daja_rozne_wyniki(self):
        reference = _random_shape(2)

        first = procrustes_align(_shape_to_flat(_random_shape(6)), reference)
        second = procrustes_align(_shape_to_flat(_random_shape(7)), reference)

        assert not np.allclose(first, second, atol=1e-3)


class TestMeanShape:
    """Kształt referencyjny metodą GPA."""

    def test_srednia_z_jednego_ksztaltu_to_ten_ksztalt_po_normalizacji(self):
        coords = _random_shape(8)

        result = mean_shape([_shape_to_flat(coords)])

        assert result.shape == (NUM_KEYPOINTS, 2)
        assert abs(float(np.mean(result))) < 1e-6, "kształt referencyjny jest wyśrodkowany"

    def test_srednia_odporna_na_przesuniecia_wejsc(self):
        coords = _random_shape(9)
        shapes = [_shape_to_flat(coords + offset) for offset in (0.0, 50.0, -30.0)]

        result = mean_shape(shapes)
        expected = mean_shape([_shape_to_flat(coords)])

        assert np.allclose(result, expected, atol=1e-6)
```

- [ ] **Step 2: Uruchom test i potwierdź czerwony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_shape_normalization.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.models.shape_normalization'`

- [ ] **Step 3: Zaimplementuj normalizację**

```python
"""
Normalizacja kształtu twarzy psa metodą Prokrustesa.

Superpozycja Prokrustesa usuwa przesunięcie, skalę i obrót, zostawiając czysty
kształt. To standard geometrycznej morfometrii, użyty przez autorów zbioru DogFLW
do analizy mimiki psów. Dzięki temu porównujemy mimikę mopsa i owczarka w jednej
przestrzeni, a poza głowy przestaje wpływać na wartości.

Wynik trafia do anotacji COCO jako `procrustes_keypoints` — wejście dla przyszłej
sieci AU (Sprint 16).
"""

import numpy as np

from packages.data.schemas import NUM_KEYPOINTS


def _to_coords(keypoints_flat: np.ndarray) -> np.ndarray:
    """Wyciąga współrzędne (46, 2) z płaskiej tablicy."""
    expected = NUM_KEYPOINTS * 3
    if len(keypoints_flat) != expected:
        raise ValueError(f"Oczekiwano {expected} wartości, otrzymano {len(keypoints_flat)}")
    array = np.asarray(keypoints_flat, dtype=float).reshape(NUM_KEYPOINTS, 3)
    return array[:, :2]


def _center_and_scale(coords: np.ndarray) -> np.ndarray:
    """Centruje kształt w zerze i skaluje do jednostkowej normy Frobeniusa."""
    centered = coords - coords.mean(axis=0)
    norm = float(np.sqrt(np.sum(centered**2)))
    if norm < 1e-9:
        return centered
    return centered / norm


def _optimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Znajduje obrót najlepiej dopasowujący `source` do `target` (rozkład SVD)."""
    correlation = source.T @ target
    u, _, vt = np.linalg.svd(correlation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def procrustes_align(
    keypoints_flat: np.ndarray,
    reference_shape: np.ndarray,
) -> np.ndarray:
    """
    Nakłada kształt na referencję metodą Prokrustesa.

    Args:
        keypoints_flat: Keypoints [x0, y0, v0, ...] (138 wartości)
        reference_shape: Kształt referencyjny (46, 2), np. z `mean_shape()`

    Returns:
        Płaska tablica 138 wartości w przestrzeni kształtu; widoczność bez zmian

    Raises:
        ValueError: Gdy liczba wartości keypoints jest nieprawidłowa
    """
    coords = _to_coords(keypoints_flat)
    normalized = _center_and_scale(coords)
    reference = _center_and_scale(np.asarray(reference_shape, dtype=float))

    aligned = normalized @ _optimal_rotation(normalized, reference)

    result = np.asarray(keypoints_flat, dtype=float).copy()
    result[0::3] = aligned[:, 0]
    result[1::3] = aligned[:, 1]
    return result


def mean_shape(shapes: list[np.ndarray], iterations: int = 3) -> np.ndarray:
    """
    Liczy kształt referencyjny metodą uogólnionej analizy Prokrustesa (GPA).

    Args:
        shapes: Lista płaskich tablic keypoints (po 138 wartości)
        iterations: Liczba iteracji uzgadniania średniej

    Returns:
        Kształt referencyjny (46, 2), wyśrodkowany i przeskalowany

    Raises:
        ValueError: Gdy lista kształtów jest pusta
    """
    if not shapes:
        raise ValueError("Potrzeba co najmniej jednego kształtu")

    normalized = [_center_and_scale(_to_coords(shape)) for shape in shapes]
    reference = normalized[0]

    for _ in range(iterations):
        aligned = [shape @ _optimal_rotation(shape, reference) for shape in normalized]
        reference = _center_and_scale(np.mean(aligned, axis=0))

    return reference
```

- [ ] **Step 4: Uruchom testy i potwierdź zielony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_shape_normalization.py -q`
Expected: PASS (7 testów)

- [ ] **Step 5: Wygeneruj kształt referencyjny z DogFLW i zapisz**

```bash
.venv/Scripts/python.exe - <<'EOF'
import glob, json
import numpy as np
from packages.data.schemas import NUM_KEYPOINTS
from packages.models.shape_normalization import mean_shape

shapes = []
for path in sorted(glob.glob("data/dogflw_raw/DogFLW/train/labels/*.json")):
    landmarks = json.load(open(path, encoding="utf-8"))["landmarks"]
    if len(landmarks) != NUM_KEYPOINTS:
        continue
    flat = np.zeros(NUM_KEYPOINTS * 3)
    flat[0::3] = [p[0] for p in landmarks]
    flat[1::3] = [p[1] for p in landmarks]
    flat[2::3] = 1.0
    shapes.append(flat)

reference = mean_shape(shapes)
json.dump(reference.tolist(), open("models/dogflw_mean_shape.json", "w"))
print(f"Kształt referencyjny z {len(shapes)} obrazów -> models/dogflw_mean_shape.json")
EOF
```

Expected: komunikat z liczbą obrazów (rzędu 3000) i utworzony plik.

- [ ] **Step 6: Sprawdź linter i zacommituj**

```bash
.venv/Scripts/python.exe -m ruff check packages/models/shape_normalization.py tests/test_models/test_shape_normalization.py
git add packages/models/shape_normalization.py tests/test_models/test_shape_normalization.py models/dogflw_mean_shape.json
git commit -m "[SPRINT-14][TASK] Kanoniczny kształt twarzy: superpozycja Prokrustesa

Standard geometrycznej morfometrii (użyty przez autorów DogFLW): usuwa przesunięcie,
skalę i obrót naraz. Obecne dzielenie przez rozstaw oczu usuwało tylko skalę i sprzęgało
AU z pozą. Kształt referencyjny policzony metodą GPA ze zbioru treningowego DogFLW.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Przetwarzanie pojedynczego treku

**Files:**
- Create: `packages/pipeline/track_processing.py`
- Test: `tests/test_pipeline/test_track_processing.py`

**Interfaces:**
- Consumes: `KeypointSmoother.smooth()` (Task 3), `estimate_head_pose()` (Task 2),
  `DeltaActionUnitsExtractor(neutral_keypoints, activation_threshold)` z metodą
  `extract(target_kp) -> dict[str, DeltaActionUnit]`,
  `NeutralFrameDetector.detect_auto(frames, keypoints_list, head_poses)`,
  `collect_neutral_baseline(keypoints_list, neutral_idx, window)`,
  `PeakFrameSelector.select(...)`, `compute_tfm(delta_aus)`.
- Produces:
  ```python
  @dataclass
  class TrackFrame:
      frame_idx: int
      keypoints: np.ndarray          # 138 wartości, wygładzone, układ obrazu
      face_box: tuple[float, float, float, float]
      head_pose: HeadPose
      delta_aus: dict[str, DeltaActionUnit]

  @dataclass
  class TrackResult:
      track_id: int
      neutral_frame_idx: int
      frames: list[TrackFrame]
      peak_indices: list[int]        # indeksy klatek wideo, nie pozycje w `frames`
      au_noise: dict[str, float]     # σ ratio na trek, per AU
      rejected_reason: Optional[str] # None gdy trek przyjęty

  MIN_TRACK_FRAMES = 3
  MIN_FACE_SIZE_PX = 64.0
  MIN_KEYPOINT_CONF = 0.4

  def evaluate_track_quality(frames: list[TrackFrame]) -> Optional[str]
  def compute_au_noise(frames: list[TrackFrame]) -> dict[str, float]
  ```

- [ ] **Step 1: Napisz test progów jakości i pomiaru szumu**

```python
"""Testy przetwarzania pojedynczego treku."""

import numpy as np

from packages.models.delta_action_units import DeltaActionUnit
from packages.models.head_pose import HeadPose
from packages.pipeline.track_processing import (
    TrackFrame,
    compute_au_noise,
    evaluate_track_quality,
)


def _track_frame(
    frame_idx: int,
    face_size: float = 120.0,
    confidence: float = 0.8,
    au_ratio: float = 1.0,
) -> TrackFrame:
    """Buduje klatkę treku o zadanym rozmiarze mordy, pewności i wartości AU."""
    keypoints = np.zeros(46 * 3, dtype=float)
    keypoints[2::3] = confidence
    return TrackFrame(
        frame_idx=frame_idx,
        keypoints=keypoints,
        face_box=(0.0, 0.0, face_size, face_size),
        head_pose=HeadPose(yaw_asymmetry=0.0, roll=0.0, is_frontal=True, confidence=0.9),
        delta_aus={
            "AU101": DeltaActionUnit(
                name="AU101",
                ratio=au_ratio,
                delta=au_ratio - 1.0,
                is_active=False,
                confidence=0.8,
            )
        },
    )


class TestEvaluateTrackQuality:
    """Próg godności treku — odrzucenie musi mieć podany powód."""

    def test_dobry_trek_jest_przyjmowany(self):
        frames = [_track_frame(i) for i in range(5)]

        assert evaluate_track_quality(frames) is None

    def test_za_malo_klatek_odrzucone_z_powodem(self):
        frames = [_track_frame(i) for i in range(2)]

        reason = evaluate_track_quality(frames)

        assert reason is not None and "klatek" in reason

    def test_za_mala_morda_odrzucona_z_powodem(self):
        frames = [_track_frame(i, face_size=40.0) for i in range(5)]

        reason = evaluate_track_quality(frames)

        assert reason is not None and "morda" in reason

    def test_niska_pewnosc_odrzucona_z_powodem(self):
        frames = [_track_frame(i, confidence=0.2) for i in range(5)]

        reason = evaluate_track_quality(frames)

        assert reason is not None and "pewność" in reason


class TestComputeAuNoise:
    """Szum AU na trek — waga wiarygodności dla przyszłego treningu."""

    def test_staly_sygnal_daje_zerowy_szum(self):
        frames = [_track_frame(i, au_ratio=1.0) for i in range(5)]

        noise = compute_au_noise(frames)

        assert noise["AU101"] == 0.0

    def test_zmienny_sygnal_daje_dodatni_szum(self):
        ratios = [0.8, 1.2, 0.9, 1.4, 1.0]
        frames = [_track_frame(i, au_ratio=r) for i, r in enumerate(ratios)]

        noise = compute_au_noise(frames)

        assert noise["AU101"] > 0.15, "rozrzut ma być widoczny w mierze szumu"
```

- [ ] **Step 2: Uruchom test i potwierdź czerwony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline/test_track_processing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.pipeline.track_processing'`

- [ ] **Step 3: Zaimplementuj moduł**

```python
"""
Przetwarzanie jednego treku psa: klatka neutralna → delta AU → klatki peak.

Każdy pies ma własny układ odniesienia. Wcześniej cała sekwencja miała jedną klatkę
neutralną, więc na wideo z wieloma psami AU liczyły się względem neutralnej innego psa.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from packages.models.delta_action_units import DeltaActionUnit
from packages.models.head_pose import HeadPose

MIN_TRACK_FRAMES: int = 3
MIN_FACE_SIZE_PX: float = 64.0
MIN_KEYPOINT_CONF: float = 0.4


@dataclass
class TrackFrame:
    """Jedna klatka w obrębie treku."""

    frame_idx: int
    keypoints: np.ndarray
    face_box: tuple[float, float, float, float]
    head_pose: HeadPose
    delta_aus: dict[str, DeltaActionUnit]


@dataclass
class TrackResult:
    """Wynik przetworzenia jednego treku."""

    track_id: int
    neutral_frame_idx: int
    frames: list[TrackFrame]
    peak_indices: list[int]
    au_noise: dict[str, float]
    rejected_reason: Optional[str] = None


def evaluate_track_quality(frames: list[TrackFrame]) -> Optional[str]:
    """
    Sprawdza, czy trek nadaje się do zbioru.

    Args:
        frames: Klatki treku z wykrytą mordą

    Returns:
        None gdy trek przyjęty, w przeciwnym razie powód odrzucenia po polsku
    """
    if len(frames) < MIN_TRACK_FRAMES:
        return f"za mało klatek z mordą: {len(frames)} < {MIN_TRACK_FRAMES}"

    median_face = float(np.median([min(frame.face_box[2], frame.face_box[3]) for frame in frames]))
    if median_face < MIN_FACE_SIZE_PX:
        return f"za mała morda: {median_face:.0f} px < {MIN_FACE_SIZE_PX:.0f} px"

    median_conf = float(
        np.median([float(np.mean(frame.keypoints[2::3])) for frame in frames])
    )
    if median_conf < MIN_KEYPOINT_CONF:
        return f"za niska pewność keypoints: {median_conf:.2f} < {MIN_KEYPOINT_CONF:.2f}"

    return None


def compute_au_noise(frames: list[TrackFrame]) -> dict[str, float]:
    """
    Liczy odchylenie standardowe ratio każdego AU w obrębie treku.

    Wartość trafia do anotacji jako waga wiarygodności: klatka z rozdygotanego treku
    nie powinna ważyć w treningu tyle samo, co ze stabilnego.

    Args:
        frames: Klatki treku

    Returns:
        Słownik nazwa AU → odchylenie standardowe ratio
    """
    ratios: dict[str, list[float]] = {}
    for frame in frames:
        for name, au in frame.delta_aus.items():
            ratios.setdefault(name, []).append(float(au.ratio))
    return {name: float(np.std(values)) for name, values in ratios.items()}
```

- [ ] **Step 4: Uruchom testy i potwierdź zielony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline/test_track_processing.py -q`
Expected: PASS (6 testów)

- [ ] **Step 5: Sprawdź linter i zacommituj**

```bash
.venv/Scripts/python.exe -m ruff check packages/pipeline/track_processing.py tests/test_pipeline/test_track_processing.py
git add packages/pipeline/track_processing.py tests/test_pipeline/test_track_processing.py
git commit -m "[SPRINT-14][TASK] Przetwarzanie treku: próg godności i pomiar szumu AU

Każdy pies dostaje własny układ odniesienia. Odrzucone treki mają zapisany powód —
bez tego znów nie wiedzielibyśmy, gdzie tracimy dane.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Integracja — `process_video_for_dataset` zwraca treki

**Files:**
- Modify: `packages/pipeline/inference.py:596-900` (`process_video_for_dataset`)
- Test: `tests/test_pipeline/test_video_tracks.py`

**Interfaces:**
- Consumes: `DogTracker` (Task 1), `KeypointSmoother` (Task 3), `procrustes_align` (Task 4),
  `TrackFrame`, `TrackResult`, `evaluate_track_quality`, `compute_au_noise` (Task 5).
- Produces: `process_video_for_dataset(frames_list, num_peaks=10, neutral_idx=None, fps=5.0, ...) -> dict`
  ze strukturą:
  ```python
  {
      "tracks": list[TrackResult],       # tylko przyjęte treki
      "rejected_tracks": list[TrackResult],  # z wypełnionym rejected_reason
      "total_frames": int,
  }
  ```
  Klucze `neutral_frame_idx`, `peak_frames` i `all_frames_data` **znikają** —
  konsumenci przechodzą na `tracks`.

- [ ] **Step 1: Napisz test integracyjny na dwóch psach**

```python
"""Testy przetwarzania wideo na treki (wiele psów)."""

import numpy as np
import pytest

from packages.pipeline.dog_tracker import DogTracker
from packages.models.bbox import Detection


def _frame_with_two_dogs() -> np.ndarray:
    """Klatka z dwoma wyraźnie różnymi kolorystycznie psami."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 50:150] = (200, 30, 30)
    frame[100:200, 400:500] = (30, 200, 30)
    return frame


class TestTrekowanieWSekwencji:
    """Trekowanie na sekwencji klatek — bez modeli AI."""

    def test_dwa_psy_zachowuja_osobne_id_przez_cala_sekwencje(self):
        tracker = DogTracker()
        frame = _frame_with_two_dogs()
        detections = [
            Detection(bbox=(50, 100, 100, 100), confidence=0.9, class_id=0, class_name="dog"),
            Detection(bbox=(400, 100, 100, 100), confidence=0.9, class_id=0, class_name="dog"),
        ]

        sequences = [tracker.update(frame, detections) for _ in range(5)]

        assert all(ids == sequences[0] for ids in sequences)
        assert sequences[0][0] != sequences[0][1]


@pytest.mark.skipif(
    not __import__("pathlib").Path("models/keypoints_dogflw.pt").exists(),
    reason="Brak wag modeli — test integracyjny wymaga pobrania przez git lfs pull",
)
class TestProcessVideoForDataset:
    """Pełna ścieżka wideo → treki (wymaga wag modeli)."""

    def test_zwraca_strukture_z_trekami(self):
        from packages.pipeline import InferencePipeline, PipelineConfig

        pipeline = InferencePipeline(PipelineConfig(device="cpu"))
        pipeline.load()
        frames = [_frame_with_two_dogs() for _ in range(6)]

        result = pipeline.process_video_for_dataset(frames_list=frames, num_peaks=2, fps=5.0)

        assert "tracks" in result
        assert "rejected_tracks" in result
        assert "total_frames" in result
        assert result["total_frames"] == 6

    def test_odrzucone_treki_maja_powod(self):
        from packages.pipeline import InferencePipeline, PipelineConfig

        pipeline = InferencePipeline(PipelineConfig(device="cpu"))
        pipeline.load()
        frames = [_frame_with_two_dogs() for _ in range(6)]

        result = pipeline.process_video_for_dataset(frames_list=frames, num_peaks=2, fps=5.0)

        for track in result["rejected_tracks"]:
            assert track.rejected_reason, "każde odrzucenie musi mieć powód"
```

- [ ] **Step 2: Uruchom test i potwierdź czerwony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline/test_video_tracks.py -q`
Expected: FAIL — `KeyError: 'tracks'` (albo test pominięty, gdy brak wag — wtedy uruchom
`git lfs pull` i powtórz)

- [ ] **Step 3: Przebuduj `process_video_for_dataset`**

Zastąp pętlę „Step 1" (obecne linie ~685-700, `detection = detections[0]`) przypisaniem
treków i zbieraniem klatek per trek:

```python
        tracker = DogTracker()
        frames_by_track: dict[int, list[tuple[int, Detection]]] = {}

        for frame_idx, frame in enumerate(frames_list):
            detections = self.bbox_model.predict(frame)
            track_ids = tracker.update(frame, detections)
            for track_id, detection in zip(track_ids, detections):
                frames_by_track.setdefault(track_id, []).append((frame_idx, detection))
```

Następnie dla każdego treku wykonaj dotychczasowe kroki 2–6 (keypoints → klatka neutralna →
delta AU → peaki → emocje), z trzema różnicami:

1. Keypoints licz tylko na klatkach tego treku, kadrując `detection.bbox`
   (`_detect_face` + `_keypoints_on_region`, fallback `_square_crop` jak dotąd).
2. Po wykryciu keypoints przepuść je przez `KeypointSmoother` tego treku:
   `smoother.smooth(keypoints, face_box, timestamp=frame_idx / fps)`.
3. Klatkę neutralną, `DeltaActionUnitsExtractor` i `PeakFrameSelector` twórz **osobno
   dla każdego treku** — nigdy nie współdziel bazy między trekami.

Trek bez klatki neutralnej albo odrzucony przez `evaluate_track_quality` trafia do
`rejected_tracks` z wypełnionym `rejected_reason` — **bez rzucania wyjątku**:

```python
            reason = evaluate_track_quality(track_frames)
            if reason is not None:
                rejected.append(
                    TrackResult(
                        track_id=track_id,
                        neutral_frame_idx=-1,
                        frames=track_frames,
                        peak_indices=[],
                        au_noise={},
                        rejected_reason=reason,
                    )
                )
                continue
```

Dodaj parametr `fps: float = 5.0` do sygnatury metody (potrzebny do znacznika czasu
w filtrze i do przeliczenia odstępu peaków na sekundy).

Szkielet pętli po trekach (zastępuje dotychczasowe kroki 2–6 działające na całym wideo):

```python
        accepted: list[TrackResult] = []
        rejected: list[TrackResult] = []

        for track_id, entries in frames_by_track.items():
            smoother = KeypointSmoother()
            track_frames: list[TrackFrame] = []

            for frame_idx, detection in entries:
                frame = frames_list[frame_idx]
                x, y, w, h = detection.bbox
                face = self._detect_face(frame, x, y, w, h)
                if face is None:
                    continue
                prediction = self._keypoints_on_region(frame, *face)
                if prediction is None:
                    continue

                raw = np.array(
                    [v for kp in prediction.keypoints for v in (kp.x, kp.y, kp.visibility)]
                )
                smoothed = smoother.smooth(raw, face, timestamp=frame_idx / fps)
                track_frames.append(
                    TrackFrame(
                        frame_idx=frame_idx,
                        keypoints=smoothed,
                        face_box=face,
                        head_pose=estimate_head_pose(smoothed),
                        delta_aus={},   # uzupełniane po wyborze klatki neutralnej
                    )
                )

            reason = evaluate_track_quality(track_frames)
            if reason is not None:
                rejected.append(
                    TrackResult(
                        track_id=track_id,
                        neutral_frame_idx=-1,
                        frames=track_frames,
                        peak_indices=[],
                        au_noise={},
                        rejected_reason=reason,
                    )
                )
                continue

            keypoints_list = [frame.keypoints for frame in track_frames]
            head_poses = [frame.head_pose for frame in track_frames]
            local_neutral = NeutralFrameDetector().detect_auto(
                [frames_list[frame.frame_idx] for frame in track_frames],
                keypoints_list,
                head_poses,
            )
            baseline = collect_neutral_baseline(keypoints_list, local_neutral, window=2)
            extractor = DeltaActionUnitsExtractor(baseline)

            for track_frame in track_frames:
                track_frame.delta_aus = extractor.extract(track_frame.keypoints)

            selector = PeakFrameSelector(
                min_separation_frames=max(1, int(round(fps))),   # ≥1 sekunda odstępu
            )
            local_peaks = selector.select(
                frames=[frames_list[frame.frame_idx] for frame in track_frames],
                keypoints_list=keypoints_list,
                neutral_idx=local_neutral,
                delta_aus_list=[frame.delta_aus for frame in track_frames],
                head_poses=head_poses,
                num_peaks=num_peaks,
            )

            accepted.append(
                TrackResult(
                    track_id=track_id,
                    neutral_frame_idx=track_frames[local_neutral].frame_idx,
                    frames=track_frames,
                    peak_indices=[track_frames[i].frame_idx for i in local_peaks],
                    au_noise=compute_au_noise(track_frames),
                )
            )

        return {
            "tracks": accepted,
            "rejected_tracks": rejected,
            "total_frames": len(frames_list),
        }
```

Uwaga na indeksy: `NeutralFrameDetector` i `PeakFrameSelector` operują na pozycjach
w liście treku, a `TrackResult.neutral_frame_idx` i `peak_indices` przechowują **numery
klatek wideo** — konwersja przez `track_frames[i].frame_idx` jak wyżej.

Odstęp peaków `min_separation_frames=int(round(fps))` odpowiada ≥1 sekundzie —
tak jak w opublikowanej metodzie (spec, sekcja 4.2). Wcześniejsza wartość 30 klatek
przy próbkowaniu 5 fps oznaczała 6 sekund i niepotrzebnie ubożyła wynik.

- [ ] **Step 4: Uruchom pełne testy**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS. Testy odwołujące się do starych kluczy (`peak_frames`, `all_frames_data`)
zaktualizuj na `tracks` — nie usuwaj ich.

- [ ] **Step 5: Sprawdź linter i zacommituj**

```bash
.venv/Scripts/python.exe -m ruff check .
git add -A
git commit -m "[SPRINT-14][TASK] Pipeline wideo: przetwarzanie per trek zamiast jednego psa

Wcześniej brany był detections[0] (najwyższa pewność, nie największy bbox), więc na
12.5% klatek z wieloma psami baza AU mogła należeć do innego psa. Teraz każdy trek ma
własną klatkę neutralną, własne wygładzanie i własne peaki. Wideo bez godnych treków
nie rzuca już wyjątku (dotąd 17.5% wideo kończyło się bledem).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Zapis do COCO z polami dla przyszłej sieci

**Files:**
- Modify: `packages/data/coco.py:225-320` (`add_annotation`)
- Modify: `scripts/annotation/batch_annotate.py:290-420` (`process_video`)
- Test: `tests/test_data/test_coco_track_fields.py`

**Interfaces:**
- Consumes: `TrackResult`, `TrackFrame` (Task 5), `procrustes_align` (Task 4),
  `au_analysis_from_delta_aus` (istnieje w `packages/data/coco.py`).
- Produces: `add_annotation(..., track_id: Optional[int] = None, frame_role: Optional[str] = None,
  label_source: str = "auto_rules", au_noise: Optional[dict[str, float]] = None,
  procrustes_keypoints: Optional[list[float]] = None)`.

- [ ] **Step 1: Napisz test nowych pól**

```python
"""Testy pól COCO potrzebnych do trenowania sieci AU."""

from packages.data.coco import COCODataset


class TestPolaTreku:
    """track_id, rola klatki, źródło etykiety, szum AU, kanoniczny kształt."""

    def _dataset_with_annotation(self, **kwargs) -> dict:
        dataset = COCODataset()
        image_id = dataset.add_image(file_name="a.jpg", width=640, height=480)
        dataset.add_annotation(image_id=image_id, bbox=[0, 0, 10, 10], **kwargs)
        return dataset.to_dict()["annotations"][0]

    def test_zapisuje_track_id(self):
        annotation = self._dataset_with_annotation(track_id=7)

        assert annotation["track_id"] == 7

    def test_zapisuje_role_klatki(self):
        annotation = self._dataset_with_annotation(frame_role="neutral")

        assert annotation["frame_role"] == "neutral"

    def test_domyslne_zrodlo_etykiety_to_reguly(self):
        annotation = self._dataset_with_annotation()

        assert annotation["label_source"] == "auto_rules"

    def test_zapisuje_szum_au_jako_wage_wiarygodnosci(self):
        annotation = self._dataset_with_annotation(au_noise={"AU101": 0.42})

        assert annotation["au_noise"]["AU101"] == 0.42

    def test_zapisuje_ksztalt_prokrustesa(self):
        shape = [0.1] * (46 * 3)
        annotation = self._dataset_with_annotation(procrustes_keypoints=shape)

        assert len(annotation["procrustes_keypoints"]) == 138
```

- [ ] **Step 2: Uruchom test i potwierdź czerwony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data/test_coco_track_fields.py -q`
Expected: FAIL — `TypeError: add_annotation() got an unexpected keyword argument 'track_id'`

- [ ] **Step 3: Dodaj pola w `add_annotation`**

Dopisz parametry do sygnatury i blok zapisu obok istniejącego bloku „DogFACS Dataset extensions":

```python
        # Pola potrzebne do trenowania własnej sieci AU (Sprint 16)
        if track_id is not None:
            ann_dict["track_id"] = track_id
        if frame_role:
            ann_dict["frame_role"] = frame_role
        ann_dict["label_source"] = label_source
        if au_noise:
            ann_dict["au_noise"] = au_noise
        if procrustes_keypoints:
            ann_dict["procrustes_keypoints"] = procrustes_keypoints
```

Uzupełnij docstring: `label_source` przyjmuje `auto_rules` (pre-etykieta z reguł) albo
`human_verified` (po weryfikacji ręcznej, Sprint 15); `au_noise` to σ ratio na trek,
używane jako waga wiarygodności próbki.

- [ ] **Step 4: Uruchom testy i potwierdź zielony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data/ -q`
Expected: PASS (poprzednie 10 testów + 5 nowych)

- [ ] **Step 5: Przepisz zapis w `batch_annotate.py`**

W `process_video` zamień iterację po `dataset_result["peak_frames"]` na iterację po trekach.
Dla każdego treku zapisz jego klatki peak **oraz** klatkę neutralną:

```python
            reference_shape = np.array(
                json.loads(Path("models/dogflw_mean_shape.json").read_text(encoding="utf-8")),
                dtype=float,
            )

            for track in dataset_result["tracks"]:
                # Klatka neutralna idzie pierwsza — jej image_id jest potrzebne
                # jako neutral_frame_id w anotacjach peaków tego treku.
                ordered = sorted(
                    (
                        frame
                        for frame in track.frames
                        if frame.frame_idx in track.peak_indices
                        or frame.frame_idx == track.neutral_frame_idx
                    ),
                    key=lambda frame: frame.frame_idx != track.neutral_frame_idx,
                )
                neutral_image_id: Optional[int] = None

                for track_frame in ordered:
                    is_neutral = track_frame.frame_idx == track.neutral_frame_idx
                    frame_img = frames_list[track_frame.frame_idx]
                    height, width = frame_img.shape[:2]

                    frame_num = frame_numbers[track_frame.frame_idx]
                    frame_id = f"{video_id}_t{track.track_id}_{frame_num:06d}"
                    frame_path = frames_video_dir / f"{frame_id}.jpg"
                    cv2.imwrite(str(frame_path), frame_img)
                    stats["frames_processed"] += 1

                    image_id = self.coco_dataset.add_image(
                        file_name=str(frame_path.relative_to(self.config.frames_dir)),
                        width=width,
                        height=height,
                        source_video=video_id,
                        frame_number=frame_num,
                        emotion_label=emotion,
                    )
                    if is_neutral:
                        neutral_image_id = image_id

                    self.coco_dataset.add_annotation(
                        image_id=image_id,
                        bbox=[int(v) for v in track_frame.face_box],
                        keypoints=[float(v) for v in track_frame.keypoints],
                        num_keypoints=int(sum(1 for v in track_frame.keypoints[2::3] if v > 0)),
                        au_analysis=au_analysis_from_delta_aus(track_frame.delta_aus),
                        neutral_frame_id=neutral_image_id,
                        track_id=track.track_id,
                        frame_role="neutral" if is_neutral else "peak",
                        label_source="auto_rules",
                        au_noise=track.au_noise,
                        procrustes_keypoints=[
                            float(v)
                            for v in procrustes_align(track_frame.keypoints, reference_shape)
                        ],
                    )

            for track in dataset_result["rejected_tracks"]:
                logger.info(
                    "Trek %s odrzucony: %s", track.track_id, track.rejected_reason
                )
```

Emocję i rasę licz jak dotąd, na klatce peak — `classify_emotion_from_delta_aus(track_frame.delta_aus)`
oraz `self.pipeline.breed_model.predict(...)` na kropie `track_frame.face_box`.

- [ ] **Step 6: Uruchom testy, linter i zacommituj**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
git add -A
git commit -m "[SPRINT-14][TASK] COCO: pola treku i dane dla przyszłej sieci AU

track_id, frame_role (peak/neutral), label_source, au_noise (waga wiarygodności próbki)
i procrustes_keypoints (kanoniczny kształt = wejście sieci). Batch zapisuje teraz
anotację na (klatka, trek) oraz klatkę neutralną każdego treku.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Webapp na wiele psów i ponowny audyt

**Files:**
- Modify: `apps/webapp/backend/main.py:290-360` (`process_video`)
- Modify: `scripts/debug/audit_pipeline.py` (obsługa treków)
- Test: `tests/test_backend/test_sessions_api.py` (rozszerzenie)

**Interfaces:**
- Consumes: `process_video_for_dataset(...) -> {"tracks", "rejected_tracks", "total_frames"}` (Task 6).
- Produces: `POST /api/process_video` zwraca `{"dogs": [{"track_id": int, "neutral_frame_idx": int,
  "peak_frames": [...]}], "total_frames": int}`.

- [ ] **Step 1: Napisz test odpowiedzi z listą psów**

```python
    async def test_klatka_sesji_niesie_track_id(self, client) -> None:
        """Test: każda klatka wie, do którego psa należy (wideo może mieć kilka)."""
        resp = await client.get(f"/api/sessions/{SESSION_ID}/frames")
        assert resp.status_code == 200

        frames = resp.json()["frames"]
        assert frames, "sesja testowa musi mieć co najmniej jedną klatkę"
        assert "track_id" in frames[0], "klatka musi wskazywać psa, którego dotyczy"
```

- [ ] **Step 2: Uruchom test i potwierdź czerwony**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backend/test_sessions_api.py -q -k track_id`
Expected: FAIL — `AssertionError: klatka musi wskazywać psa, którego dotyczy`
(pole `track_id` nie istnieje jeszcze w `FrameAnnotation`)

- [ ] **Step 3: Zaktualizuj `main.py`**

W `process_video` zamień odczyt `result["peak_frames"]` na iterację po `result["tracks"]`
i zbuduj odpowiedź:

```python
    dogs = [
        {
            "track_id": track.track_id,
            "neutral_frame_idx": track.neutral_frame_idx,
            "peak_frames": [
                _serialize_track_frame(frame)
                for frame in track.frames
                if frame.frame_idx in track.peak_indices
            ],
        }
        for track in result["tracks"]
    ]
    return {"dogs": dogs, "total_frames": result["total_frames"]}
```

`SessionStore` zapisuje `track_id` w każdej `FrameAnnotation` — dodaj pole do dataclassy
z wartością domyślną `None`, żeby stare sesje wczytywały się bez błędu.

- [ ] **Step 4: Zaktualizuj `audit_pipeline.py` na treki**

W `audit_video` zamień odczyt `all_frames_data` i `peak_frames` na iterację po
`result["tracks"]` (klatki treku) i `result["rejected_tracks"]` (powody odrzucenia).
Dodaj do raportu sekcję:

```python
        "etap_0_treki": {
            "przyjete": stats.tracks_accepted,
            "odrzucone": stats.tracks_rejected,
            "powody": dict(stats.track_reject_reasons.most_common(5)),
        },
```

- [ ] **Step 5: Uruchom ponowny audyt na tych samych 40 wideo**

```bash
.venv/Scripts/python.exe -u scripts/debug/audit_pipeline.py --limit 40 --max-frames 20 --fps 5 --output data/audit_after.json
```

Porównaj z `data/audit_pipeline.json` (stan przed zmianami). Kryteria akceptacji ze spec:
- σ ratio AU (`podloga_szumu_std_ratio`) **spada** względem 0.35–0.76;
- udział klatek przechodzących filtr pozy **rośnie** względem 24.2%;
- liczba wideo bez wyniku **spada** względem 7/40.

Zapisz porównanie w `docs/sprints/14-batch-annotation/AUDYT.md` jako tabelę przed/po.

- [ ] **Step 6: Uruchom pełne testy, linter i zacommituj**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
git add -A
git commit -m "[SPRINT-14][TASK] Webapp i audyt na wiele psów + pomiar efektu zmian

Sesja wystawia listę psów z własnymi track_id (front ma już DogSelector).
Audyt raportuje treki przyjęte/odrzucone z powodami i mierzy efekt: szum AU,
udział klatek przez filtr pozy, liczba wideo bez wyniku.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Po wykonaniu planu

Scal gałąź zgodnie z workflow projektu:

```bash
git checkout develop && git merge --no-ff feature/pipeline-audit
.venv/Scripts/python.exe -m pytest -q
git checkout main && git merge --no-ff develop
```

Następnie zaktualizuj `docs/sprints/14-batch-annotation/SPRINT.md` o wyniki audytu
przed/po i przejdź do Sprintu 15 (import COCO do webapp + eksport CSV) — to on
odblokowuje ręczną weryfikację, bez której sieć AU (Sprint 16) nie ma na czym się uczyć.
