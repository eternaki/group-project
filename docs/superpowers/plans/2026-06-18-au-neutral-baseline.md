# AU Neutral Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Uczynić neutralny baseline dla delta-AU odpornym i uczciwym (data-free), aby pre-fill AU był wiarygodniejszy.

**Architecture:** Poprawiamy wybór klatki neutralnej w `packages/pipeline/neutral_frame.py` (normalizacja stabilności, AU-istotne keypoints, okno po timeline, wybór "typowej" konfiguracji) oraz dodajemy medianowy baseline; wpinamy go w `packages/pipeline/inference.py`. `DeltaActionUnitsExtractor` pozostaje bez zmian (przyjmuje jeden wektor 138).

**Tech Stack:** Python 3.12, NumPy, pytest, ruff. Keypoints: 46×3 (DogFLW), `packages.data.schemas.KP`.

## Global Constraints

- Type hints zawsze; format `ruff` (PEP 8); f-strings. Docstringi publicznego API po polsku.
- Funkcja ≤50 linii, złożoność ≤10, max 5-6 parametrów. Bez mutable default args, bez bare `except`, bez magic numbers (stałe nazwane).
- Bez nowych zależności. Data-free (żadnych etykiet). Per-AU progi POZA zakresem.
- `NUM_KEYPOINTS = 46`; format keypoints `[x0,y0,v0,...]` (138 wartości).
- Commit messages po polsku z prefiksem `[SPRINT-12][TASK]` i stopką
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Branch roboczy: `anton-eventyr/continue-au-improvement` (nie commituj do main/develop).
- Po każdym tasku: `pytest -q` zielone + `ruff check .` czyste.

---

## File Structure

- `packages/pipeline/neutral_frame.py` — MODIFY: normalizacja stabilności, krytyczne keypoints, okno timeline, wybór typowej klatki, nowa funkcja `compute_neutral_baseline`.
- `packages/pipeline/inference.py` — MODIFY (linia ~772): użyć `compute_neutral_baseline` zamiast pojedynczej klatki; przekazać `valid_frame_indices` do `detect_auto`.
- `tests/test_pipeline/test_neutral_frame.py` — MODIFY: dodać testy normalizacji, krytycznych keypoints, okna timeline, typowej klatki, baseline.

Reużywamy istniejący helper `make_frontal_kp()` z `tests/test_pipeline/test_neutral_frame.py`.

---

### Task 1: Normalizacja stability score (skala + translacja)

**Files:**
- Modify: `packages/pipeline/neutral_frame.py` (`_compute_stability_score`, dodać helper `_normalize_shape`, stałą `VARIANCE_SCALE`)
- Test: `tests/test_pipeline/test_neutral_frame.py`

**Interfaces:**
- Consumes: `KP.LEFT_EYE_INNER/OUTER`, `KP.RIGHT_EYE_INNER/OUTER`, `NUM_KEYPOINTS`.
- Produces: `_normalize_shape(coords: np.ndarray) -> np.ndarray` (centruje na mid-eye, dzieli przez eye-distance); `_compute_stability_score` liczy wariancję na znormalizowanych współrzędnych.

- [ ] **Step 1: Write the failing test**

W klasie `TestNeutralFrameDetector` w `tests/test_pipeline/test_neutral_frame.py`:

```python
    def test_stability_score_is_scale_and_translation_invariant(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: stability score niezmienniczy względem skali i translacji twarzy."""
        rng = np.random.default_rng(7)
        base = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        seq = [
            (base + np.column_stack([
                rng.random((NUM_KEYPOINTS, 2)) * 0.8,
                np.zeros((NUM_KEYPOINTS, 1)),
            ]))
            for _ in range(12)
        ]
        seq_raw = [f.flatten() for f in seq]
        # Ta sama sekwencja przeskalowana x2.5 i przesunięta o +400px (x,y)
        seq_scaled = []
        for f in seq:
            g = f.copy()
            g[:, :2] = g[:, :2] * 2.5 + 400.0
            seq_scaled.append(g.flatten())

        score_raw = detector._compute_stability_score(seq_raw, center_idx=6)
        score_scaled = detector._compute_stability_score(seq_scaled, center_idx=6)

        assert abs(score_raw - score_scaled) < 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_neutral_frame.py::TestNeutralFrameDetector::test_stability_score_is_scale_and_translation_invariant -v`
Expected: FAIL (obecnie wariancja w surowych pikselach → score różny po skali).

- [ ] **Step 3: Write minimal implementation**

W `neutral_frame.py` dodać stałą blisko góry modułu (po importach):

```python
# Skala wariancji w jednostkach znormalizowanych (po podziale przez eye-distance).
# Dobrana tak, by ~0.05 (5% odległości oczu) jitteru dawało wyraźnie niższy score.
VARIANCE_SCALE: float = 1000.0
```

Dodać helper w sekcji funkcji pomocniczych:

```python
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
```

Dodać helper odległości (jeśli go nie ma w module) obok `_normalize_shape`:

```python
def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Odległość euklidesowa między dwoma punktami."""
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))
```

Zmodyfikować `_compute_stability_score` — zamiast surowych współrzędnych użyć znormalizowanych i `VARIANCE_SCALE`:

```python
        window_coords = [
            _normalize_shape(kp.reshape(NUM_KEYPOINTS, 3)[:, :2])
            for kp in keypoints_list[start:end]
            if kp is not None
        ]

        if len(window_coords) < 2:
            return 0.0

        coords_array = np.array(window_coords)   # (window, 46, 2)
        mean_variance = float(np.mean(np.var(coords_array, axis=0)))
        return 1.0 / (1.0 + mean_variance * VARIANCE_SCALE)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_neutral_frame.py -v`
Expected: PASS — nowy test zielony ORAZ istniejące `test_stability_score_stable_sequence` (>0.8) i `test_stability_score_unstable_sequence` (<0.5) nadal zielone.

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/neutral_frame.py tests/test_pipeline/test_neutral_frame.py
git commit -m "[SPRINT-12][TASK] Neutral frame: normalizacja stability (skala+translacja)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Okno stabilności po oryginalnym timeline

**Files:**
- Modify: `packages/pipeline/neutral_frame.py` (`detect_auto`, `_compute_stability_score`)
- Test: `tests/test_pipeline/test_neutral_frame.py`

**Interfaces:**
- Consumes: `_compute_stability_score` z Task 1.
- Produces: `_compute_stability_score(self, keypoints_list, center_idx, frame_indices=None)` — gdy `frame_indices` podane, okno wybiera klatki o oryginalnym indeksie w zasięgu ±window_size//2 od `frame_indices[center_idx]`. `detect_auto(..., frame_indices: Optional[list[int]] = None)`.

- [ ] **Step 1: Write the failing test**

```python
    def test_stability_window_uses_original_timeline(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: z frame_indices okno liczy sąsiadów po realnym czasie, nie po pozycji."""
        base = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        # 3 stabilne klatki, ale w realnym czasie odległe o 100 klatek od siebie.
        seq = [base.flatten() for _ in range(3)]
        far_indices = [0, 100, 200]

        # Z frame_indices: sąsiedzi poza oknem (window 10) → brak sąsiadów → 0.0
        score_far = detector._compute_stability_score(
            seq, center_idx=1, frame_indices=far_indices
        )
        # Bez frame_indices: sąsiedztwo pozycyjne → liczy wariancję (stabilne → wysoki)
        score_positional = detector._compute_stability_score(seq, center_idx=1)

        assert score_far == 0.0
        assert score_positional > 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_neutral_frame.py::TestNeutralFrameDetector::test_stability_window_uses_original_timeline -v`
Expected: FAIL (parametr `frame_indices` jeszcze nie istnieje → TypeError).

- [ ] **Step 3: Write minimal implementation**

Zmodyfikować sygnaturę i logikę okna w `_compute_stability_score`:

```python
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

        coords_array = np.array(window_coords)
        mean_variance = float(np.mean(np.var(coords_array, axis=0)))
        return 1.0 / (1.0 + mean_variance * VARIANCE_SCALE)
```

Dodać `frame_indices` do `detect_auto` i przekazać przy liczeniu scores:

```python
    def detect_auto(
        self,
        frames: list[np.ndarray],
        keypoints_list: list[Optional[np.ndarray]],
        head_poses: Optional[list[Optional[HeadPose]]] = None,
        debug: bool = False,
        frame_indices: Optional[list[int]] = None,
    ) -> int:
```

W bloku liczenia scores (wewnątrz `detect_auto`, gdzie jest `scores = [...]`):

```python
        scores = [
            (idx, self._compute_stability_score(keypoints_list, idx, frame_indices))
            for idx in candidates
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_neutral_frame.py -v`
Expected: PASS (nowy + wszystkie poprzednie).

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/neutral_frame.py tests/test_pipeline/test_neutral_frame.py
git commit -m "[SPRINT-12][TASK] Neutral frame: okno stabilności po realnym timeline

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: AU-istotne krytyczne keypoints (rygor ust)

**Files:**
- Modify: `packages/pipeline/neutral_frame.py` (`_CRITICAL_KP_INDICES`)
- Test: `tests/test_pipeline/test_neutral_frame.py`

**Interfaces:**
- Consumes: `KP.MOUTH_LEFT_CORNER` (39), `KP.MOUTH_RIGHT_CORNER` (40), `KP.UPPER_LIP_CENTER` (38), `KP.LOWER_LIP_CENTER` (41).
- Produces: rozszerzona lista `_CRITICAL_KP_INDICES` używana przez `_critical_keypoints_visible` i `_count_visible_critical_kps`.

- [ ] **Step 1: Write the failing test**

```python
    def test_garbage_mouth_frame_is_not_valid_candidate(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: dobre oczy/nos, ale niewidoczne usta → NIE kandydat (baseline ust = śmieć)."""
        kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        for idx in (KP.MOUTH_LEFT_CORNER, KP.MOUTH_RIGHT_CORNER,
                    KP.UPPER_LIP_CENTER, KP.LOWER_LIP_CENTER):
            kp[idx, 2] = 0.05
        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        assert detector._is_valid_candidate(kp_flat, pose) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_neutral_frame.py::TestNeutralFrameDetector::test_garbage_mouth_frame_is_not_valid_candidate -v`
Expected: FAIL (obecnie usta nie są krytyczne → klatka uznana za kandydata).

- [ ] **Step 3: Write minimal implementation**

Rozszerzyć `_CRITICAL_KP_INDICES`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_neutral_frame.py -v`
Expected: PASS — nowy zielony; `test_frontal_frame_is_valid_candidate` nadal zielony (usta w `make_frontal_kp` mają vis 0.9).

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/neutral_frame.py tests/test_pipeline/test_neutral_frame.py
git commit -m "[SPRINT-12][TASK] Neutral frame: usta/wargi jako krytyczne keypoints

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Medianowy baseline keypoints (`compute_neutral_baseline`)

**Files:**
- Modify: `packages/pipeline/neutral_frame.py` (nowa publiczna funkcja + stałe)
- Test: `tests/test_pipeline/test_neutral_frame.py`

**Interfaces:**
- Consumes: `HeadPose`, `_is_frontal_pose`, `NUM_KEYPOINTS`, `KP`.
- Produces:
  ```python
  def compute_neutral_baseline(
      keypoints_list: list[Optional[np.ndarray]],
      neutral_idx: int,
      head_poses: list[Optional[HeadPose]],
      window_size: int = 10,
      max_yaw: float = 35.0,
      max_pitch: float = 40.0,
  ) -> np.ndarray
  ```
  Zwraca wektor (138,): per-keypoint mediana x,y po valid+frontalnych klatkach okna
  (±window_size//2 wokół neutral_idx, po realnym indeksie). Kanał visibility = mediana;
  punkt bez próbek widocznych (vis ≥ `BASELINE_VIS_THRESHOLD`) → wartość z klatki neutral_idx.

- [ ] **Step 1: Write the failing test**

```python
    def test_compute_neutral_baseline_is_robust_to_outlier_frame(self) -> None:
        """Test: pojedyncza klatka-outlier w oknie nie psuje medianowej bazy."""
        from packages.pipeline.neutral_frame import compute_neutral_baseline

        base = make_frontal_kp()
        n = 7
        kps = [base.copy() for _ in range(n)]
        # Klatka 3 = mocny outlier (przesunięcie ust o 60px)
        outlier = base.reshape(NUM_KEYPOINTS, 3)
        outlier[KP.LOWER_LIP_CENTER, 1] += 60
        kps[3] = outlier.flatten()
        poses = [estimate_head_pose(k) for k in kps]

        baseline = compute_neutral_baseline(kps, neutral_idx=3, head_poses=poses)
        bl = baseline.reshape(NUM_KEYPOINTS, 3)

        # Mediana ignoruje outlier → dolna warga blisko wartości bazowej (228), nie 288.
        assert abs(bl[KP.LOWER_LIP_CENTER, 1] - 228.0) < 5.0
        assert baseline.shape == (NUM_KEYPOINTS * 3,)

    def test_compute_neutral_baseline_falls_back_when_keypoint_invisible(self) -> None:
        """Test: punkt niewidoczny we wszystkich klatkach okna → wartość z klatki neutral."""
        from packages.pipeline.neutral_frame import compute_neutral_baseline

        base = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        base[KP.TONGUE_TIP] = [151.0, 240.0, 0.02]  # niewidoczny wszędzie
        kps = [base.flatten() for _ in range(5)]
        poses = [estimate_head_pose(k) for k in kps]

        baseline = compute_neutral_baseline(kps, neutral_idx=2, head_poses=poses)
        bl = baseline.reshape(NUM_KEYPOINTS, 3)

        assert abs(bl[KP.TONGUE_TIP, 0] - 151.0) < 1e-3
        assert abs(bl[KP.TONGUE_TIP, 1] - 240.0) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_neutral_frame.py -k compute_neutral_baseline -v`
Expected: FAIL (ImportError: `compute_neutral_baseline` nie istnieje).

- [ ] **Step 3: Write minimal implementation**

Dodać stałe blisko `VARIANCE_SCALE`:

```python
# Próg widoczności keypointa, by wszedł do medianowej bazy.
BASELINE_VIS_THRESHOLD: float = 0.3
```

Dodać funkcję (sekcja publiczna na końcu pliku, przed lub po istniejących helperach):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_neutral_frame.py -k compute_neutral_baseline -v`
Expected: PASS (oba testy).

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/neutral_frame.py tests/test_pipeline/test_neutral_frame.py
git commit -m "[SPRINT-12][TASK] Neutral frame: medianowy baseline keypoints (okno)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Wybór "typowej" klatki wśród stabilnych (łagodzenie stabilność≠neutralność)

**Files:**
- Modify: `packages/pipeline/neutral_frame.py` (`detect_auto`, helper `_select_most_typical`, stała `TOP_STABLE_FRACTION`)
- Test: `tests/test_pipeline/test_neutral_frame.py`

**Interfaces:**
- Consumes: `_normalize_shape` (Task 1), scores z `_compute_stability_score`.
- Produces: `_select_most_typical(candidates: list[int], scores: dict[int, float], keypoints_list: list[Optional[np.ndarray]]) -> int` — wśród top-`TOP_STABLE_FRACTION` najstabilniejszych kandydatów wybiera najbliższego globalnej medianie kształtu po WSZYSTKICH kandydatach. `detect_auto` używa go zamiast `max(scores)`.

- [ ] **Step 1: Write the failing test**

```python
    def test_detect_auto_prefers_typical_expression_among_stable(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: gdy kilka klatek równie stabilnych, wybierana jest typowa (modalna) konfiguracja."""
        base = make_frontal_kp()
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(9)]
        # 8 klatek typowych (zamknięty pysk) + 1 nietypowa (szeroko otwarty pysk),
        # wszystkie pojedynczo stabilne (powtórzone identycznie w oknie nie są — więc
        # budujemy listę gdzie każda klatka ma stabilne sąsiedztwo identycznych kopii).
        typical = base
        atypical = base.reshape(NUM_KEYPOINTS, 3).copy()
        atypical[KP.LOWER_LIP_CENTER, 1] += 50
        atypical[KP.CHIN, 1] += 50
        kps = [typical.copy() for _ in range(8)] + [atypical.flatten()]

        idx = detector.detect_auto(frames, kps)

        # Wybór NIE powinien paść na nietypową klatkę (indeks 8).
        assert idx != 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_neutral_frame.py::TestNeutralFrameDetector::test_detect_auto_prefers_typical_expression_among_stable -v`
Expected: może FAIL, jeśli atypowa klatka trafi na najwyższy stability (zależnie od sąsiedztwa) — test pilnuje, by wybór bronił się przed nietypową konfiguracją.

- [ ] **Step 3: Write minimal implementation**

Dodać stałą:

```python
# Frakcja najstabilniejszych kandydatów branych pod uwagę przy wyborze "typowej" klatki.
TOP_STABLE_FRACTION: float = 0.34
```

Dodać helper:

```python
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
```

Dodać import `math` na górze pliku (jeśli brak): `import math`.

Zmienić finał `detect_auto` — zamiast `max(scores, key=...)`:

```python
        score_map = {
            idx: self._compute_stability_score(keypoints_list, idx, frame_indices)
            for idx in candidates
        }
        return _select_most_typical(candidates, score_map, keypoints_list)
```

(usuwa wcześniejszą listę `scores = [...]` i `best_idx, _ = max(...)`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline/test_neutral_frame.py -v`
Expected: PASS — nowy zielony; `test_detect_auto_returns_valid_index` i `test_single_frame_returns_zero` nadal zielone.

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/neutral_frame.py tests/test_pipeline/test_neutral_frame.py
git commit -m "[SPRINT-12][TASK] Neutral frame: wybór typowej konfiguracji wśród stabilnych

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Wpięcie w pipeline + dokumentacja ograniczeń

**Files:**
- Modify: `packages/pipeline/inference.py` (~759-772)
- Modify: `packages/pipeline/neutral_frame.py` (docstring modułu — known limitations)

**Interfaces:**
- Consumes: `compute_neutral_baseline` (Task 4), `detect_auto(..., frame_indices=...)` (Task 2).
- Produces: `neutral_keypoints` w pipeline = medianowy baseline zamiast pojedynczej klatki.

- [ ] **Step 1: Zmodyfikować wywołanie `detect_auto` (przekazać frame_indices)**

W `inference.py`, w bloku `if neutral_idx is None:` dodać `frame_indices=valid_frame_indices`:

```python
            neutral_idx = detector.detect_auto(
                frames=[frames_list[i] for i in valid_frame_indices],
                keypoints_list=valid_keypoints,
                head_poses=valid_head_poses,
                debug=False,
                frame_indices=valid_frame_indices,
            )
            neutral_idx = valid_frame_indices[neutral_idx]
            print(f"  → Auto-detected neutral frame: {neutral_idx}")
```

UWAGA: `detect_auto` zwraca indeks w przekazanej (spłaszczonej) liście; mapowanie
`neutral_idx = valid_frame_indices[neutral_idx]` na oryginalny indeks zostaje bez zmian.

- [ ] **Step 2: Zmodyfikować budowę `neutral_keypoints` (mediana zamiast jednej klatki)**

Zastąpić linię `neutral_keypoints = keypoints_list[neutral_idx]` (i zostawić walidację None tuż przed):

```python
        if keypoints_list[neutral_idx] is None:
            raise ValueError(f"Neutral frame {neutral_idx} nie ma keypoints!")

        neutral_keypoints = compute_neutral_baseline(
            keypoints_list=keypoints_list,
            neutral_idx=neutral_idx,
            head_poses=head_poses,
        )
```

Dodać import na górze bloku importów funkcji `process_video_for_dataset` (obok
istniejącego importu z `packages.pipeline.neutral_frame`):

```python
        from packages.pipeline.neutral_frame import (
            NeutralFrameDetector,
            compute_neutral_baseline,
            estimate_head_pose,
        )
```

- [ ] **Step 3: Zaktualizować docstring modułu `neutral_frame.py` o znane ograniczenia**

Dodać na końcu docstringu modułu (góra pliku):

```
Znane ograniczenia (data-free):
- Stabilność ≠ neutralność: detektor mierzy brak ruchu, nie rozluźnienie wyrazu.
  Łagodzone heurystyką "typowej konfiguracji" (_select_most_typical), ale to proxy 2D.
- Medianowy baseline zakłada ~stałą skalę twarzy w obrębie okna (frontalny, krótki
  odcinek czasu). Per-AU progi i kalibracja wyrazu — poza zakresem (wariant C).
```

- [ ] **Step 4: Pełna weryfikacja (testy + linter)**

Run: `pytest -q`
Expected: PASS — wszystkie testy zielone (≥ poprzednia liczba, bez regresji).

Run: `ruff check .`
Expected: brak błędów.

- [ ] **Step 5: Commit**

```bash
git add packages/pipeline/inference.py packages/pipeline/neutral_frame.py
git commit -m "[SPRINT-12][TASK] Pipeline: medianowy baseline neutralny dla delta-AU

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Spec #1 (normalizacja stability) → Task 1. ✅
- Spec #2 (AU-istotne krytyczne keypoints) → Task 3. ✅
- Spec #3 (okno po timeline) → Task 2. ✅
- Spec #4 (median-baseline + heurystyka typowej klatki) → Task 4 (baseline) + Task 5 (heurystyka). ✅
- Spec: punkt integracji `inference.py` → Task 6. ✅
- Spec: dokumentacja ograniczeń → Task 6 Step 3. ✅
- Spec: per-AU progi poza zakresem → nie ma tasku (poprawnie). ✅

**Placeholder scan:** brak TBD/TODO; każdy krok ma konkretny kod i komendy. ✅

**Type consistency:** `_compute_stability_score(keypoints_list, center_idx, frame_indices=None)` spójne w Task 1/2/5; `compute_neutral_baseline(...)` sygnatura spójna Task 4↔Task 6; `_select_most_typical(candidates, scores, keypoints_list)` używa `score_map` (dict) zgodnie z definicją w Task 5; `_normalize_shape`/`_dist` użyte w Task 1/5 zdefiniowane w Task 1. ✅
