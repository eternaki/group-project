# Design: Qualitatywny neutralny baseline dla delta-AU (data-free)

> Data: 2026-06-18. Branch: `anton-eventyr/continue-au-improvement`.
> Kontekst: kontynuacja "wariantu A" ulepszeń AU (patrz `docs/SESSION_HANDOFF.md` §5–6).

## Cel

Uczynić bazę neutralną dla delta-AU **odporną i uczciwą**, aby pre-fill AU
(formuła keypoints → 21 AU) był wyraźnie bardziej wiarygodny. To fundament pod
przyszły zbiór danych (wariant C): lepszy pre-fill = lepsze etykiety bootstrapu.

**Bez etykiet (data-free).** Per-AU progi wymagają kalibracji na danych — odłożone
do wariantu C. Tu poprawiamy wyłącznie to, co da się poprawić bez zbioru etykiet.

## Zakres

**W zakresie:**
1. Normalizacja stability score (niezależność od skali/translacji twarzy).
2. AU-istotne krytyczne keypoints (rygor widoczności ust/warg u kandydata neutral).
3. Okno stabilności po oryginalnym timeline (nie po "spłaszczonej" liście valid).
4. Median-baseline keypoints + lekka heurystyka wyboru "typowej" konfiguracji (#4).

**Poza zakresem (świadomie):**
- Per-AU progi aktywacji (wymagają kalibracji → wariant C).
- Median-baseline NIE zmienia interfejsu `DeltaActionUnitsExtractor`
  (nadal przyjmuje jeden wektor 138).

## Punkt integracji

`packages/pipeline/inference.py` — obecnie `neutral_keypoints = keypoints_list[neutral_idx]`
(jeden klatka). Zastępujemy wywołaniem nowej funkcji `compute_neutral_baseline(...)`,
która zwraca medianowy wektor 138. `DeltaActionUnitsExtractor` bez zmian.

## Zmiany w `packages/pipeline/neutral_frame.py`

### 1. Normalizacja stability score (#1)
W `_compute_stability_score`: przed `np.var` znormalizować współrzędne każdej klatki —
odjąć punkt środkowy oczu (centrowanie, usuwa translację twarzy w kadrze) i podzielić
przez eye-distance tej klatki (usuwa skalę: blisko/daleko). Wtedy score mierzy zmianę
**kształtu twarzy**, a nie ruch całej głowy ani wielkość. Eliminuje systematyczne
faworyzowanie klatek, gdzie pies jest mały/daleko (tam keypoints są najmniej dokładne).

### 2. AU-istotne krytyczne keypoints (#2)
Rozszerzyć `_CRITICAL_KP_INDICES` o kąciki ust i centra warg (górna/dolna).
Kandydat na neutralną MUSI mieć widoczne punkty ust — inaczej baseline ust jest śmieciowy,
a wszystkie mouth-AU (AU12/25/26/27, AD33/35, AD37/137) liczone od śmiecia.

### 3. Okno po oryginalnym timeline (#3)
`detect_auto` dostaje spłaszczoną listę valid (None usunięte) → indeksy w oknie ±window
mogą być odległe w czasie. Okno stabilności i zbiór do mediany liczyć po oryginalnych
indeksach klatek (±window), biorąc wewnątrz tylko klatki nie-`None` i frontalne.

### 4. Median-baseline + heurystyka wyboru (#4)
Nowa funkcja `compute_neutral_baseline(keypoints_list, neutral_idx, window_size)`:
- per-keypoint **mediana x,y** po valid+frontalnych klatkach okna wokół `neutral_idx`,
- kanał visibility = mediana widoczności; jeśli punkt rzadko widoczny → fallback na
  wartość z klatki `neutral_idx`,
- gasi szum lokalizacji keypoints (±piksele) bez "uciekania" w inną pozę.

Łagodzenie #4 (stabilność ≠ neutralność): wśród top-stabilnych kandydatów preferować
tego najbliższego **globalnej medianie konfiguracji** po wszystkich kandydatach
(założenie: "typowe = rozluźnione"). To heurystyka — jawnie udokumentowana, nie twardy fakt.

## Testy

- Normalizacja: stability score niezmienniczy względem przeskalowania i przesunięcia
  współrzędnych (ten sam kształt → ten sam score).
- Krytyczne keypoints: kandydat ze śmieciowymi ustami odrzucony mimo dobrych oczu/nosa.
- Median-baseline: pojedyncza klatka-outlier w oknie nie psuje bazy (mediana odporna).
- Okno timeline: poprawne grupowanie przy lukach detekcji (None w środku).
- Regresja: zielone istniejące `tests/test_pipeline/test_neutral_frame.py`,
  `tests/test_models/test_delta_action_units.py`.

## Znane ograniczenia (do udokumentowania)

- #4 pozostaje heurystyką 2D — nie ma czysto data-free rozwiązania na "neutralność wyrazu".
- EAD104 (rotacja uszu) — ograniczenie 2D (bez zmian, już udokumentowane).
- Per-AU progi — odłożone do wariantu C (kalibracja na zebranym zbiorze).

## Ryzyka

- Median-baseline może zamaskować subtelny baseline, jeśli okno obejmie zmianę wyrazu —
  mitygowane przez filtr frontalności + ograniczone okno wokół najstabilniejszej klatki.
- Rozszerzony rygor krytycznych keypoints może odrzucić więcej kandydatów → fallback
  (`_find_relaxed_candidates`, last-resort) musi pozostać sprawny; testy to pokrywają.
