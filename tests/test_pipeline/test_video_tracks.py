"""
Testy przetwarzania wideo na treki psów (`process_video_for_dataset`).

Testy nie wymagają wag modeli — detektor psa i model keypoints są podmienione
na atrapy o znanej geometrii. Dzięki temu sprawdzamy to, co naprawdę jest tu
trudne: rozdzielenie psów na treki, własną bazę AU każdego psa, dotarcie KAŻDEGO
progu z konfiguracji do miejsca decyzji i wygładzanie drgań punktów.

Trek bywa DZIURAWY (pies wychodzi z kadru), więc pozycja w treku nie równa się
numerowi klatki wideo. Testy na treku ciągłym tej pomyłki nie wychwytują —
dlatego atrapa detektora umie chować psa na wybranych klatkach.
"""

from typing import Optional

import numpy as np
import pytest

from packages.data.schemas import KP, NUM_KEYPOINTS, Keypoint
from packages.models.bbox import Detection
from packages.models.keypoints import KeypointsPrediction
from packages.pipeline.inference import (
    InferencePipeline,
    PipelineConfig,
    VideoDatasetConfig,
)
from packages.pipeline.landmark_smoothing import KeypointSmoother
from packages.pipeline.quality_gate import QualityThresholds
from packages.pipeline.track_processing import (
    NEUTRAL_SOURCE_AUTO,
    NEUTRAL_SOURCE_MANUAL,
    TrackQuality,
)
from tests.test_pipeline.kp_fixtures import make_frontal_kp

# Bramka jakości otwarta na oścież — do testów, które sprawdzają INNY próg
# i nie chcą, żeby odsiew kadru zamazał badany efekt. Każdy próg trzeba wymienić
# wprost: kryterium pominięte dziedziczy wartość domyślną, więc dodanie nowej
# miary do `QualityThresholds` po cichu zamyka „otwartą" bramkę i psuje testy
# w miejscu, które z nową miarą nie ma nic wspólnego.
_OPEN_GATE = QualityThresholds(
    max_asymmetry=1.0,
    max_weak_ratio=1.0,
    min_face_width=0.0,
    max_shape_distance=float("inf"),
)

# Psy w kadrze: lewy mniejszy, prawy większy (różny rozmiar cropu = różna geometria
# mordy w atrapie modelu keypoints — patrz `_FakeKeypointsModel`)
LEFT_DOG_BOX: tuple[int, int, int, int] = (50, 100, 200, 200)
RIGHT_DOG_BOX: tuple[int, int, int, int] = (360, 100, 260, 260)

FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
DEFAULT_FRAMES: int = 6

# Próg szerokości cropu rozdzielający geometrię obu psów w atrapie
_WIDE_CROP_PX: int = 250
# Średnia widoczność punktów w fixturze (make_frontal_kp) jest w okolicy 0.93
_ABOVE_FIXTURE_CONF: float = 0.99
# Otwarcie pyska na WSKAZANYCH klatkach. Selektor peaków przyjmuje wyłącznie kadry
# powyżej progu TFM, więc pies bez ANI JEDNEJ zmiany mimiki nie ma peaków — i tak
# ma być. Testy peaków muszą więc dać psu realny ruch, a nie polegać na dobieraniu
# kadrów spod progu (mechanizm usunięty po audycie: dawał 1 peak na 39).
# Klatka 0 zostaje bez mimiki, żeby detektor klatki neutralnej miał co wybrać —
# gdyby bazą została morda z otwartym pyskiem, wszystkie delty wyszłyby ujemne
# i TFM (suma DODATNICH delt) byłby zerowy w całym treku.
MOUTH_OPEN_PX: float = 25.0


def _frame_with_two_dogs() -> np.ndarray:
    """Klatka z dwoma wyraźnie różnymi kolorystycznie psami."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    for box, color in ((LEFT_DOG_BOX, (200, 30, 30)), (RIGHT_DOG_BOX, (30, 200, 30))):
        x, y, w, h = box
        frame[y : y + h, x : x + w] = color
    return frame


class _FakeBBoxModel:
    """
    Atrapa detektora psów — stałe boksy, z możliwością zniknięcia psa.

    `gaps` mapuje indeks psa na numery klatek, w których go nie widać. Dzięki temu
    powstaje trek dziurawy, w którym pozycja w treku różni się od numeru klatki.
    """

    def __init__(
        self,
        boxes: list[tuple[int, int, int, int]],
        gaps: Optional[dict[int, set[int]]] = None,
    ) -> None:
        self.boxes = boxes
        self.gaps = gaps or {}
        self.frame_idx = -1

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Zwraca detekcje psów widocznych w bieżącej klatce."""
        self.frame_idx += 1
        return [
            Detection(bbox=box, confidence=0.9, class_id=0, class_name="dog")
            for dog, box in enumerate(self.boxes)
            if self.frame_idx not in self.gaps.get(dog, set())
        ]


class _FakeKeypointsModel:
    """
    Atrapa modelu keypoints — stały układ punktów w układzie cropu.

    Geometria zależy od szerokości cropu: pies z większym boksem dostaje opuszczoną
    żuchwę. Dzięki temu widać, czy każdy trek liczy AU względem WŁASNEJ klatki
    neutralnej — przy wspólnej bazie ratio odjechałoby od 1.0.

    Dwa niezależne drgania:
    - `jitter` rusza czubkiem nosa (punkt wewnętrzny) — drganie pomiaru przy
      nieruchomej mordzie; drganie WSZYSTKICH punktów skasowałoby się przy
      normalizacji boksem mordy i niczego by nie sprawdzało,
    - `scale_jitter` rusza czubkiem prawego ucha, czyli punktem SKRAJNYM otoczki —
      zmienia ROZMIAR boksu mordy odtwarzanego z punktów, czyli skalę normalizacji.

    `expression_frames` numeruje WYWOŁANIA atrapy, nie klatki wideo — przy psie
    znikającym z kadru te numeracje się rozjeżdżają.
    """

    def __init__(
        self,
        jitter: float = 0.0,
        scale_jitter: float = 0.0,
        nose_shift: float = 0.0,
        eye_tilt: float = 0.0,
        expression_frames: Optional[set[int]] = None,
    ) -> None:
        self.jitter = jitter
        self.scale_jitter = scale_jitter
        self.nose_shift = nose_shift
        self.eye_tilt = eye_tilt
        self.expression_frames = expression_frames or set()
        self.calls = 0

    def predict(self, crop: np.ndarray) -> KeypointsPrediction:
        """Zwraca keypoints w układzie cropu."""
        keypoints = make_frontal_kp().reshape(NUM_KEYPOINTS, 3).copy()
        if crop.shape[1] >= _WIDE_CROP_PX:
            keypoints[KP.LOWER_LIP_CENTER, 1] += 20.0
            keypoints[KP.CHIN, 1] += 20.0

        if self.calls in self.expression_frames:
            # Czubek języka idzie z żuchwą. Bez niego otwarcie pyska rozjeżdżało
            # geometrię dolnej części mordy: język zostawał na miejscu, przez co
            # klatka z ekspresją wypadała detektorowi klatki neutralnej
            # STABILNIEJ niż klatka spokojna i lądowała jako baza AU.
            for point in (KP.LOWER_LIP_CENTER, KP.CHIN, KP.JAW_CENTER, KP.TONGUE_TIP):
                keypoints[point, 1] += MOUTH_OPEN_PX

        sign = 1.0 if self.calls % 2 else -1.0
        keypoints[KP.NOSE_TIP, 0] += self.jitter * sign + self.nose_shift
        keypoints[KP.RIGHT_EAR_TIP, 0] += self.scale_jitter * sign
        keypoints[KP.RIGHT_EYE_INNER, 1] += self.eye_tilt
        keypoints[KP.RIGHT_EYE_OUTER, 1] += self.eye_tilt
        self.calls += 1

        return KeypointsPrediction(
            keypoints=[
                Keypoint(x=float(x), y=float(y), visibility=float(v))
                for x, y, v in keypoints
            ],
            confidence=float(np.mean(keypoints[:, 2])),
            num_detected=NUM_KEYPOINTS,
        )


def _pipeline(
    boxes: Optional[list[tuple[int, int, int, int]]] = None,
    gaps: Optional[dict[int, set[int]]] = None,
    **keypoint_options,
) -> InferencePipeline:
    """Buduje pipeline z atrapami modeli (bez wag, bez detektora mordy)."""
    pipeline = InferencePipeline(
        PipelineConfig(device="cpu", keypoints_two_pass=False, use_face_detector=False)
    )
    default_boxes = [LEFT_DOG_BOX, RIGHT_DOG_BOX]
    pipeline.bbox_model = _FakeBBoxModel(default_boxes if boxes is None else boxes, gaps)
    pipeline.keypoints_model = _FakeKeypointsModel(**keypoint_options)
    pipeline._models_loaded = True
    return pipeline


def _config(**overrides) -> VideoDatasetConfig:
    """Konfiguracja testowa — filtr ostrości wyłączony (klatki są syntetyczne)."""
    settings: dict = {"num_peaks": 3, "min_sharpness": 0.0}
    settings.update(overrides)
    return VideoDatasetConfig(**settings)


def _run(
    pipeline: InferencePipeline,
    frames: int = DEFAULT_FRAMES,
    **overrides,
) -> dict:
    """Uruchamia pipeline na zadanej liczbie identycznych klatek."""
    return pipeline.process_video_for_dataset(
        [_frame_with_two_dogs() for _ in range(frames)],
        config=_config(**overrides),
    )


def _spy_on_smoother(monkeypatch) -> list[dict]:
    """Podstawia filtr zapisujący swoje wywołania (znacznik czasu i boks)."""
    calls: list[dict] = []

    class _SpySmoother(KeypointSmoother):
        def smooth(self, keypoints_flat, face_box, timestamp):
            calls.append({"box": tuple(face_box), "timestamp": timestamp})
            return super().smooth(keypoints_flat, face_box, timestamp)

    monkeypatch.setattr("packages.pipeline.inference.KeypointSmoother", _SpySmoother)
    return calls


def _nose_x(track) -> list[float]:
    """Współrzędne x czubka nosa w kolejnych klatkach treku."""
    return [
        float(frame.keypoints.reshape(NUM_KEYPOINTS, 3)[KP.NOSE_TIP, 0])
        for frame in track.frames
    ]


class TestStrukturaWyniku:
    """Kontrakt zwracanej struktury."""

    def test_zwraca_treki_zamiast_jednego_psa(self) -> None:
        result = _run(_pipeline())

        assert set(result) == {"tracks", "rejected_tracks", "total_frames"}
        assert result["total_frames"] == DEFAULT_FRAMES

    def test_dwa_psy_daja_dwa_osobne_treki(self) -> None:
        result = _run(_pipeline())

        track_ids = {track.track_id for track in result["tracks"]}
        assert len(result["tracks"]) == 2
        assert len(track_ids) == 2

    def test_klatka_treku_niesie_oba_boksy(self) -> None:
        """Boks psa (do zbioru i pod rasę) i boks mordy (pod keypoints) to co innego."""
        result = _run(_pipeline(boxes=[LEFT_DOG_BOX]))

        for frame in result["tracks"][0].frames:
            assert frame.body_box == LEFT_DOG_BOX
            assert frame.face_box != frame.body_box


class TestDziurawyTrek:
    """Pies znika z kadru i wraca — pozycja w treku ≠ numer klatki wideo."""

    GAPS: dict[int, set[int]] = {0: {2, 3}}
    FRAMES: int = 8
    VISIBLE: set[int] = {0, 1, 4, 5, 6, 7}

    def _track(self, **overrides):
        result = _run(
            _pipeline(
                boxes=[LEFT_DOG_BOX],
                gaps=self.GAPS,
                # Numeracja WYWOŁAŃ atrapy, nie klatek wideo: pies znika na
                # klatkach 2-3, więc wywołania 2, 3, 5 to klatki 4, 5, 7.
                expression_frames={2, 3, 5},
            ),
            frames=self.FRAMES,
            **overrides,
        )
        assert len(result["tracks"]) == 1, "przerwa 2 klatek nie może rozerwać treku"
        return result["tracks"][0]

    def test_trek_pomija_klatki_bez_psa(self) -> None:
        track = self._track()

        assert {frame.frame_idx for frame in track.frames} == self.VISIBLE

    def test_klatka_neutralna_to_numer_klatki_wideo(self) -> None:
        """Zwrócenie POZYCJI zamiast numeru klatki wskazałoby klatkę bez psa."""
        track = self._track()

        assert track.neutral_frame_idx in self.VISIBLE

    def test_peaki_to_numery_klatek_wideo(self) -> None:
        track = self._track(min_peak_separation_s=0.2)

        assert track.peak_indices, "trek musi mieć peaki, inaczej test nic nie sprawdza"
        assert set(track.peak_indices) <= self.VISIBLE

    def test_recznie_wskazana_klatka_spoza_treku_nie_jest_podmieniana_po_cichu(self) -> None:
        """Pies, którego wtedy nie było w kadrze, dostaje auto — i to widać w wyniku."""
        track = self._track(neutral_frame_idx=2)

        assert track.neutral_frame_idx != 2
        assert track.neutral_source == NEUTRAL_SOURCE_AUTO


class TestOsobnaBazaKazdegoPsa:
    """Każdy pies ma własną klatkę neutralną — bazy nie wolno współdzielić."""

    def test_ratio_au_wynosi_jeden_dla_nieruchomego_psa(self) -> None:
        result = _run(_pipeline())
        assert result["tracks"], "oba psy powinny dać godne treki"

        for track in result["tracks"]:
            for frame in track.frames:
                for au in frame.delta_aus.values():
                    assert au.ratio == pytest.approx(1.0, abs=1e-6), (
                        f"AU {au.name} treku {track.track_id} liczone względem obcej bazy"
                    )

    def test_kazdy_trek_ma_klatke_neutralna_ze_swoich_klatek(self) -> None:
        result = _run(_pipeline())

        for track in result["tracks"]:
            assert track.neutral_frame_idx in {frame.frame_idx for frame in track.frames}


class TestProgiDocierajaDoDecyzji:
    """Każdy próg z konfiguracji musi zmieniać wynik — inaczej jest martwy."""

    def test_fps_wyznacza_znaczniki_czasu_wygladzania(self, monkeypatch) -> None:
        """Stała zamiast `config.fps` przeszłaby resztę zestawu niezauważona."""
        calls = _spy_on_smoother(monkeypatch)

        _run(_pipeline(boxes=[LEFT_DOG_BOX]), frames=4, fps=2.0)

        assert [call["timestamp"] for call in calls] == [0.0, 0.5, 1.0, 1.5]

    def test_odstep_peakow_jest_twardy(self) -> None:
        """Selektor woli oddać mniej peaków niż dosypać sąsiadujące klatki."""
        track = _run(
            _pipeline(boxes=[LEFT_DOG_BOX], expression_frames={1, 2, 3, 7}),
            frames=8,
            fps=5.0,
            min_peak_separation_s=1.0,
            num_peaks=8,
        )["tracks"][0]

        gaps = [
            abs(a - b)
            for i, a in enumerate(track.peak_indices)
            for b in track.peak_indices[i + 1 :]
        ]
        assert track.peak_indices
        assert all(gap >= 5 for gap in gaps), f"peaki za blisko siebie: {track.peak_indices}"

    def test_luzniejszy_odstep_daje_wiecej_peakow(self) -> None:
        def peaks(separation_s: float) -> int:
            result = _run(
                # Mimika na kilku SĄSIEDNICH klatkach: przy luźnej separacji
                # selektor weźmie kilka, przy twardej tylko najsilniejszą.
                _pipeline(boxes=[LEFT_DOG_BOX], expression_frames={1, 2, 3, 4, 7}),
                frames=8,
                num_peaks=8,
                min_peak_separation_s=separation_s,
            )
            return len(result["tracks"][0].peak_indices)

        assert peaks(0.2) > peaks(1.0)

    def test_prog_obrotu_glowy_dociera_do_wyboru_peakow(self) -> None:
        """
        Nos przesunięty w bok = morda w profilu; przy domyślnych progach odpada.

        Obrócony kadr zatrzymują teraz DWA niezależne progi: poza głowy
        (`max_yaw_asymmetry`) i bramka jakości kadru (`frame_quality`).
        Poluzowanie jednego nie wystarcza — dlatego wariant „loose" luzuje oba.
        """
        turned = {"nose_shift": 60.0}
        # Ekspresja na JEDNEJ klatce. Przy czterech na sześć detektor klatki
        # neutralnej brał kadr z otwartym pyskiem: utrzymana grymasa jest
        # temporalnie stabilniejsza od spokoju, a detektor optymalizuje
        # stabilność. Baza AU lądowała wtedy na szczycie i delta wychodziła
        # zerowa — scenariusz mierzyłby wtedy tę pułapkę, a nie próg pozy.
        expressions = {3}

        strict = _run(_pipeline(boxes=[LEFT_DOG_BOX], expression_frames=expressions, **turned), min_peak_separation_s=0.2)
        loose = _run(
            _pipeline(boxes=[LEFT_DOG_BOX], expression_frames=expressions, **turned),
            min_peak_separation_s=0.2,
            max_yaw_asymmetry=0.5,
            frame_quality=_OPEN_GATE,
        )

        assert strict["tracks"][0].peak_indices == []
        assert loose["tracks"][0].peak_indices

    def test_bramka_jakosci_nie_zeruje_treku_nagranego_w_zlych_warunkach(self) -> None:
        """
        Bramka przy WYBORZE peaków jest preferencją, nie wetem.

        Rozdział jest celowy. Selektor pyta „które kadry TEGO treku nadają się
        najlepiej" i musi coś zwrócić — trek nagrany gorzej ma dawać gorsze
        kadry, a nie zero kadrów. Weto należy do kuracji zbioru, która zna próg
        „człowiek to zweryfikuje" (`packages.pipeline.quality_gate`).

        Bez tego ustępstwa na materiale stockowym wychodziło zero peaków: sam
        próg rozmiaru mordy odrzucał 67 klatek na 100, bo mediana szerokości
        mordy w tym materiale to 26 px.
        """
        turned = {"nose_shift": 60.0}
        # Ekspresja na JEDNEJ klatce. Przy czterech na sześć detektor klatki
        # neutralnej brał kadr z otwartym pyskiem: utrzymana grymasa jest
        # temporalnie stabilniejsza od spokoju, a detektor optymalizuje
        # stabilność. Baza AU lądowała wtedy na szczycie i delta wychodziła
        # zerowa — scenariusz mierzyłby wtedy tę pułapkę, a nie próg pozy.
        expressions = {3}
        result = _run(
            _pipeline(boxes=[LEFT_DOG_BOX], expression_frames=expressions, **turned),
            min_peak_separation_s=0.2,
            # Próg pozy głowy poluzowany — zostaje sama bramka jakości kadru
            max_yaw_asymmetry=0.5,
        )
        assert result["tracks"][0].peak_indices

    def test_prog_przechylenia_glowy_dociera_do_wyboru_peakow(self) -> None:
        tilted = {"eye_tilt": 60.0}

        strict = _run(_pipeline(boxes=[LEFT_DOG_BOX], expression_frames={1, 2, 3, 7}, **tilted), min_peak_separation_s=0.2)
        loose = _run(
            _pipeline(boxes=[LEFT_DOG_BOX], expression_frames={1, 2, 3, 7}, **tilted),
            min_peak_separation_s=0.2,
            max_roll=45.0,
        )

        assert strict["tracks"][0].peak_indices == []
        assert loose["tracks"][0].peak_indices

    def test_prog_pewnosci_odsiewa_klatki_przed_liczeniem_au(self) -> None:
        """Klatka poniżej progu nie ma prawa współtworzyć bazy ani szumu treku."""
        result = _run(_pipeline(), min_keypoint_conf=_ABOVE_FIXTURE_CONF)

        assert not result["tracks"]
        for track in result["rejected_tracks"]:
            assert track.frames == []
            assert "za mało klatek" in track.rejected_reason

    def test_prog_godnosci_treku_jest_osobna_galka(self) -> None:
        """Luźny filtr klatek nie może obniżać bramki godności całego treku."""
        result = _run(
            _pipeline(),
            min_keypoint_conf=0.3,
            track_quality=TrackQuality(min_conf=_ABOVE_FIXTURE_CONF),
        )

        assert not result["tracks"]
        for track in result["rejected_tracks"]:
            assert track.frames, "klatki miały przejść filtr, odpaść ma dopiero trek"
            assert "pewność" in track.rejected_reason

    def test_wskazana_recznie_klatka_neutralna_jest_uzywana(self) -> None:
        result = _run(_pipeline(), neutral_frame_idx=3)

        assert result["tracks"]
        for track in result["tracks"]:
            assert track.neutral_frame_idx == 3
            assert track.neutral_source == NEUTRAL_SOURCE_MANUAL

    def test_niepoprawne_probkowanie_jest_odrzucane(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            VideoDatasetConfig(fps=0.0)


class TestOdrzucaniaTrekow:
    """Odrzucenie musi być zapisane z powodem, nie ciche."""

    def test_zbyt_krotkie_wideo_nie_rzuca_wyjatkiem(self) -> None:
        result = _run(_pipeline(), frames=2)

        assert result["tracks"] == []
        assert len(result["rejected_tracks"]) == 2

    def test_kazde_odrzucenie_ma_powod(self) -> None:
        result = _run(_pipeline(), frames=2)

        for track in result["rejected_tracks"]:
            assert track.rejected_reason, "każde odrzucenie musi mieć powód"

    def test_wideo_bez_psow_daje_pusty_wynik(self) -> None:
        result = _run(_pipeline(boxes=[]))

        assert result["tracks"] == []
        assert result["rejected_tracks"] == []
        assert result["total_frames"] == DEFAULT_FRAMES

    def test_awaria_jednego_treku_nie_zabiera_pozostalych(self, monkeypatch) -> None:
        """Jeden zepsuty pies nie może kosztować całego nagrania."""
        from packages.pipeline import inference

        original = inference.DeltaActionUnitsExtractor
        state = {"first": True}

        def _failing_extractor(baseline):
            if state["first"]:
                state["first"] = False
                raise ValueError("zdegenerowana geometria")
            return original(baseline)

        monkeypatch.setattr(inference, "DeltaActionUnitsExtractor", _failing_extractor)

        result = _run(_pipeline())

        assert len(result["tracks"]) == 1
        assert len(result["rejected_tracks"]) == 1
        assert "zdegenerowana geometria" in result["rejected_tracks"][0].rejected_reason


class TestSzumAu:
    """Szum AU treku opisuje wiarygodność jego klatek."""

    def test_nieruchomy_pies_ma_zerowy_szum_i_pelna_liczbe_prob(self) -> None:
        track = _run(_pipeline(boxes=[LEFT_DOG_BOX]))["tracks"][0]

        assert track.au_noise, "przyjęty trek musi mieć zmierzony szum AU"
        assert all(value == pytest.approx(0.0) for value in track.au_noise.values())
        assert all(
            track.au_sample_count[name] == len(track.frames) for name in track.au_noise
        )

    def test_drganie_punktow_podnosi_szum(self) -> None:
        still = _run(_pipeline(boxes=[LEFT_DOG_BOX]))["tracks"][0]
        shaky = _run(_pipeline(boxes=[LEFT_DOG_BOX], jitter=2.0))["tracks"][0]

        shared = set(still.au_noise) & set(shaky.au_noise)
        assert shared
        assert max(shaky.au_noise[name] for name in shared) > max(
            still.au_noise[name] for name in shared
        )


class TestWygladzania:
    """Filtr One Euro musi być podpięty osobno dla każdego treku."""

    # Zmierzona krotność redukcji przy drganiu 2 px i 5 kl./s to 3.35;
    # próg 2.0 zostawia margines na zmiany parametrów filtra, ale nie przepuszcza
    # filtra praktycznie przezroczystego (krotność ~1.0 — to był realny błąd w zadaniu 3).
    MIN_REDUCTION: float = 2.0

    def test_drganie_nosa_jest_tlumione(self) -> None:
        jitter = 2.0
        track = _run(_pipeline(boxes=[LEFT_DOG_BOX], jitter=jitter), frames=8)["tracks"][0]

        nose_x = _nose_x(track)[1:]
        spread = max(nose_x) - min(nose_x)
        reduction = (2 * jitter) / max(spread, 1e-9)

        assert reduction >= self.MIN_REDUCTION, (
            f"drganie tłumione tylko {reduction:.2f}x (rozrzut {spread:.2f} px "
            f"przy amplitudzie {2 * jitter:.2f} px)"
        )

    def test_skala_normalizacji_jest_stala_w_calym_treku(self, monkeypatch) -> None:
        """
        Boks mordy drga i zmienia źródło; skala normalizacji nie ma prawa drgać.

        Skok skali filtr czyta jako dużą prędkość i przestaje tłumić dokładnie tam,
        gdzie pomiar był najtrudniejszy.
        """
        calls = _spy_on_smoother(monkeypatch)

        result = _run(_pipeline(boxes=[LEFT_DOG_BOX], scale_jitter=6.0), frames=6)

        face_widths = {
            round(frame.face_box[2], 6) for frame in result["tracks"][0].frames
        }
        assert len(face_widths) > 1, "boks mordy ma drgać, inaczej test nic nie sprawdza"

        scales = {(round(call["box"][2], 6), round(call["box"][3], 6)) for call in calls}
        assert len(scales) == 1, f"skala normalizacji drga w obrębie treku: {scales}"
