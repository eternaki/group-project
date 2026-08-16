"""Testy przetwarzania pojedynczego treku."""

import numpy as np
import pytest

from packages.data.schemas import NUM_KEYPOINTS
from packages.models.delta_action_units import (
    DEFAULT_ACTIVATION_THRESHOLD,
    DeltaActionUnit,
    DeltaActionUnitsExtractor,
)
from packages.models.head_pose import HeadPose
from packages.pipeline.landmark_smoothing import KeypointSmoother
from packages.pipeline.track_processing import (
    MIN_NOISE_SAMPLES,
    NO_NEUTRAL_FRAME,
    TrackFrame,
    TrackQuality,
    build_track_result,
    compute_au_noise,
    count_au_samples,
    evaluate_track_quality,
    rejected_track,
)
from tests.test_pipeline.kp_fixtures import make_frontal_kp

FACE_BOX: tuple[float, float, float, float] = (60.0, 60.0, 200.0, 200.0)
# Boks całego psa — szerszy niż morda (to on trafia do zbioru jako bbox anotacji)
BODY_BOX: tuple[int, int, int, int] = (20, 20, 300, 320)
FPS: float = 5.0
# Szum 2 px przy rozstawie oczu 100 px w fixturze = NME_iod 0.02 (model idealny)
JITTER_PX: float = 2.0
# Szum odpowiadający realnej jakości naszego modelu keypoints: NME_iod 0.091
REALISTIC_JITTER_PX: float = 9.1
JITTER_FRAMES: int = 25
BASELINE_FRAMES: int = 5
# Liczba realizacji szumu w teście charakteryzującym — pojedyncze losowanie daje
# rozrzut mediany sigma 0.121-0.211, więc jeden przebieg przypinałby skrajność
NOISE_REALIZATIONS: int = 20
# Próg aktywacji AU wyrażony jako delta (ratio 1.15 → 0.15)
ACTIVATION_DELTA: float = DEFAULT_ACTIVATION_THRESHOLD - 1.0
# AU liczone z SAMEJ widoczności punktów — filtr nie rusza kanału widoczności,
# więc wygładzanie nie ma jak obniżyć ich szumu (w fixturze widoczność jest stała).
#
# AD19 tu NIE należy, choć długo tak było zapisane. Liczy się z widoczności języka
# ORAZ z geometrii (o ile czubek języka wystaje poniżej dolnej wargi), więc reaguje
# na drganie punktów jak każde AU geometryczne. Wyglądał na czysto widocznościowy
# tylko dlatego, że fixtura zostawiała TONGUE_TIP w (0, 0) z zerową pewnością —
# psa bez języka. Po uzupełnieniu fixtury szum AD19 spada z 0.020 do 0.008.
VISIBILITY_BASED_AUS: frozenset[str] = frozenset({"AU145"})


def _au(name: str, ratio: float) -> DeltaActionUnit:
    """Buduje pojedynczy pomiar AU o zadanym ratio."""
    return DeltaActionUnit(
        name=name,
        ratio=ratio,
        delta=ratio - 1.0,
        is_active=False,
        confidence=0.8,
    )


def _track_frame(
    frame_idx: int,
    face_size: float = 120.0,
    confidence: float = 0.8,
    au_ratios: dict[str, float] | None = None,
) -> TrackFrame:
    """Buduje klatkę treku o zadanym rozmiarze mordy, pewności i wartościach AU."""
    keypoints = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
    keypoints[2::3] = confidence
    ratios = {"AU101": 1.0} if au_ratios is None else au_ratios
    return TrackFrame(
        frame_idx=frame_idx,
        keypoints=keypoints,
        face_box=(0.0, 0.0, face_size, face_size),
        body_box=BODY_BOX,
        head_pose=HeadPose(yaw_asymmetry=0.0, roll=0.0, is_frontal=True, confidence=0.9),
        delta_aus={name: _au(name, ratio) for name, ratio in ratios.items()},
    )


def _jittered_series(
    rng: np.random.Generator, jitter_px: float = JITTER_PX
) -> list[np.ndarray]:
    """Nieruchoma frontalna morda z szumem gaussowskim na każdej współrzędnej."""
    base = make_frontal_kp().astype(float)

    series = []
    for _ in range(JITTER_FRAMES):
        frame = base.copy()
        frame[0::3] += rng.normal(0, jitter_px, NUM_KEYPOINTS)
        frame[1::3] += rng.normal(0, jitter_px, NUM_KEYPOINTS)
        series.append(frame)
    return series


def _smoothed_series(series: list[np.ndarray]) -> list[np.ndarray]:
    """Przepuszcza serię przez filtr treku (jedna instancja na trek)."""
    smoother = KeypointSmoother()
    return [
        smoother.smooth(frame, FACE_BOX, timestamp=i / FPS)
        for i, frame in enumerate(series)
    ]


def _noise_of_series(series: list[np.ndarray]) -> dict[str, float]:
    """Mierzy szum AU serii: baza z pierwszych klatek, pomiar na pozostałych."""
    extractor = DeltaActionUnitsExtractor(series[:BASELINE_FRAMES])
    frames = [
        TrackFrame(
            frame_idx=idx,
            keypoints=keypoints,
            face_box=FACE_BOX,
            body_box=BODY_BOX,
            head_pose=HeadPose(
                yaw_asymmetry=0.0, roll=0.0, is_frontal=True, confidence=0.9
            ),
            delta_aus=extractor.extract(keypoints),
        )
        for idx, keypoints in enumerate(series[BASELINE_FRAMES:], start=BASELINE_FRAMES)
    ]
    return compute_au_noise(frames)


class TestTrackQualityValidation:
    """Progi godności nie mogą dać się ustawić tak, żeby wyłączyć bramkę."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_frames": 0},
            {"min_frames": 1},
            {"min_frames": -3},
            {"min_face_px": 0.0},
            {"min_face_px": -10.0},
            {"min_conf": 0.0},
            {"min_conf": 1.5},
        ],
    )
    def test_out_of_range_thresholds_raise(self, kwargs: dict) -> None:
        """
        Regresja: `min_frames=0` przepuszczał PUSTY trek jako przyjęty.

        Mechanizm: `np.median([])` daje NaN, a `NaN < próg` jest fałszywe, więc
        pozostałe kontrole też przepuszczały. Dopiero dalej leciał nieczytelny
        `IndexError` po dwóch ostrzeżeniach numpy.
        """
        with pytest.raises(ValueError):
            TrackQuality(**kwargs)

    def test_empty_track_cannot_be_accepted_by_loosened_thresholds(self) -> None:
        """Nie ma konfiguracji, przy której pusty trek przechodzi bramkę."""
        loosest = TrackQuality(min_frames=MIN_NOISE_SAMPLES, min_face_px=1e-6, min_conf=1e-6)

        assert evaluate_track_quality([], quality=loosest) is not None


class TestEvaluateTrackQuality:
    """Próg godności treku — odrzucenie musi mieć podany powód."""

    def test_good_track_is_accepted(self) -> None:
        frames = [_track_frame(i) for i in range(5)]

        assert evaluate_track_quality(frames) is None

    def test_too_few_frames_rejected_with_reason(self) -> None:
        frames = [_track_frame(i) for i in range(2)]

        reason = evaluate_track_quality(frames)

        assert reason is not None and "klatek" in reason

    def test_empty_track_rejected_with_reason(self) -> None:
        """Trek bez ani jednej klatki z mordą też musi mieć powód, nie wyjątek."""
        reason = evaluate_track_quality([])

        assert reason is not None and "klatek" in reason

    def test_small_face_rejected_with_reason(self) -> None:
        frames = [_track_frame(i, face_size=40.0) for i in range(5)]

        reason = evaluate_track_quality(frames)

        assert reason is not None and "morda" in reason

    def test_low_confidence_rejected_with_reason(self) -> None:
        frames = [_track_frame(i, confidence=0.2) for i in range(5)]

        reason = evaluate_track_quality(frames)

        assert reason is not None and "pewność" in reason

    def test_single_bad_frame_does_not_sink_whole_track(self) -> None:
        """Próg liczony z mediany, więc jedna zła klatka nie przekreśla treku."""
        frames = [_track_frame(i) for i in range(4)]
        frames.append(_track_frame(4, face_size=10.0, confidence=0.05))

        assert evaluate_track_quality(frames) is None

    def test_narrow_face_box_is_measured_by_shorter_side(self) -> None:
        """Morda 200x30 px to zbyt mała morda — decyduje krótszy bok, nie dłuższy."""
        frames = [_track_frame(i) for i in range(5)]
        for frame in frames:
            frame.face_box = (0.0, 0.0, 200.0, 30.0)

        reason = evaluate_track_quality(frames)

        assert reason is not None and "morda" in reason

    @pytest.mark.parametrize(
        ("quality", "expected"),
        [
            (TrackQuality(min_frames=6), "klatek"),
            (TrackQuality(min_face_px=200.0), "morda"),
            (TrackQuality(min_conf=0.9), "pewność"),
        ],
    )
    def test_custom_thresholds_are_respected(
        self, quality: TrackQuality, expected: str
    ) -> None:
        """
        Próg z konfiguracji pipeline'u musi dojechać do oceny treku.

        Bez tego użytkownik ustawiający min_keypoint_conf=0.7 dostawał treki przyjęte
        przy 0.4 — ten sam martwy próg, który dwa razy przeszedł w tym sprincie.
        """
        frames = [_track_frame(i) for i in range(5)]

        reason = evaluate_track_quality(frames, quality=quality)

        assert reason is not None and expected in reason


class TestComputeAuNoise:
    """Szum AU na trek — waga wiarygodności dla przyszłego treningu."""

    def test_constant_signal_gives_zero_noise(self) -> None:
        frames = [_track_frame(i) for i in range(5)]

        noise = compute_au_noise(frames)

        assert noise["AU101"] == 0.0

    def test_varying_signal_gives_positive_noise(self) -> None:
        ratios = [0.8, 1.2, 0.9, 1.4, 1.0]
        frames = [_track_frame(i, au_ratios={"AU101": r}) for i, r in enumerate(ratios)]

        noise = compute_au_noise(frames)

        assert noise["AU101"] > 0.15, "rozrzut ma być widoczny w mierze szumu"

    def test_noise_is_measured_per_au_separately(self) -> None:
        """Jeden rozdygotany AU nie może zanieczyścić miary spokojnego AU."""
        ratios = [0.6, 1.4, 0.7, 1.5, 0.9]
        frames = [
            _track_frame(i, au_ratios={"AU101": r, "AU25": 1.0})
            for i, r in enumerate(ratios)
        ]

        noise = compute_au_noise(frames)

        assert noise["AU25"] == 0.0
        assert noise["AU101"] > 0.3

    def test_empty_track_gives_empty_noise(self) -> None:
        assert compute_au_noise([]) == {}

    def test_estimator_is_unbiased(self) -> None:
        """
        ddof=1, nie ddof=0.

        Wariant populacyjny zaniża szum tym mocniej, im krótszy trek (przy 3 klatkach
        średnio o 28%), więc dawałby najkrótszym trekom najwyższą wagę wiarygodności.
        """
        frames = [_track_frame(i, au_ratios={"AU101": r}) for i, r in enumerate((1.0, 2.0))]

        noise = compute_au_noise(frames)

        assert noise["AU101"] == pytest.approx(0.7071, abs=1e-4)

    def test_single_sample_au_is_omitted(self) -> None:
        """
        Sigma z jednej próbki nie istnieje — zero udawałoby pomiar idealnie stabilny.

        Brak klucza jest jednoznaczny i poprawnie się serializuje; ile prób było,
        mówi `count_au_samples`.
        """
        frames = [_track_frame(i) for i in range(3)]
        frames[0].delta_aus["AU25"] = _au("AU25", 1.4)

        noise = compute_au_noise(frames)

        assert "AU25" not in noise
        assert count_au_samples(frames)["AU25"] == 1

    def test_sample_count_reports_frames_per_au(self) -> None:
        frames = [_track_frame(i) for i in range(4)]

        assert count_au_samples(frames) == {"AU101": 4}

    def test_nan_ratio_is_dropped_and_counted(self) -> None:
        """
        NaN nie może przeciec do zbioru: `json.dumps` zapisuje literał `NaN`
        (niepoprawny JSON), a `allow_nan=False` wysypuje zapis. `np.clip` w
        ekstraktorze AU NaN-a nie zatrzymuje, więc odsiewamy go tutaj.
        """
        ratios = [1.0, float("nan"), 1.4, float("inf")]
        frames = [_track_frame(i, au_ratios={"AU101": r}) for i, r in enumerate(ratios)]

        noise = compute_au_noise(frames)

        assert np.isfinite(noise["AU101"])
        assert count_au_samples(frames)["AU101"] == 2

    def test_all_measurements_nan_gives_no_noise_but_zero_count(self) -> None:
        nan_ratios = {"AU101": float("nan")}
        frames = [_track_frame(i, au_ratios=nan_ratios) for i in range(3)]

        assert compute_au_noise(frames) == {}
        assert count_au_samples(frames) == {"AU101": 0}

    def test_noise_is_measured_on_smoothed_ratios(self) -> None:
        """
        Sedno zadania: do zbioru trafia ratio PO wygładzeniu, więc szum musi być
        mierzony na tej samej wartości. Zmierzone wcześniej sigma 0.35-0.76 przy progu
        aktywacji 0.15 brało się z drgania punktów — wygładzenie ma je wyraźnie obniżyć.
        """
        rng = np.random.default_rng(0)
        raw_series = _jittered_series(rng)

        raw_noise = _noise_of_series(raw_series)
        smoothed_noise = _noise_of_series(_smoothed_series(raw_series))

        raw_median = float(np.median(list(raw_noise.values())))
        smoothed_median = float(np.median(list(smoothed_noise.values())))
        assert smoothed_median < raw_median / 1.5, (
            f"wygładzenie ma obniżyć szum AU, było {raw_median:.3f}, "
            f"jest {smoothed_median:.3f}"
        )
        geometric = [name for name in raw_noise if name not in VISIBILITY_BASED_AUS]
        assert all(
            smoothed_noise[name] < raw_noise[name] for name in geometric
        ), "każdy AU liczony z geometrii ma zejść z szumem, nie tylko nie urosnąć"
        assert all(
            smoothed_noise[name] == raw_noise[name] == 0.0
            for name in VISIBILITY_BASED_AUS
        ), "AU145 i AD19 liczą się z widoczności — filtr ich nie dotyka (patrz stała)"

    def test_noise_at_real_model_error_is_of_the_order_of_activation_threshold(self) -> None:
        """
        Test charakteryzujący (nie progowy): zapisuje, gdzie naprawdę jesteśmy.

        Poprzedni test używa jitteru 2 px = NME_iod 0.02, czyli modelu idealnego.
        Nasz model keypoints ma NME_iod 0.091 (9.1 px przy rozstawie oczu 100 px).

        Liczby uśrednione po `NOISE_REALIZATIONS` realizacjach szumu, bo pojedyncze
        losowanie daje rozrzut, który łatwo wziąć za wynik: mediana sigma po
        wygładzeniu waha się 0.121-0.211 (mediana 0.166), a liczba AU powyżej progu
        9-16. Zapisanie jednego przebiegu oznaczałoby podanie skrajności za regułę.

        Stan faktyczny: mediana sigma PO wygładzeniu jest RZĘDU progu aktywacji
        0.15, a nie bezpiecznie pod nim; zysk z wygładzania spada z ~2.8x (jitter
        2 px) do ~1.5x. Wniosek: samo wygładzanie nie wystarczy — dopóki keypoints
        są tej jakości, część AU zapala się od szumu, reguły AU pozostają
        pre-etykietami, a `au_noise` musi ważyć te klatki w treningu.
        """
        medians: list[float] = []
        counts: list[int] = []
        for seed in range(NOISE_REALIZATIONS):
            rng = np.random.default_rng(seed)
            raw_series = _jittered_series(rng, jitter_px=REALISTIC_JITTER_PX)
            smoothed_noise = _noise_of_series(_smoothed_series(raw_series))
            medians.append(float(np.median(list(smoothed_noise.values()))))
            counts.append(sum(1 for s in smoothed_noise.values() if s > ACTIVATION_DELTA))

        typical_median = float(np.median(medians))
        typical_count = float(np.median(counts))

        assert typical_median > ACTIVATION_DELTA / 2, (
            f"stan faktyczny: typowa mediana sigma {typical_median:.3f} przy progu "
            f"aktywacji {ACTIVATION_DELTA:.2f} (zakres {min(medians):.3f}-{max(medians):.3f})"
        )
        assert typical_count >= 9, (
            f"stan faktyczny: typowo {typical_count:.0f} z 21 AU powyżej progu "
            f"aktywacji (zakres {min(counts)}-{max(counts)})"
        )


class TestBuildTrackResult:
    """Zamiana pozycji w treku na numery klatek wideo."""

    def _gapped_frames(self) -> list[TrackFrame]:
        """Trek zaczynający się od klatki 7, z lukami (pies znikał z kadru)."""
        return [_track_frame(idx) for idx in (7, 8, 11, 12, 20)]

    def test_peak_indices_are_video_frame_indices(self) -> None:
        frames = self._gapped_frames()

        result = build_track_result(
            track_id=3, frames=frames, neutral_position=0, peak_positions=[2, 4]
        )

        assert result.peak_indices == [11, 20], "peaki to numery klatek wideo"

    def test_neutral_frame_idx_is_video_frame_index(self) -> None:
        frames = self._gapped_frames()

        result = build_track_result(
            track_id=3, frames=frames, neutral_position=1, peak_positions=[]
        )

        assert result.neutral_frame_idx == 8

    def test_peak_order_is_preserved(self) -> None:
        """Selektor zwraca peaki wg siły mimiki — kolejności nie wolno gubić."""
        frames = self._gapped_frames()

        result = build_track_result(
            track_id=3, frames=frames, neutral_position=0, peak_positions=[4, 1, 2]
        )

        assert result.peak_indices == [20, 8, 11]

    def test_accepted_track_has_no_rejected_reason(self) -> None:
        frames = self._gapped_frames()

        result = build_track_result(
            track_id=3, frames=frames, neutral_position=0, peak_positions=[2]
        )

        assert result.rejected_reason is None
        assert result.track_id == 3
        assert result.frames == frames

    def test_au_noise_and_sample_count_are_computed_for_track(self) -> None:
        frames = [
            _track_frame(idx, au_ratios={"AU101": ratio})
            for idx, ratio in zip((7, 8, 11), (0.8, 1.2, 1.0))
        ]

        result = build_track_result(
            track_id=1, frames=frames, neutral_position=0, peak_positions=[1]
        )

        assert result.au_noise["AU101"] > 0.0
        assert result.au_sample_count == {"AU101": 3}

    def test_peak_position_out_of_range_raises(self) -> None:
        """Pomyłka pozycja/numer klatki wybucha, o ile numer wypada poza trekiem."""
        frames = self._gapped_frames()

        with pytest.raises(IndexError, match="peak"):
            build_track_result(
                track_id=3, frames=frames, neutral_position=0, peak_positions=[20]
            )

    def test_frame_number_inside_range_is_not_detectable(self) -> None:
        """
        Granica strażnika, świadomie udokumentowana.

        W długim treku numer klatki wideo bywa poprawną pozycją, więc pomyłki nie da
        się wykryć — dlatego konwersja ma zostać wyłącznie w `build_track_result`.
        """
        frames = [_track_frame(idx) for idx in range(10, 40)]

        result = build_track_result(
            track_id=3, frames=frames, neutral_position=0, peak_positions=[20]
        )

        assert result.peak_indices == [30], "pozycja 20 to klatka 30, nie klatka 20"

    def test_neutral_position_out_of_range_raises(self) -> None:
        frames = self._gapped_frames()

        with pytest.raises(IndexError, match="neutraln"):
            build_track_result(
                track_id=3, frames=frames, neutral_position=7, peak_positions=[]
            )

    def test_duplicate_peak_positions_raise(self) -> None:
        """Ta sama klatka trzy razy w zbiorze to potrójna waga w treningu."""
        frames = self._gapped_frames()

        with pytest.raises(ValueError, match="Powtórzone"):
            build_track_result(
                track_id=3, frames=frames, neutral_position=0, peak_positions=[3, 3, 3]
            )

    def test_empty_track_is_rejected_not_raised(self) -> None:
        result = build_track_result(
            track_id=3, frames=[], neutral_position=0, peak_positions=[]
        )

        assert result.rejected_reason is not None and "klatek" in result.rejected_reason
        assert result.neutral_frame_idx == NO_NEUTRAL_FRAME

    def test_track_below_quality_threshold_is_rejected_here(self) -> None:
        """
        Bramka jakości nie może być opcjonalna.

        Trek 1-klatkowy przechodził jako przyjęty z szumem 0.0, czyli z najwyższą
        wagą wiarygodności, gdyby wywołujący pominął `evaluate_track_quality`.
        """
        frames = [_track_frame(7)]

        result = build_track_result(
            track_id=3, frames=frames, neutral_position=0, peak_positions=[0]
        )

        assert result.rejected_reason is not None
        assert result.au_noise == {}
        assert result.peak_indices == []

    def test_quality_thresholds_reach_the_gate(self) -> None:
        """Próg z konfiguracji ma działać także przez `build_track_result`."""
        frames = self._gapped_frames()

        result = build_track_result(
            track_id=3,
            frames=frames,
            neutral_position=0,
            peak_positions=[2],
            quality=TrackQuality(min_conf=0.9),
        )

        assert result.rejected_reason is not None and "pewność" in result.rejected_reason


class TestRejectedTrack:
    """Odrzucony trek zapisuje powód — audyt lejka opiera się na tych powodach."""

    def test_rejected_track_keeps_reason(self) -> None:
        frames = [_track_frame(7)]

        result = rejected_track(track_id=2, frames=frames, reason="za mało klatek")

        assert result.rejected_reason == "za mało klatek"
        assert result.track_id == 2
        assert result.frames == frames

    def test_rejected_track_has_no_neutral_frame_and_no_peaks(self) -> None:
        result = rejected_track(track_id=2, frames=[], reason="brak klatki neutralnej")

        assert result.neutral_frame_idx == NO_NEUTRAL_FRAME
        assert result.peak_indices == []
        assert result.au_noise == {}
        assert result.au_sample_count == {}

    @pytest.mark.parametrize("reason", ["", "   "])
    def test_empty_reason_raises(self, reason: str) -> None:
        """Odrzucenie bez powodu jest bezużyteczne dla audytu lejka."""
        with pytest.raises(ValueError, match="powód"):
            rejected_track(track_id=2, frames=[], reason=reason)
