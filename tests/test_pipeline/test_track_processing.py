"""Testy przetwarzania pojedynczego treku."""

import numpy as np
import pytest

from packages.data.schemas import NUM_KEYPOINTS
from packages.models.delta_action_units import DeltaActionUnit, DeltaActionUnitsExtractor
from packages.models.head_pose import HeadPose
from packages.pipeline.landmark_smoothing import KeypointSmoother
from packages.pipeline.track_processing import (
    NO_NEUTRAL_FRAME,
    TrackFrame,
    build_track_result,
    compute_au_noise,
    evaluate_track_quality,
    rejected_track,
)
from tests.test_pipeline.kp_fixtures import make_frontal_kp

FACE_BOX: tuple[float, float, float, float] = (60.0, 60.0, 200.0, 200.0)
FPS: float = 5.0
JITTER_PX: float = 2.0
JITTER_FRAMES: int = 25
BASELINE_FRAMES: int = 5


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
        head_pose=HeadPose(yaw_asymmetry=0.0, roll=0.0, is_frontal=True, confidence=0.9),
        delta_aus={name: _au(name, ratio) for name, ratio in ratios.items()},
    )


def _jittered_series(rng: np.random.Generator) -> list[np.ndarray]:
    """Nieruchoma frontalna morda z szumem gaussowskim na każdej współrzędnej."""
    base = make_frontal_kp().astype(float)

    series = []
    for _ in range(JITTER_FRAMES):
        frame = base.copy()
        frame[0::3] += rng.normal(0, JITTER_PX, NUM_KEYPOINTS)
        frame[1::3] += rng.normal(0, JITTER_PX, NUM_KEYPOINTS)
        series.append(frame)
    return series


def _noise_of_series(series: list[np.ndarray]) -> dict[str, float]:
    """Mierzy szum AU serii: baza z pierwszych klatek, pomiar na pozostałych."""
    extractor = DeltaActionUnitsExtractor(series[:BASELINE_FRAMES])
    frames = [
        TrackFrame(
            frame_idx=idx,
            keypoints=keypoints,
            face_box=FACE_BOX,
            head_pose=HeadPose(
                yaw_asymmetry=0.0, roll=0.0, is_frontal=True, confidence=0.9
            ),
            delta_aus=extractor.extract(keypoints),
        )
        for idx, keypoints in enumerate(series[BASELINE_FRAMES:], start=BASELINE_FRAMES)
    ]
    return compute_au_noise(frames)


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

    def test_noise_is_measured_on_smoothed_ratios(self) -> None:
        """
        Sedno zadania: do zbioru trafia ratio PO wygładzeniu, więc szum musi być
        mierzony na tej samej wartości. Zmierzone wcześniej sigma 0.35-0.76 przy progu
        aktywacji 0.15 brało się z drgania punktów — wygładzenie ma je wyraźnie obniżyć.
        """
        rng = np.random.default_rng(0)
        raw_series = _jittered_series(rng)
        smoother = KeypointSmoother()
        smoothed_series = [
            smoother.smooth(frame, FACE_BOX, timestamp=i / FPS)
            for i, frame in enumerate(raw_series)
        ]

        raw_noise = _noise_of_series(raw_series)
        smoothed_noise = _noise_of_series(smoothed_series)

        raw_median = float(np.median(list(raw_noise.values())))
        smoothed_median = float(np.median(list(smoothed_noise.values())))
        assert smoothed_median < raw_median / 1.5, (
            f"wygładzenie ma obniżyć szum AU, było {raw_median:.3f}, "
            f"jest {smoothed_median:.3f}"
        )
        assert all(
            smoothed_noise[name] <= raw_noise[name] for name in raw_noise
        ), "żaden AU nie może po wygładzeniu zrobić się bardziej rozdygotany"


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

    def test_au_noise_is_computed_for_track(self) -> None:
        frames = [
            _track_frame(idx, au_ratios={"AU101": ratio})
            for idx, ratio in zip((7, 8, 11), (0.8, 1.2, 1.0))
        ]

        result = build_track_result(
            track_id=1, frames=frames, neutral_position=0, peak_positions=[1]
        )

        assert result.au_noise["AU101"] > 0.0

    def test_peak_position_out_of_range_raises(self) -> None:
        """Pomyłka pozycja/numer klatki ma wybuchnąć głośno, nie po cichu."""
        frames = self._gapped_frames()

        with pytest.raises(IndexError, match="peak"):
            build_track_result(
                track_id=3, frames=frames, neutral_position=0, peak_positions=[20]
            )

    def test_neutral_position_out_of_range_raises(self) -> None:
        frames = self._gapped_frames()

        with pytest.raises(IndexError, match="neutraln"):
            build_track_result(
                track_id=3, frames=frames, neutral_position=7, peak_positions=[]
            )

    def test_empty_track_raises(self) -> None:
        with pytest.raises(ValueError, match="pust"):
            build_track_result(
                track_id=3, frames=[], neutral_position=0, peak_positions=[]
            )


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

    @pytest.mark.parametrize("reason", ["", "   "])
    def test_empty_reason_raises(self, reason: str) -> None:
        """Odrzucenie bez powodu jest bezużyteczne dla audytu lejka."""
        with pytest.raises(ValueError, match="powód"):
            rejected_track(track_id=2, frames=[], reason=reason)
