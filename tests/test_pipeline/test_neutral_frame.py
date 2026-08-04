"""
Testy dla modułu detekcji klatki neutralnej (HeadPose, NeutralFrameDetector).

Uruchomienie:
    pytest tests/test_pipeline/test_neutral_frame.py -v
"""

import numpy as np
import pytest

from packages.data.schemas import KP, NUM_KEYPOINTS
from packages.models.head_pose import HeadPose, estimate_head_pose
from packages.pipeline.neutral_frame import (
    NeutralFrameDetector,
    collect_neutral_baseline,
)
from tests.test_pipeline.kp_fixtures import make_frontal_kp

# =============================================================================
# Testy klasy HeadPose
# =============================================================================

class TestCollectNeutralBaseline:
    """Testy funkcji collect_neutral_baseline — okno klatek wokół neutralnej.

    Zamiast jednej (zaszumionej) klatki neutralnej, zbiera okno sąsiednich
    poprawnych klatek → median jako stabilna baza dla ekstraktora AU.
    """

    def test_collects_window_around_neutral(self) -> None:
        """Okno ±2 wokół klatki 3 zwraca 5 poprawnych klatek (indeksy 1..5)."""
        frames = [make_frontal_kp() for _ in range(7)]
        baseline = collect_neutral_baseline(frames, neutral_idx=3, window=2)
        assert len(baseline) == 5

    def test_skips_none_frames(self) -> None:
        """Klatki None w oknie są pomijane."""
        frames: list = [make_frontal_kp() for _ in range(7)]
        frames[2] = None
        frames[4] = None
        baseline = collect_neutral_baseline(frames, neutral_idx=3, window=2)
        # Okno 1..5 → wykluczone 2 i 4 → zostają 1, 3, 5
        assert len(baseline) == 3

    def test_clamps_at_left_boundary(self) -> None:
        """Okno przy lewej krawędzi nie wychodzi poza zakres."""
        frames = [make_frontal_kp() for _ in range(7)]
        baseline = collect_neutral_baseline(frames, neutral_idx=0, window=2)
        # Indeksy 0..2 → 3 klatki
        assert len(baseline) == 3

    def test_window_zero_returns_single_frame(self) -> None:
        """Okno 0 zwraca tylko klatkę neutralną."""
        frames = [make_frontal_kp() for _ in range(7)]
        baseline = collect_neutral_baseline(frames, neutral_idx=3, window=0)
        assert len(baseline) == 1

    def test_no_valid_frames_raises(self) -> None:
        """Brak poprawnych klatek w oknie rzuca ValueError."""
        frames: list = [None, None, None]
        with pytest.raises(ValueError, match="poprawn|valid|neutral"):
            collect_neutral_baseline(frames, neutral_idx=1, window=1)


class TestNeutralPrefersFrontal:
    """Detektor neutralny powinien przy równej stabilności preferować frontalną pozę.

    Zła baza (głowa odwrócona/pochylona) psuje wszystkie delta AU — zwłaszcza uszy.
    """

    def test_prefers_frontal_among_equally_stable(self) -> None:
        """Przy identycznych (stabilnych) keypoints wybierana jest najbardziej frontalna klatka."""
        kp = make_frontal_kp()
        keypoints_list = [kp.copy() for _ in range(7)]
        # Wszystkie klatki jednakowo stabilne; różni je tylko poza głowy.
        # Klatki bardziej odwrócone (wysoka asymetria yaw, ale wciąż w granicach kandydata)
        # i jedna wyraźnie frontalna (idx 3).
        def hp(yaw_asymmetry: float) -> HeadPose:
            return HeadPose(
                yaw_asymmetry=yaw_asymmetry, roll=0.0, is_frontal=True, confidence=0.9
            )
        head_poses = [
            hp(0.30), hp(0.28), hp(0.30),
            hp(0.02),                      # idx 3 — najbardziej frontalna
            hp(0.30), hp(0.28), hp(0.30),
        ]
        detector = NeutralFrameDetector()
        idx = detector.detect_auto(
            frames=[None] * 7, keypoints_list=keypoints_list, head_poses=head_poses
        )
        assert idx == 3, f"oczekiwano najbardziej frontalnej klatki (3), wybrano {idx}"


class TestHeadPose:
    """Testy dla klasy HeadPose."""

    def test_creation_frontal(self) -> None:
        """Test tworzenia frontalnej pozy głowy."""
        pose = HeadPose(
            yaw_asymmetry=0.05,
            roll=2.0,
            is_frontal=True,
            confidence=0.95,
        )

        assert pose.yaw_asymmetry == 0.05
        assert pose.roll == 2.0
        assert pose.is_frontal is True
        assert pose.confidence == 0.95

    def test_to_dict(self) -> None:
        """Test konwersji HeadPose do słownika."""
        pose = HeadPose(yaw_asymmetry=0.1, roll=3.0, is_frontal=True, confidence=0.9)

        result = pose.to_dict()

        assert result["yaw_asymmetry"] == 0.1
        assert result["roll"] == 3.0
        assert result["is_frontal"] is True
        assert result["confidence"] == 0.9


# =============================================================================
# Testy funkcji estimate_head_pose
# =============================================================================

class TestEstimateHeadPose:
    """Testy dla funkcji estimate_head_pose."""

    def test_frontal_face_is_detected_as_frontal(self) -> None:
        """Test: symetryczna twarz powinna być wykryta jako frontalna."""
        kp = make_frontal_kp()
        pose = estimate_head_pose(kp)

        assert pose.is_frontal is True
        assert abs(pose.yaw_asymmetry) < 0.05
        assert abs(pose.roll) < 25

    def test_left_turned_face_has_negative_yaw_asymmetry(self) -> None:
        """Test: twarz obrócona w lewo (nos bliżej lewego oka) ma yaw_asymmetry < 0."""
        kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        # Przesuń nos w lewo
        kp[KP.NOSE_TIP, 0] -= 35
        kp[KP.RIGHT_EYE_INNER, 0] -= 20
        kp[KP.RIGHT_EYE_OUTER, 0] -= 20

        pose = estimate_head_pose(kp.flatten())

        assert pose.yaw_asymmetry < -0.05

    def test_right_turned_face_has_positive_yaw_asymmetry(self) -> None:
        """Test: twarz obrócona w prawo (nos bliżej prawego oka) ma yaw_asymmetry > 0."""
        kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        kp[KP.NOSE_TIP, 0] += 35
        kp[KP.LEFT_EYE_INNER, 0] += 20
        kp[KP.LEFT_EYE_OUTER, 0] += 20

        pose = estimate_head_pose(kp.flatten())

        assert pose.yaw_asymmetry > 0.05

    def test_tilted_eyes_produce_nonzero_roll(self) -> None:
        """Test: przekrzywione oczy dają roll ≠ 0."""
        kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        # Prawe oko wyżej, lewe niżej
        kp[KP.RIGHT_EYE_INNER, 1] -= 15
        kp[KP.RIGHT_EYE_OUTER, 1] -= 15
        kp[KP.LEFT_EYE_INNER, 1] += 15
        kp[KP.LEFT_EYE_OUTER, 1] += 15

        pose = estimate_head_pose(kp.flatten())

        assert abs(pose.roll) > 10

    def test_low_visibility_reduces_confidence(self) -> None:
        """Test: niska widoczność kluczowych keypoints obniża confidence."""
        kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        kp[KP.LEFT_EYE_INNER, 2] = 0.2
        kp[KP.NOSE_TIP, 2] = 0.3

        pose = estimate_head_pose(kp.flatten())

        assert pose.confidence < 0.9

    def test_invalid_keypoints_raises(self) -> None:
        """Test: nieprawidłowa liczba keypoints rzuca ValueError."""
        with pytest.raises(ValueError):
            estimate_head_pose(np.zeros(60))  # Stare 20 keypoints


# =============================================================================
# Testy klasy NeutralFrameDetector
# =============================================================================

class TestNeutralFrameDetector:
    """Testy dla klasy NeutralFrameDetector."""

    @pytest.fixture
    def detector(self) -> NeutralFrameDetector:
        """Fixture: detektor z domyślnymi parametrami."""
        return NeutralFrameDetector()

    def test_default_initialization(self, detector: NeutralFrameDetector) -> None:
        """Test inicjalizacji detektora z domyślnymi parametrami."""
        assert detector.min_keypoint_conf == 0.5
        assert detector.max_yaw_asymmetry == 0.35
        assert detector.max_roll == 30.0

    def test_custom_initialization(self) -> None:
        """Test inicjalizacji z niestandardowymi parametrami."""
        detector = NeutralFrameDetector(
            min_keypoint_conf=0.8,
            max_yaw_asymmetry=0.15,
            max_roll=10.0,
        )

        assert detector.min_keypoint_conf == 0.8
        assert detector.max_yaw_asymmetry == 0.15

    def test_empty_sequence_raises_value_error(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: pusta sekwencja klatek rzuca ValueError."""
        with pytest.raises(ValueError):
            detector.detect_auto([], [])

    def test_single_frame_returns_zero(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: pojedyncza klatka → zawsze indeks 0."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp = make_frontal_kp()

        neutral_idx = detector.detect_auto([frame], [kp])

        assert neutral_idx == 0

    def test_detect_auto_returns_valid_index(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: detect_auto zwraca poprawny indeks w zakresie sekwencji."""
        n_frames = 10
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(n_frames)]
        keypoints = [make_frontal_kp() for _ in range(n_frames)]

        neutral_idx = detector.detect_auto(frames, keypoints)

        assert 0 <= neutral_idx < n_frames

    def test_stability_score_stable_sequence(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: wysoki wynik stabilności dla stabilnej sekwencji."""
        base_kp = make_frontal_kp()
        rng = np.random.default_rng(42)

        # Sekwencja z minimalnym szumem
        keypoints_list = [
            (base_kp.reshape(NUM_KEYPOINTS, 3)
             + np.column_stack([
                 rng.random((NUM_KEYPOINTS, 2)) * 0.5,
                 np.zeros((NUM_KEYPOINTS, 1)),
             ])).flatten()
            for _ in range(20)
        ]

        score = detector._compute_stability_score(keypoints_list, center_idx=10)

        assert score > 0.8

    def test_stability_score_unstable_sequence(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: niski wynik stabilności dla niestabilnej sekwencji."""
        rng = np.random.default_rng(42)
        base_kp = make_frontal_kp()

        # Sekwencja z dużym szumem
        keypoints_list = [
            (base_kp.reshape(NUM_KEYPOINTS, 3)
             + np.column_stack([
                 rng.random((NUM_KEYPOINTS, 2)) * 30,
                 np.zeros((NUM_KEYPOINTS, 1)),
             ])).flatten()
            for _ in range(20)
        ]

        score = detector._compute_stability_score(keypoints_list, center_idx=10)

        assert score < 0.5

    def test_frontal_frame_is_valid_candidate(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: klatka z frontalną twarzą i wysoką visibility jest kandydatem."""
        kp = make_frontal_kp()
        pose = estimate_head_pose(kp)

        is_valid = detector._is_valid_candidate(kp, pose)

        assert is_valid is True

    def test_non_frontal_frame_is_not_valid_candidate(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: klatka z obrócona twarzą nie jest kandydatem."""
        kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        # Duże obrócenie — nose bardzo z boku
        kp[KP.NOSE_TIP, 0] += 80
        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        is_valid = detector._is_valid_candidate(kp_flat, pose)

        assert is_valid is False

    def test_low_visibility_frame_is_not_valid_candidate(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: klatka z niską widocznością keypoints nie jest kandydatem."""
        kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        kp[:, 2] = 0.1  # Niska visibility dla wszystkich
        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        is_valid = detector._is_valid_candidate(kp_flat, pose)

        assert is_valid is False

    def test_detect_manual_returns_given_index(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: detect_manual zwraca podany indeks bez zmian."""
        assert detector.detect_manual(5) == 5
        assert detector.detect_manual(0) == 0
