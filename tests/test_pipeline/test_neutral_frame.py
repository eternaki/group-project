"""
Testy dla modułu detekcji klatki neutralnej (HeadPose, NeutralFrameDetector).

Uruchomienie:
    pytest tests/test_pipeline/test_neutral_frame.py -v
"""

import numpy as np
import pytest

from packages.data.schemas import KP, NUM_KEYPOINTS
from packages.models.head_pose import HeadPose, estimate_head_pose
from packages.pipeline.neutral_frame import NeutralFrameDetector

# =============================================================================
# Fixture: realistyczna twarz psa (46 keypoints)
# =============================================================================

def make_frontal_kp() -> np.ndarray:
    """
    Tworzy realistyczną tablicę 46 keypoints — frontalna twarz psa.

    Centrum (150, 150), odległość między centrami oczu ~100px.
    Symetria lewo/prawa → minimalne yaw/pitch/roll.

    Returns:
        Tablica (138,) z wartościami [x0, y0, v0, ...]
    """
    kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    kp[:, 2] = 0.95

    # Oczy
    kp[KP.LEFT_EYE_INNER]  = [105, 150, 0.95]
    kp[KP.LEFT_EYE_TOP]    = [100, 144, 0.95]
    kp[KP.LEFT_EYE_OUTER]  = [95, 150, 0.95]
    kp[KP.LEFT_EYE_BOTTOM] = [100, 156, 0.95]
    kp[KP.RIGHT_EYE_INNER] = [195, 150, 0.95]
    kp[KP.RIGHT_EYE_TOP]   = [200, 144, 0.95]
    kp[KP.RIGHT_EYE_OUTER] = [205, 150, 0.95]
    kp[KP.RIGHT_EYE_BOTTOM]= [200, 156, 0.95]

    # Brwi
    kp[KP.LEFT_BROW_INNER]  = [108, 137, 0.9]
    kp[KP.LEFT_BROW_CENTER] = [100, 134, 0.9]
    kp[KP.LEFT_BROW_OUTER]  = [92, 137, 0.9]
    kp[KP.RIGHT_BROW_INNER] = [192, 137, 0.9]
    kp[KP.RIGHT_BROW_CENTER]= [200, 134, 0.9]
    kp[KP.RIGHT_BROW_OUTER] = [208, 137, 0.9]

    # Uszy (symetryczne)
    kp[KP.LEFT_EAR_BASE_FRONT]  = [80, 120, 0.9]
    kp[KP.LEFT_EAR_BASE_BACK]   = [75, 115, 0.85]
    kp[KP.LEFT_EAR_MID]         = [72, 95, 0.85]
    kp[KP.LEFT_EAR_TIP]         = [68, 70, 0.8]
    kp[KP.RIGHT_EAR_BASE_FRONT] = [220, 120, 0.9]
    kp[KP.RIGHT_EAR_BASE_BACK]  = [225, 115, 0.85]
    kp[KP.RIGHT_EAR_MID]        = [228, 95, 0.85]
    kp[KP.RIGHT_EAR_TIP]        = [232, 70, 0.8]

    # Nos — pośrodku
    kp[KP.NOSE_TIP]       = [150, 200, 0.95]
    kp[KP.NOSE_LEFT_WING] = [143, 196, 0.9]
    kp[KP.NOSE_RIGHT_WING]= [157, 196, 0.9]
    kp[KP.NOSE_BRIDGE]    = [150, 175, 0.9]

    # Usta
    kp[KP.MOUTH_LEFT_CORNER]  = [120, 218, 0.9]
    kp[KP.UPPER_LIP_LEFT]     = [132, 214, 0.9]
    kp[KP.UPPER_LIP_CENTER]   = [150, 212, 0.9]
    kp[KP.UPPER_LIP_RIGHT]    = [168, 214, 0.9]
    kp[KP.MOUTH_RIGHT_CORNER] = [180, 218, 0.9]
    kp[KP.LOWER_LIP_RIGHT]    = [168, 226, 0.9]
    kp[KP.LOWER_LIP_CENTER]   = [150, 228, 0.9]
    kp[KP.LOWER_LIP_LEFT]     = [132, 226, 0.9]

    # Pysk i kontur
    kp[KP.MUZZLE_TOP]        = [150, 207, 0.9]
    kp[KP.MUZZLE_LEFT]       = [122, 222, 0.85]
    kp[KP.MUZZLE_RIGHT]      = [178, 222, 0.85]
    kp[KP.CHIN]              = [150, 250, 0.85]
    kp[KP.FOREHEAD_CENTER]   = [150, 115, 0.8]
    kp[KP.FOREHEAD_LEFT]     = [120, 118, 0.8]
    kp[KP.FOREHEAD_RIGHT]    = [180, 118, 0.8]
    kp[KP.LEFT_CHEEK_UPPER]  = [88, 162, 0.8]
    kp[KP.LEFT_CHEEK_LOWER]  = [88, 202, 0.8]
    kp[KP.RIGHT_CHEEK_UPPER] = [212, 162, 0.8]
    kp[KP.RIGHT_CHEEK_LOWER] = [212, 202, 0.8]
    kp[KP.JAW_CENTER]        = [150, 245, 0.85]

    return kp.flatten()


# =============================================================================
# Testy klasy HeadPose
# =============================================================================

class TestHeadPose:
    """Testy dla klasy HeadPose."""

    def test_creation_frontal(self) -> None:
        """Test tworzenia frontalnej pozy głowy."""
        pose = HeadPose(
            yaw=5.0,
            pitch=-3.0,
            roll=2.0,
            is_frontal=True,
            confidence=0.95,
        )

        assert pose.yaw == 5.0
        assert pose.pitch == -3.0
        assert pose.roll == 2.0
        assert pose.is_frontal is True
        assert pose.confidence == 0.95

    def test_to_dict(self) -> None:
        """Test konwersji HeadPose do słownika."""
        pose = HeadPose(yaw=10.0, pitch=-5.0, roll=3.0, is_frontal=True, confidence=0.9)

        result = pose.to_dict()

        assert result["yaw"] == 10.0
        assert result["pitch"] == -5.0
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
        assert abs(pose.yaw) < 25
        assert abs(pose.pitch) < 30  # nos naturalnie poniżej oczu ~26°
        assert abs(pose.roll) < 25

    def test_left_turned_face_has_positive_yaw(self) -> None:
        """Test: twarz obrócona w lewo ma yaw > 0."""
        kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        # Przesuń nos w lewo
        kp[KP.NOSE_TIP, 0] -= 35
        kp[KP.RIGHT_EYE_INNER, 0] -= 20
        kp[KP.RIGHT_EYE_OUTER, 0] -= 20

        pose = estimate_head_pose(kp.flatten())

        assert pose.yaw > 5

    def test_right_turned_face_has_negative_yaw(self) -> None:
        """Test: twarz obrócona w prawo ma yaw < 0."""
        kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
        kp[KP.NOSE_TIP, 0] += 35
        kp[KP.LEFT_EYE_INNER, 0] += 20
        kp[KP.LEFT_EYE_OUTER, 0] += 20

        pose = estimate_head_pose(kp.flatten())

        assert pose.yaw < -5

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
        assert detector.max_yaw == 35.0
        assert detector.max_pitch == 40.0
        assert detector.max_roll == 20.0

    def test_custom_initialization(self) -> None:
        """Test inicjalizacji z niestandardowymi parametrami."""
        detector = NeutralFrameDetector(
            min_keypoint_conf=0.8,
            max_yaw=15.0,
            max_pitch=15.0,
            max_roll=10.0,
        )

        assert detector.min_keypoint_conf == 0.8
        assert detector.max_yaw == 15.0

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
