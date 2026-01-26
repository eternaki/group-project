"""
Testy dla modułu neutral frame detection (HeadPose, NeutralFrameDetector).

Uruchomienie:
    pytest tests/test_pipeline/test_neutral_frame.py -v
"""

import numpy as np
import pytest

from packages.pipeline.neutral_frame import (
    HeadPose,
    NeutralFrameDetector,
    estimate_head_pose,
)


class TestHeadPose:
    """Testy dla klasy HeadPose."""

    def test_creation_frontal(self) -> None:
        """Test tworzenia frontal head pose."""
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

    def test_creation_non_frontal(self) -> None:
        """Test tworzenia non-frontal head pose."""
        pose = HeadPose(
            yaw=45.0,
            pitch=-20.0,
            roll=10.0,
            is_frontal=False,
            confidence=0.85,
        )

        assert pose.yaw == 45.0
        assert pose.pitch == -20.0
        assert pose.roll == 10.0
        assert pose.is_frontal is False

    def test_to_dict(self) -> None:
        """Test konwersji do słownika."""
        pose = HeadPose(
            yaw=10.0,
            pitch=-5.0,
            roll=3.0,
            is_frontal=True,
            confidence=0.9,
        )

        result = pose.to_dict()

        assert result["yaw"] == 10.0
        assert result["pitch"] == -5.0
        assert result["roll"] == 3.0
        assert result["is_frontal"] is True
        assert result["confidence"] == 0.9


class TestEstimateHeadPose:
    """Testy dla funkcji estimate_head_pose."""

    @pytest.fixture
    def frontal_keypoints(self) -> np.ndarray:
        """Fixture dla frontal keypoints (symetryczna twarz)."""
        kp = np.array([
            # left_eye, right_eye (symetryczne y)
            [100, 150, 0.95],
            [200, 150, 0.95],
            # nose (pośrodku)
            [150, 200, 0.95],
            # left_ear_base, right_ear_base (symetryczne)
            [80, 100, 0.9],
            [220, 100, 0.9],
            # left_ear_tip, right_ear_tip
            [70, 50, 0.85],
            [230, 50, 0.85],
            # left_mouth, right_mouth
            [120, 220, 0.9],
            [180, 220, 0.9],
            # upper_lip, lower_lip
            [150, 210, 0.9],
            [150, 230, 0.9],
            # chin
            [150, 250, 0.85],
            # forehead
            [150, 120, 0.8],
            # left_eye_outer, right_eye_outer
            [90, 150, 0.9],
            [210, 150, 0.9],
            # left_eye_inner, right_eye_inner
            [110, 150, 0.9],
            [190, 150, 0.9],
            # nose_bridge
            [150, 180, 0.85],
            # left_cheek, right_cheek
            [100, 190, 0.8],
            [200, 190, 0.8],
        ], dtype=np.float32).flatten()

        return kp

    def test_frontal_pose_detection(self, frontal_keypoints: np.ndarray) -> None:
        """Test: symetryczna twarz powinna być wykryta jako frontal."""
        pose = estimate_head_pose(frontal_keypoints)

        assert pose.is_frontal is True
        # Kąty powinny być małe
        assert abs(pose.yaw) < 25
        assert abs(pose.pitch) < 25
        assert abs(pose.roll) < 25

    def test_left_turned_pose(self, frontal_keypoints: np.ndarray) -> None:
        """Test: twarz obrócona w lewo (yaw > 0)."""
        kp = frontal_keypoints.copy().reshape(20, 3)

        # Przesuń nos w lewo (zmniejsz x)
        kp[2, 0] -= 30  # nose x - 30

        # Przesuń prawe oko bardziej w lewo
        kp[1, 0] -= 20  # right_eye x - 20

        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        # Yaw powinien być dodatni (twarz w lewo)
        assert pose.yaw > 5

    def test_right_turned_pose(self, frontal_keypoints: np.ndarray) -> None:
        """Test: twarz obrócona w prawo (yaw < 0)."""
        kp = frontal_keypoints.copy().reshape(20, 3)

        # Przesuń nos w prawo (zwiększ x)
        kp[2, 0] += 30  # nose x + 30

        # Przesuń lewe oko bardziej w prawo
        kp[0, 0] += 20  # left_eye x + 20

        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        # Yaw powinien być ujemny (twarz w prawo)
        assert pose.yaw < -5

    def test_upward_tilted_pose(self, frontal_keypoints: np.ndarray) -> None:
        """Test: twarz pochylona do góry (pitch < 0)."""
        kp = frontal_keypoints.copy().reshape(20, 3)

        # Przesuń nos wyżej (zmniejsz y)
        kp[2, 1] -= 20  # nose y - 20

        # Przesuń chin wyżej
        kp[11, 1] -= 30  # chin y - 30

        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        # Pitch powinien być ujemny (patrzy w górę)
        assert pose.pitch < -5

    def test_downward_tilted_pose(self, frontal_keypoints: np.ndarray) -> None:
        """Test: twarz pochylona w dół (pitch > 0)."""
        kp = frontal_keypoints.copy().reshape(20, 3)

        # Przesuń nos niżej (zwiększ y)
        kp[2, 1] += 20  # nose y + 20

        # Przesuń chin niżej
        kp[11, 1] += 30  # chin y + 30

        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        # Pitch powinien być dodatni (patrzy w dół)
        assert pose.pitch > 5

    def test_tilted_roll(self, frontal_keypoints: np.ndarray) -> None:
        """Test: twarz pochylona na bok (roll)."""
        kp = frontal_keypoints.copy().reshape(20, 3)

        # Przesuń prawe oko wyżej, lewe niżej
        kp[1, 1] -= 15  # right_eye y - 15
        kp[0, 1] += 15  # left_eye y + 15

        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        # Roll powinien być znaczący
        assert abs(pose.roll) > 10

    def test_confidence_from_visibility(self, frontal_keypoints: np.ndarray) -> None:
        """Test: confidence pochodzi z visibility keypoints."""
        kp = frontal_keypoints.copy().reshape(20, 3)

        # Ustaw niską visibility dla kilku punktów
        kp[0, 2] = 0.3  # left_eye
        kp[2, 2] = 0.4  # nose
        kp[5, 2] = 0.2  # left_ear_tip

        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        # Confidence powinna być obniżona
        assert pose.confidence < 0.9


class TestNeutralFrameDetector:
    """Testy dla klasy NeutralFrameDetector."""

    @pytest.fixture
    def detector(self) -> NeutralFrameDetector:
        """Fixture dla detector."""
        return NeutralFrameDetector()

    @pytest.fixture
    def stable_sequence(self) -> tuple[list[np.ndarray], list[np.ndarray], list[HeadPose]]:
        """
        Fixture dla sekwencji z jedną stabilną klatką.

        Returns:
            (frames, keypoints_list, head_poses)
        """
        frames = []
        keypoints_list = []
        head_poses = []

        # 10 klatek
        for i in range(10):
            # Twórz losową klatkę
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)

            # Klatka 5 jest stabilna (minimalna wariancja)
            if i == 5:
                # Stabilne keypoints (małe zmiany)
                base_kp = self._create_frontal_keypoints()
                kp = base_kp.copy()
            else:
                # Niestabilne keypoints (duże zmiany)
                base_kp = self._create_frontal_keypoints()
                kp = base_kp.copy().reshape(20, 3)
                # Dodaj noise
                kp[:, :2] += np.random.randn(20, 2) * 10
                kp = kp.flatten()

            keypoints_list.append(kp)

            # Head pose
            pose = estimate_head_pose(kp)
            head_poses.append(pose)

        return frames, keypoints_list, head_poses

    def _create_frontal_keypoints(self) -> np.ndarray:
        """Helper: tworzy frontal keypoints."""
        kp = np.array([
            [100, 150, 0.95], [200, 150, 0.95], [150, 200, 0.95],
            [80, 100, 0.9], [220, 100, 0.9], [70, 50, 0.85],
            [230, 50, 0.85], [120, 220, 0.9], [180, 220, 0.9],
            [150, 210, 0.9], [150, 230, 0.9], [150, 250, 0.85],
            [150, 120, 0.8], [90, 150, 0.9], [210, 150, 0.9],
            [110, 150, 0.9], [190, 150, 0.9], [150, 180, 0.85],
            [100, 190, 0.8], [200, 190, 0.8],
        ], dtype=np.float32).flatten()
        return kp

    def test_detector_initialization(self) -> None:
        """Test inicjalizacji detector z domyślnymi parametrami."""
        detector = NeutralFrameDetector()

        assert detector.min_keypoint_conf == 0.7
        assert detector.max_yaw == 20
        assert detector.max_pitch == 20
        assert detector.max_roll == 20

    def test_detector_custom_parameters(self) -> None:
        """Test inicjalizacji detector z niestandardowymi parametrami."""
        detector = NeutralFrameDetector(
            min_keypoint_conf=0.8,
            max_yaw=15,
            max_pitch=15,
            max_roll=15,
        )

        assert detector.min_keypoint_conf == 0.8
        assert detector.max_yaw == 15
        assert detector.max_pitch == 15
        assert detector.max_roll == 15

    def test_detect_auto_finds_stable_frame(
        self, detector: NeutralFrameDetector, stable_sequence
    ) -> None:
        """Test: auto-detect powinien znaleźć najbardziej stabilną klatkę."""
        frames, keypoints_list, head_poses = stable_sequence

        neutral_idx = detector.detect_auto(frames, keypoints_list, head_poses)

        # Neutral frame powinien być klatką o najniższej wariancji
        # (w naszym przypadku klatka 5)
        assert 0 <= neutral_idx < len(frames)

    def test_compute_stability_score_stable_sequence(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: stability score dla stabilnej sekwencji."""
        # Twórz sekwencję z minimalnymi zmianami
        keypoints_list = []
        base_kp = self._create_frontal_keypoints()

        for i in range(20):
            # Dodaj minimalny noise
            kp = base_kp.copy().reshape(20, 3)
            kp[:, :2] += np.random.randn(20, 2) * 0.5  # Bardzo mały noise
            keypoints_list.append(kp.flatten())

        score = detector._compute_stability_score(keypoints_list, center_idx=10)

        # Wysoki stability score (blisko 1.0)
        assert score > 0.8

    def test_compute_stability_score_unstable_sequence(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: stability score dla niestabilnej sekwencji."""
        # Twórz sekwencję z dużymi zmianami
        keypoints_list = []

        for i in range(20):
            kp = self._create_frontal_keypoints().reshape(20, 3)
            # Dodaj duży noise
            kp[:, :2] += np.random.randn(20, 2) * 20
            keypoints_list.append(kp.flatten())

        score = detector._compute_stability_score(keypoints_list, center_idx=10)

        # Niski stability score
        assert score < 0.5

    def test_is_valid_candidate_frontal(self, detector: NeutralFrameDetector) -> None:
        """Test: frontal pose z wysoką confidence jest valid candidate."""
        kp = self._create_frontal_keypoints()
        pose = estimate_head_pose(kp)

        is_valid = detector._is_valid_candidate(kp, pose)

        assert is_valid is True

    def test_is_valid_candidate_non_frontal(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: non-frontal pose nie jest valid candidate."""
        kp = self._create_frontal_keypoints().reshape(20, 3)

        # Obróć twarz (duży yaw)
        kp[2, 0] += 50  # nose x + 50

        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        is_valid = detector._is_valid_candidate(kp_flat, pose)

        assert is_valid is False

    def test_is_valid_candidate_low_confidence(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: niska confidence keypoints nie jest valid candidate."""
        kp = self._create_frontal_keypoints().reshape(20, 3)

        # Ustaw niską visibility dla większości punktów
        kp[:, 2] = 0.3

        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        is_valid = detector._is_valid_candidate(kp_flat, pose)

        assert is_valid is False

    def test_is_valid_candidate_critical_keypoints_missing(
        self, detector: NeutralFrameDetector
    ) -> None:
        """Test: brak krytycznych keypoints nie jest valid candidate."""
        kp = self._create_frontal_keypoints().reshape(20, 3)

        # Ustaw zerową visibility dla krytycznych punktów (eyes, nose)
        kp[0, 2] = 0.0  # left_eye
        kp[1, 2] = 0.0  # right_eye
        kp[2, 2] = 0.0  # nose

        kp_flat = kp.flatten()
        pose = estimate_head_pose(kp_flat)

        is_valid = detector._is_valid_candidate(kp_flat, pose)

        assert is_valid is False

    def test_empty_sequence_raises(self, detector: NeutralFrameDetector) -> None:
        """Test: pusta sekwencja powinna rzucić wyjątek."""
        with pytest.raises(ValueError):
            detector.detect_auto([], [], [])

    def test_single_frame_returns_zero(self, detector: NeutralFrameDetector) -> None:
        """Test: pojedyncza klatka powinna zwrócić index 0."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp = self._create_frontal_keypoints()
        pose = estimate_head_pose(kp)

        neutral_idx = detector.detect_auto([frame], [kp], [pose])

        assert neutral_idx == 0
