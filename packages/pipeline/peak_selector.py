"""
Peak frame selection based on Total Facial Movement (TFM).

Selects frames with maximum facial expression intensity for dataset annotation.
Uses delta Action Units to compute facial movement magnitude.
"""

from typing import Optional

import numpy as np

from packages.data.schemas import NUM_KEYPOINTS
from packages.models.delta_action_units import DeltaActionUnit
from packages.models.head_pose import DEFAULT_MAX_ROLL, DEFAULT_MAX_YAW_ASYMMETRY
from packages.pipeline.neutral_frame import HeadPose

# Weights for TFM computation (expressive AUs weighted higher)
TFM_WEIGHTS = {
    "AU26": 1.5,      # Jaw drop - very visible
    "AU12": 1.5,      # Lip corners - smile/snarl
    "EAD102": 1.2,    # Ears forward
    "EAD103": 1.2,    # Ears flattened
    "AU101": 1.1,     # Inner brow raiser
    "AU102": 1.0,     # Outer brow raiser
    "AU115": 0.8,     # Upper eyelid raiser (subtle)
    "AU116": 0.8,     # Lower eyelid raiser (subtle)
    "AU117": 0.8,     # Eye closure
    "AU121": 0.9,     # Eye widener
    "AD19": 1.0,      # Tongue show
    "AD37": 1.0,      # Nose lick
}


def compute_tfm(delta_aus: dict[str, DeltaActionUnit]) -> float:
    """
    Compute Total Facial Movement (TFM) score.

    TFM = weighted sum of AU activations (only increases counted)

    Args:
        delta_aus: Dictionary of AU_name -> DeltaActionUnit

    Returns:
        TFM score (higher = more expressive)
    """
    tfm = 0.0

    for au_name, au in delta_aus.items():
        weight = TFM_WEIGHTS.get(au_name, 1.0)

        # Only count increases (positive deltas)
        # This focuses on activations, not decreases
        if au.delta > 0:
            tfm += weight * au.delta

    return float(tfm)


class PeakFrameSelector:
    """
    Selects peak expression frames from video sequence.

    Peak frames are selected based on:
    1. High TFM (Total Facial Movement) score
    2. Minimum temporal separation (avoid consecutive frames)
    3. Valid head pose and keypoint confidence

    Example:
        >>> selector = PeakFrameSelector(min_separation_frames=30)
        >>> peak_indices = selector.select(
        ...     frames=frames_list,
        ...     keypoints_list=keypoints_list,
        ...     neutral_idx=neutral_frame_idx,
        ...     delta_aus_list=delta_aus_list,
        ...     head_poses=head_poses,
        ...     num_peaks=10,
        ... )
        >>> print(f"Selected {len(peak_indices)} peak frames")
    """

    def __init__(
        self,
        min_separation_frames: int = 30,  # 1 second @ 30fps
        min_tfm_threshold: float = 0.15,   # Minimum movement
        frontal_only: bool = False,  # Zmieniono na False - zbyt restrykcyjne
        min_keypoint_conf: float = 0.5,  # Zmniejszono z 0.7 na 0.5
        max_yaw_asymmetry: float = DEFAULT_MAX_YAW_ASYMMETRY,
        max_roll: float = DEFAULT_MAX_ROLL,
        min_sharpness: float = 60.0,  # Min. ostrość mordy (var Laplacian) — filtr rozmycia
    ):
        """
        Initialize peak frame selector.

        Args:
            min_separation_frames: Minimum frames between selected peaks
            min_tfm_threshold: Minimum TFM score to consider
            frontal_only: Only select poses within max_yaw_asymmetry and max_roll
            min_keypoint_conf: Minimum keypoint confidence
            max_yaw_asymmetry: Maximum yaw asymmetry (eye corner <-> nose)
            max_roll: Maximum roll angle in degrees
        """
        self.min_separation = min_separation_frames
        self.min_tfm = min_tfm_threshold
        self.frontal_only = frontal_only
        self.min_kp_conf = min_keypoint_conf
        self.max_yaw_asymmetry = max_yaw_asymmetry
        self.max_roll = max_roll
        self.min_sharpness = min_sharpness

    def select(
        self,
        frames: list[np.ndarray],
        keypoints_list: list[np.ndarray],
        neutral_idx: int,
        delta_aus_list: list[dict[str, DeltaActionUnit]],
        head_poses: Optional[list[HeadPose]] = None,
        num_peaks: int = 10,
    ) -> list[int]:
        """
        Select peak expression frames.

        Args:
            frames: List of video frames
            keypoints_list: List of keypoints (60 values each)
            neutral_idx: Index of neutral baseline frame
            delta_aus_list: List of delta AU dictionaries for each frame
            head_poses: Optional list of HeadPose objects
            num_peaks: Number of peak frames to select

        Returns:
            Wybrane pozycje, od najwyższego TFM. Może być ich MNIEJ niż `num_peaks`
            — separacja `min_separation_frames` jest twarda i ma pierwszeństwo
            przed zamówioną liczbą.
        """
        # Estimate head poses if not provided
        if head_poses is None:
            from packages.pipeline.neutral_frame import estimate_head_pose
            # Progi z konstruktora muszą trafić do estymatora — inaczej
            # `is_frontal` (używane przy frontal_only) liczy się na domyślnych.
            head_poses = [
                estimate_head_pose(
                    kp,
                    max_yaw_asymmetry=self.max_yaw_asymmetry,
                    max_roll=self.max_roll,
                )
                for kp in keypoints_list
            ]

        # Step 1: Compute TFM for valid candidates.
        # Zbieramy WSZYSTKIE poprawne kadry (z TFM). Kadry powyżej progu min_tfm
        # są preferowane, ale jeśli jest ich za mało (np. spokojny pies), dobieramy
        # z pozostałych poprawnych — żeby uszanować żądaną liczbę peaków.
        strong: list[tuple[int, float]] = []
        valid_all: list[tuple[int, float]] = []
        for i, delta_aus in enumerate(delta_aus_list):
            if i == neutral_idx or delta_aus is None:
                continue
            frame_i = frames[i] if i < len(frames) else None
            if not self._is_valid_peak(keypoints_list[i], head_poses[i], frame_i):
                continue
            tfm = compute_tfm(delta_aus)
            valid_all.append((i, tfm))
            if tfm >= self.min_tfm:
                strong.append((i, tfm))

        tfm_scores = strong if len(strong) >= num_peaks else valid_all

        # Step 2: Sort by TFM (descending)
        tfm_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Non-maximum suppression z TWARDĄ separacją.
        # Wcześniej separacja była adaptacyjna — połowiona aż do 1, gdy nie
        # uzbierało się `num_peaks`. Przy zamówionych 5 klatkach i 12 peakach
        # wychodziły kadry sąsiadujące (odstępy same jedynki), czyli dokładnie
        # duplikaty, przed którymi separacja miała bronić. Przy naszym rozmiarze
        # zbioru duplikat jest gorszy niż brak próbki: zawyża pozorną liczebność
        # i wagę jednej chwili. Wideo bez dość odległych szczytów daje ich mniej.
        selected_indices: list[int] = []
        for idx, _ in tfm_scores:
            if all(abs(idx - selected) >= self.min_separation for selected in selected_indices):
                selected_indices.append(idx)
            if len(selected_indices) >= num_peaks:
                break

        return selected_indices

    def select_diverse_peaks(
        self,
        frames: list[np.ndarray],
        keypoints_list: list[np.ndarray],
        neutral_idx: int,
        delta_aus_list: list[dict[str, DeltaActionUnit]],
        emotions: list[str],
        head_poses: Optional[list[HeadPose]] = None,
        num_peaks: int = 10,
    ) -> list[int]:
        """
        Select peak frames ensuring emotional diversity.

        Selects top-N frames per emotion class to ensure varied dataset.

        Args:
            frames: List of video frames
            keypoints_list: List of keypoints
            neutral_idx: Neutral frame index
            delta_aus_list: Delta AUs for each frame
            emotions: Classified emotion for each frame
            head_poses: Optional head poses
            num_peaks: Total number of peaks to select

        Returns:
            List of selected frame indices with diverse emotions
        """
        # Estimate head poses if needed
        if head_poses is None:
            from packages.pipeline.neutral_frame import estimate_head_pose
            # Progi z konstruktora muszą trafić do estymatora — inaczej
            # `is_frontal` (używane przy frontal_only) liczy się na domyślnych.
            head_poses = [
                estimate_head_pose(
                    kp,
                    max_yaw_asymmetry=self.max_yaw_asymmetry,
                    max_roll=self.max_roll,
                )
                for kp in keypoints_list
            ]

        # Group candidates by emotion
        emotion_groups = {}
        for i, (delta_aus, emotion) in enumerate(zip(delta_aus_list, emotions)):
            if i == neutral_idx:
                continue

            # Skip frames without delta AUs
            if delta_aus is None:
                continue

            frame_i = frames[i] if i < len(frames) else None
            if not self._is_valid_peak(keypoints_list[i], head_poses[i], frame_i):
                continue

            tfm = compute_tfm(delta_aus)
            if tfm < self.min_tfm:
                continue

            if emotion not in emotion_groups:
                emotion_groups[emotion] = []
            emotion_groups[emotion].append((i, tfm))

        # Sort each emotion group by TFM
        for emotion in emotion_groups:
            emotion_groups[emotion].sort(key=lambda x: x[1], reverse=True)

        # Select top-K from each emotion (balanced)
        selected = []
        per_emotion = max(1, num_peaks // len(emotion_groups))

        for emotion, candidates in emotion_groups.items():
            # Apply temporal separation within emotion group
            emotion_selected = []
            for idx, tfm in candidates:
                if self._is_separated(idx, emotion_selected):
                    emotion_selected.append(idx)

                if len(emotion_selected) >= per_emotion:
                    break

            selected.extend(emotion_selected)

        # If we don't have enough, fill with highest TFM regardless of emotion
        if len(selected) < num_peaks:
            all_candidates = []
            for emotion_cands in emotion_groups.values():
                all_candidates.extend(emotion_cands)
            all_candidates.sort(key=lambda x: x[1], reverse=True)

            for idx, tfm in all_candidates:
                if idx not in selected and self._is_separated(idx, selected):
                    selected.append(idx)

                if len(selected) >= num_peaks:
                    break

        return selected[:num_peaks]

    def _is_valid_peak(
        self,
        keypoints: Optional[np.ndarray],
        head_pose: Optional[HeadPose],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Check if frame is valid for peak selection.

        Args:
            keypoints: Keypoints array (60 values) or None
            head_pose: Head pose estimation or None
            frame: Pełna klatka (do oceny ostrości mordy)

        Returns:
            True if valid peak candidate
        """
        # None check - skip frames without keypoints
        if keypoints is None or head_pose is None:
            return False

        kp = keypoints.reshape(NUM_KEYPOINTS, 3)

        # 1. Minimum keypoint confidence
        mean_visibility = np.mean(kp[:, 2])
        if mean_visibility < self.min_kp_conf:
            return False

        # 2. Head pose check
        if self.frontal_only:
            # Strict frontal: all angles < 20°
            if not head_pose.is_frontal:
                return False
        else:
            # Relaxed: just check max threshold
            if abs(head_pose.yaw_asymmetry) > self.max_yaw_asymmetry:
                return False
            if abs(head_pose.roll) > self.max_roll:
                return False

        # 3. Morda przycięta krawędzią kadru — keypoints "przyklejone" do brzegu
        # oznaczają, że część mordy wyszła poza obraz. Odrzucamy.
        if frame is not None:
            fh, fw = frame.shape[:2]
            vis = kp[:, 2] > 0.1
            if vis.sum() >= 4:
                xs, ys = kp[vis, 0], kp[vis, 1]
                m = max(8.0, 0.03 * min(fw, fh))  # margines ~3% (morda z zapasem od krawędzi)
                if (
                    xs.min() <= m
                    or ys.min() <= m
                    or xs.max() >= fw - m
                    or ys.max() >= fh - m
                ):
                    return False

        # 4. Ostrość mordy (filtr rozmycia ruchu) — var Laplacian na cropie mordy
        if frame is not None and self.min_sharpness > 0:
            if self._face_sharpness(frame, kp) < self.min_sharpness:
                return False

        return True

    @staticmethod
    def _face_sharpness(frame: np.ndarray, kp: np.ndarray) -> float:
        """Ostrość regionu mordy = wariancja Laplaciana (niska = rozmycie)."""
        import cv2

        h, w = frame.shape[:2]
        vis = kp[:, 2] > 0.1
        if vis.sum() < 4:
            return 0.0
        xs, ys = kp[vis, 0], kp[vis, 1]
        x0, y0 = max(0, int(xs.min())), max(0, int(ys.min()))
        x1, y1 = min(w, int(xs.max())), min(h, int(ys.max()))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return 0.0
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _is_separated(self, idx: int, selected: list[int]) -> bool:
        """
        Check if frame is temporally separated from selected frames.

        Args:
            idx: Frame index to check
            selected: List of already selected frame indices

        Returns:
            True if frame is far enough from all selected frames
        """
        for sel_idx in selected:
            if abs(idx - sel_idx) < self.min_separation:
                return False
        return True
