"""
Peak frame selection based on Total Facial Movement (TFM).

Selects frames with maximum facial expression intensity for dataset annotation.
Uses delta Action Units to compute facial movement magnitude.
"""

from typing import Optional
import numpy as np

from packages.data.schemas import NUM_KEYPOINTS
from packages.models.delta_action_units import DeltaActionUnit
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
        max_head_angle: float = 40.0,  # Maksymalny kąt yaw/pitch (nowy parametr)
    ):
        """
        Initialize peak frame selector.

        Args:
            min_separation_frames: Minimum frames between selected peaks
            min_tfm_threshold: Minimum TFM score to consider
            frontal_only: Only select strictly frontal poses (<20°)
            min_keypoint_conf: Minimum keypoint confidence
            max_head_angle: Maximum yaw/pitch angle in degrees (used when frontal_only=False)
        """
        self.min_separation = min_separation_frames
        self.min_tfm = min_tfm_threshold
        self.frontal_only = frontal_only
        self.min_kp_conf = min_keypoint_conf
        self.max_head_angle = max_head_angle

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
            List of selected frame indices (sorted by TFM, descending)
        """
        # Estimate head poses if not provided
        if head_poses is None:
            from packages.pipeline.neutral_frame import estimate_head_pose
            head_poses = [estimate_head_pose(kp) for kp in keypoints_list]

        # Step 1: Compute TFM for each frame and filter valid candidates
        tfm_scores = []
        for i, delta_aus in enumerate(delta_aus_list):
            # Skip neutral frame itself
            if i == neutral_idx:
                continue

            # Skip frames without delta AUs (None)
            if delta_aus is None:
                continue

            # Check if valid peak candidate
            if not self._is_valid_peak(keypoints_list[i], head_poses[i]):
                continue

            # Compute TFM
            tfm = compute_tfm(delta_aus)

            # Filter by minimum TFM threshold
            if tfm >= self.min_tfm:
                tfm_scores.append((i, tfm))

        # Step 2: Sort by TFM (descending)
        tfm_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Non-maximum suppression - enforce temporal separation
        selected_indices = []
        for idx, tfm in tfm_scores:
            # Check separation from already selected frames
            if self._is_separated(idx, selected_indices):
                selected_indices.append(idx)

            # Stop when we have enough peaks
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
            head_poses = [estimate_head_pose(kp) for kp in keypoints_list]

        # Group candidates by emotion
        emotion_groups = {}
        for i, (delta_aus, emotion) in enumerate(zip(delta_aus_list, emotions)):
            if i == neutral_idx:
                continue

            # Skip frames without delta AUs
            if delta_aus is None:
                continue

            if not self._is_valid_peak(keypoints_list[i], head_poses[i]):
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
    ) -> bool:
        """
        Check if frame is valid for peak selection.

        Args:
            keypoints: Keypoints array (60 values) or None
            head_pose: Head pose estimation or None

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
            # Relaxed: just check max angle threshold
            if abs(head_pose.yaw) > self.max_head_angle:
                return False
            if abs(head_pose.pitch) > self.max_head_angle:
                return False

        return True

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
