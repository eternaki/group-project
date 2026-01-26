"""
Neutral frame detection for delta-based AU calculation.

Detects the "neutral baseline" frame in a video sequence where the dog's
facial expression is most relaxed and stable (minimal movement).

This neutral frame serves as the reference point for computing delta AUs.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from packages.data.schemas import NUM_KEYPOINTS


# Keypoint indices for critical points
KP_LEFT_EYE = 0
KP_RIGHT_EYE = 1
KP_NOSE = 2
KP_LEFT_EAR_BASE = 3
KP_RIGHT_EAR_BASE = 4


@dataclass
class HeadPose:
    """
    Head pose estimation from keypoints.

    Attributes:
        yaw: Rotation around vertical axis (degrees)
        pitch: Rotation around horizontal axis (degrees)
        roll: Rotation around depth axis (degrees)
        is_frontal: Whether pose is frontal (all angles < threshold)
        confidence: Confidence of pose estimation
    """
    yaw: float
    pitch: float
    roll: float
    is_frontal: bool
    confidence: float


def estimate_head_pose(keypoints: np.ndarray) -> HeadPose:
    """
    Estimate head pose from keypoints.

    Simple heuristic-based estimation using eye, ear, and nose positions.

    Args:
        keypoints: Keypoints array [x0,y0,v0,...] (60 values) or (20, 3)

    Returns:
        HeadPose estimation
    """
    if len(keypoints.shape) == 1:
        kp = keypoints.reshape(NUM_KEYPOINTS, 3)
    else:
        kp = keypoints

    coords = kp[:, :2]
    visibility = kp[:, 2]

    # YAW: Left-right rotation
    # Based on nose position relative to eyes
    eye_center_x = (coords[KP_LEFT_EYE][0] + coords[KP_RIGHT_EYE][0]) / 2
    nose_x = coords[KP_NOSE][0]
    eye_distance = np.linalg.norm(coords[KP_LEFT_EYE] - coords[KP_RIGHT_EYE])

    if eye_distance > 1e-6:
        yaw_ratio = (nose_x - eye_center_x) / eye_distance
        yaw = np.arctan(yaw_ratio) * (180 / np.pi)  # Convert to degrees
        yaw = np.clip(yaw, -45, 45)
    else:
        yaw = 0.0

    # PITCH: Up-down rotation
    # Based on nose position relative to ears
    ear_center_y = (coords[KP_LEFT_EAR_BASE][1] + coords[KP_RIGHT_EAR_BASE][1]) / 2
    nose_y = coords[KP_NOSE][1]

    if eye_distance > 1e-6:
        pitch_ratio = (nose_y - ear_center_y) / eye_distance
        pitch = np.arctan(pitch_ratio) * (180 / np.pi)
        pitch = np.clip(pitch, -30, 30)
    else:
        pitch = 0.0

    # ROLL: Tilt rotation
    # Based on eye alignment
    left_eye_y = coords[KP_LEFT_EYE][1]
    right_eye_y = coords[KP_RIGHT_EYE][1]
    dy = right_eye_y - left_eye_y
    dx = coords[KP_RIGHT_EYE][0] - coords[KP_LEFT_EYE][0]

    if abs(dx) > 1e-6:
        roll = np.arctan2(dy, dx) * (180 / np.pi)
        roll = np.clip(roll, -30, 30)
    else:
        roll = 0.0

    # Is frontal? (all angles within threshold)
    is_frontal = abs(yaw) < 20 and abs(pitch) < 20 and abs(roll) < 20

    # Confidence from keypoint visibility
    critical_kps = [KP_LEFT_EYE, KP_RIGHT_EYE, KP_NOSE, KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE]
    confidence = float(np.mean([visibility[idx] for idx in critical_kps]))

    return HeadPose(
        yaw=float(yaw),
        pitch=float(pitch),
        roll=float(roll),
        is_frontal=is_frontal,
        confidence=confidence,
    )


class NeutralFrameDetector:
    """
    Detects neutral baseline frame from video sequence.

    The neutral frame should have:
    1. Minimal keypoint variance (stable, no movement)
    2. Frontal head pose
    3. High keypoint confidence
    4. Critical keypoints visible

    Example:
        >>> detector = NeutralFrameDetector()
        >>> frames = [frame1, frame2, ...]  # Video frames
        >>> keypoints_list = [kp1, kp2, ...]  # Keypoints for each frame
        >>> head_poses = [pose1, pose2, ...]  # Head poses
        >>> neutral_idx = detector.detect_auto(frames, keypoints_list, head_poses)
        >>> print(f"Neutral frame: {neutral_idx}")
    """

    def __init__(
        self,
        window_size: int = 10,
        variance_threshold: float = 0.02,
        min_confidence: float = 0.7,
        frontal_yaw_threshold: float = 20.0,
        frontal_pitch_threshold: float = 20.0,
    ):
        """
        Initialize detector.

        Args:
            window_size: Size of sliding window for stability computation
            variance_threshold: Max variance for stable frame
            min_confidence: Minimum keypoint confidence
            frontal_yaw_threshold: Max yaw angle for frontal (degrees)
            frontal_pitch_threshold: Max pitch angle for frontal (degrees)
        """
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.min_confidence = min_confidence
        self.frontal_yaw_threshold = frontal_yaw_threshold
        self.frontal_pitch_threshold = frontal_pitch_threshold

    def detect_auto(
        self,
        frames: list[np.ndarray],
        keypoints_list: list[np.ndarray],
        head_poses: Optional[list[HeadPose]] = None,
    ) -> int:
        """
        Auto-detect neutral frame from video sequence.

        Args:
            frames: List of video frames (not used currently, for future)
            keypoints_list: List of keypoints arrays (60 values each)
            head_poses: Optional list of HeadPose objects (computed if None)

        Returns:
            Index of neutral frame

        Raises:
            ValueError: If no valid candidates found
        """
        # Estimate head poses if not provided
        if head_poses is None:
            head_poses = [estimate_head_pose(kp) for kp in keypoints_list]

        # Step 1: Filter valid candidates
        candidates = []
        for i in range(len(frames)):
            if self._is_valid_candidate(keypoints_list[i], head_poses[i]):
                candidates.append(i)

        if not candidates:
            raise ValueError(
                "No valid neutral frame candidates found. "
                "Check if video has frontal, high-confidence frames."
            )

        # Step 2: Compute stability score for each candidate
        scores = []
        for idx in candidates:
            stability = self._compute_stability_score(keypoints_list, idx)
            scores.append((idx, stability))

        # Step 3: Return frame with highest stability (lowest variance)
        best_idx, best_score = max(scores, key=lambda x: x[1])
        return best_idx

    def detect_manual(self, frame_idx: int) -> int:
        """
        Manual neutral frame selection.

        Args:
            frame_idx: User-selected frame index

        Returns:
            The same frame_idx (for consistency with auto detection)
        """
        return frame_idx

    def _is_valid_candidate(
        self,
        keypoints: np.ndarray,
        head_pose: HeadPose,
    ) -> bool:
        """
        Check if frame is valid neutral candidate.

        Args:
            keypoints: Keypoints array (60 values)
            head_pose: Head pose estimation

        Returns:
            True if valid candidate
        """
        # Reshape keypoints
        kp = keypoints.reshape(NUM_KEYPOINTS, 3)

        # 1. Frontal head pose
        if not head_pose.is_frontal:
            return False
        if abs(head_pose.yaw) > self.frontal_yaw_threshold:
            return False
        if abs(head_pose.pitch) > self.frontal_pitch_threshold:
            return False

        # 2. High overall keypoint confidence
        mean_visibility = np.mean(kp[:, 2])
        if mean_visibility < self.min_confidence:
            return False

        # 3. Critical keypoints must be visible
        critical_kps = [KP_LEFT_EYE, KP_RIGHT_EYE, KP_NOSE,
                        KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE]
        for kp_idx in critical_kps:
            if kp[kp_idx, 2] < 0.5:
                return False

        return True

    def _compute_stability_score(
        self,
        keypoints_list: list[np.ndarray],
        center_idx: int,
    ) -> float:
        """
        Compute stability score for a frame.

        Stability = 1 / (1 + variance)

        Higher stability = more likely neutral frame.

        Args:
            keypoints_list: List of all keypoints
            center_idx: Index of frame to evaluate

        Returns:
            Stability score (higher = more stable)
        """
        # Extract window around center frame
        start = max(0, center_idx - self.window_size // 2)
        end = min(len(keypoints_list), center_idx + self.window_size // 2 + 1)

        window_kps = []
        for kp in keypoints_list[start:end]:
            kp_reshaped = kp.reshape(NUM_KEYPOINTS, 3)
            # Only use x,y coordinates (ignore visibility)
            window_kps.append(kp_reshaped[:, :2])

        # Convert to numpy array: (window_size, 20, 2)
        coords_array = np.array(window_kps)

        # Compute variance of each keypoint across time
        variance_per_kp = np.var(coords_array, axis=0)  # (20, 2)
        mean_variance = np.mean(variance_per_kp)

        # Stability score: inverse of variance
        # Higher variance = lower stability
        stability = 1.0 / (1.0 + mean_variance * 100)

        return float(stability)
