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


def estimate_head_pose(keypoints: Optional[np.ndarray]) -> Optional[HeadPose]:
    """
    Estimate head pose from keypoints.

    Simple heuristic-based estimation using eye, ear, and nose positions.

    Args:
        keypoints: Keypoints array [x0,y0,v0,...] (60 values) or (20, 3), or None

    Returns:
        HeadPose estimation or None if keypoints is None
    """
    # None check
    if keypoints is None:
        return None

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
    # Używamy 40° dla yaw/pitch (bardziej tolerancyjne dla realistycznych wideo)
    # Roll pozostaje 20° (przechył głowy jest mniej akceptowalny)
    is_frontal = abs(yaw) < 40 and abs(pitch) < 40 and abs(roll) < 20

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
        min_confidence: float = 0.5,  # Zmniejszono z 0.7 (zbyt restrykcyjne)
        frontal_yaw_threshold: float = 40.0,  # Zwiększono z 20.0
        frontal_pitch_threshold: float = 40.0,  # Zwiększono z 20.0
    ):
        """
        Initialize detector.

        Args:
            window_size: Size of sliding window for stability computation
            variance_threshold: Max variance for stable frame
            min_confidence: Minimum keypoint confidence (default 0.5)
            frontal_yaw_threshold: Max yaw angle for frontal (default 40°)
            frontal_pitch_threshold: Max pitch angle for frontal (default 40°)
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
        debug: bool = False,
    ) -> int:
        """
        Auto-detect neutral frame from video sequence.

        Args:
            frames: List of video frames (not used currently, for future)
            keypoints_list: List of keypoints arrays (60 values each)
            head_poses: Optional list of HeadPose objects (computed if None)
            debug: Enable debug logging to see why frames are rejected

        Returns:
            Index of neutral frame

        Raises:
            ValueError: If no valid candidates found
        """
        # Estimate head poses if not provided
        if head_poses is None:
            head_poses = [estimate_head_pose(kp) for kp in keypoints_list]

        # Step 1: Filter valid candidates (strict criteria)
        candidates = []
        if debug:
            print(f"  Checking {len(frames)} frames for ideal neutral candidate...")

        for i in range(len(frames)):
            if self._is_valid_candidate(
                keypoints_list[i], head_poses[i], frame_idx=i, debug=debug
            ):
                candidates.append(i)

        if candidates:
            print(f"  → Found {len(candidates)} ideal neutral candidates")

        # If no strict candidates, use relaxed criteria (best available)
        if not candidates:
            print("⚠️  No ideal neutral frames found. Using best available frame...")
            candidates = self._get_relaxed_candidates(keypoints_list, head_poses)

            if not candidates:
                # Last resort: find any frame with keypoints
                for i, kp in enumerate(keypoints_list):
                    if kp is not None:
                        kp_reshaped = kp.reshape(NUM_KEYPOINTS, 3)
                        if np.mean(kp_reshaped[:, 2]) > 0.3:  # At least 30% visible
                            print(f"  → Using fallback frame: {i}")
                            return i

                raise ValueError(
                    "No valid neutral frame candidates found. "
                    "Video may have too few keypoints detected."
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
        keypoints: Optional[np.ndarray],
        head_pose: Optional[HeadPose],
        frame_idx: int = -1,
        debug: bool = False,
    ) -> bool:
        """
        Check if frame is valid neutral candidate (strict criteria).

        Args:
            keypoints: Keypoints array (60 values) or None
            head_pose: Head pose estimation or None
            frame_idx: Frame index for debug logging
            debug: Enable debug logging

        Returns:
            True if valid candidate
        """
        # None check
        if keypoints is None or head_pose is None:
            return False

        # Reshape keypoints
        kp = keypoints.reshape(NUM_KEYPOINTS, 3)

        # 1. Frontal head pose
        if not head_pose.is_frontal:
            if debug:
                print(f"  Frame {frame_idx}: rejected - not frontal "
                      f"(yaw={head_pose.yaw:.1f}, pitch={head_pose.pitch:.1f})")
            return False

        # 2. Overall keypoint confidence (relaxed to 0.3)
        mean_visibility = np.mean(kp[:, 2])
        if mean_visibility < 0.3:  # Zmniejszono z min_confidence (było 0.5)
            if debug:
                print(f"  Frame {frame_idx}: rejected - low visibility "
                      f"({mean_visibility:.2f} < 0.3)")
            return False

        # 3. Critical keypoints must be visible (relaxed to 0.3)
        critical_kps = [KP_LEFT_EYE, KP_RIGHT_EYE, KP_NOSE,
                        KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE]
        critical_names = ["left_eye", "right_eye", "nose", "left_ear", "right_ear"]
        for kp_idx, kp_name in zip(critical_kps, critical_names):
            if kp[kp_idx, 2] < 0.3:  # Zmniejszono z 0.5
                if debug:
                    print(f"  Frame {frame_idx}: rejected - {kp_name} not visible "
                          f"({kp[kp_idx, 2]:.2f} < 0.3)")
                return False

        if debug:
            print(f"  Frame {frame_idx}: VALID candidate "
                  f"(vis={mean_visibility:.2f}, yaw={head_pose.yaw:.1f})")

        return True

    def _get_relaxed_candidates(
        self,
        keypoints_list: list[Optional[np.ndarray]],
        head_poses: list[Optional[HeadPose]],
    ) -> list[int]:
        """
        Get candidates with relaxed criteria (fallback when no ideal frames).

        Relaxed criteria:
        - Yaw/pitch up to 40° (instead of 20°)
        - Min confidence 0.4 (instead of 0.7)
        - At least 3 critical keypoints visible

        Args:
            keypoints_list: List of keypoints (may contain None)
            head_poses: List of head poses (may contain None)

        Returns:
            List of candidate indices
        """
        candidates = []

        for i in range(len(keypoints_list)):
            # Skip None keypoints
            if keypoints_list[i] is None or head_poses[i] is None:
                continue

            kp = keypoints_list[i].reshape(NUM_KEYPOINTS, 3)
            pose = head_poses[i]

            # Relaxed head pose (40° instead of 20°)
            if abs(pose.yaw) > 40 or abs(pose.pitch) > 40:
                continue

            # Relaxed confidence (40% instead of 70%)
            mean_visibility = np.mean(kp[:, 2])
            if mean_visibility < 0.4:
                continue

            # At least 3 out of 5 critical keypoints visible
            critical_kps = [KP_LEFT_EYE, KP_RIGHT_EYE, KP_NOSE,
                            KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE]
            visible_count = sum(1 for idx in critical_kps if kp[idx, 2] > 0.4)
            if visible_count < 3:
                continue

            candidates.append(i)

        return candidates

    def _compute_stability_score(
        self,
        keypoints_list: list[Optional[np.ndarray]],
        center_idx: int,
    ) -> float:
        """
        Compute stability score for a frame.

        Stability = 1 / (1 + variance)

        Higher stability = more likely neutral frame.

        Args:
            keypoints_list: List of all keypoints (may contain None)
            center_idx: Index of frame to evaluate

        Returns:
            Stability score (higher = more stable)
        """
        # Extract window around center frame
        start = max(0, center_idx - self.window_size // 2)
        end = min(len(keypoints_list), center_idx + self.window_size // 2 + 1)

        window_kps = []
        for kp in keypoints_list[start:end]:
            # Skip None keypoints in window
            if kp is None:
                continue
            kp_reshaped = kp.reshape(NUM_KEYPOINTS, 3)
            # Only use x,y coordinates (ignore visibility)
            window_kps.append(kp_reshaped[:, :2])

        # If too few valid keypoints in window, return low stability
        if len(window_kps) < 2:
            return 0.0

        # Convert to numpy array: (window_size, 20, 2)
        coords_array = np.array(window_kps)

        # Compute variance of each keypoint across time
        variance_per_kp = np.var(coords_array, axis=0)  # (20, 2)
        mean_variance = np.mean(variance_per_kp)

        # Stability score: inverse of variance
        # Higher variance = lower stability
        stability = 1.0 / (1.0 + mean_variance * 100)

        return float(stability)
