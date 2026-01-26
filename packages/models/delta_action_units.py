"""
Delta-based Action Units (AU) extractor for DogFACS.

Computes Action Units as **delta ratios** relative to a neutral baseline frame.
This allows for comparative facial expression analysis independent of breed morphology.

Key Principle: Delta_AU = (distance_target / distance_neutral) - 1.0

Official DogFACS codes implemented:
- AU101: Inner Brow Raiser
- AU102: Outer Brow Raiser
- AU12: Lip Corner Puller (smile)
- AU115: Upper Eyelid Raiser
- AU116: Lower Eyelid Raiser (squint)
- AU117: Closure of Eyelids (blink)
- AU121: Eye Widener
- EAD102: Ears Forward
- EAD103: Ears Flattener
- AD19: Tongue Show
- AD37: Nose Lick
- AU26: Jaw Drop

Scientific basis: Mota-Rojas et al. 2021, DogFACS
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import math

from packages.data.schemas import NUM_KEYPOINTS, KEYPOINT_NAMES

# Keypoint indices (for readability)
KP_LEFT_EYE = 0
KP_RIGHT_EYE = 1
KP_NOSE = 2
KP_LEFT_EAR_BASE = 3
KP_RIGHT_EAR_BASE = 4
KP_LEFT_EAR_TIP = 5
KP_RIGHT_EAR_TIP = 6
KP_LEFT_MOUTH = 7
KP_RIGHT_MOUTH = 8
KP_UPPER_LIP = 9
KP_LOWER_LIP = 10
KP_CHIN = 11
KP_LEFT_CHEEK = 12
KP_RIGHT_CHEEK = 13
KP_FOREHEAD = 14
KP_LEFT_BROW = 15
KP_RIGHT_BROW = 16
KP_MUZZLE_TOP = 17
KP_MUZZLE_LEFT = 18
KP_MUZZLE_RIGHT = 19


# Official DogFACS Action Unit names
ACTION_UNIT_NAMES = [
    "AU101",   # Inner Brow Raiser
    "AU102",   # Outer Brow Raiser
    "AU12",    # Lip Corner Puller (smile)
    "AU115",   # Upper Eyelid Raiser
    "AU116",   # Lower Eyelid Raiser (squint)
    "AU117",   # Closure of Eyelids (blink)
    "AU121",   # Eye Widener
    "EAD102",  # Ears Forward
    "EAD103",  # Ears Flattener
    "AD19",    # Tongue Show
    "AD37",    # Nose Lick
    "AU26",    # Jaw Drop
]

NUM_ACTION_UNITS = len(ACTION_UNIT_NAMES)


# Mapping from AU names to measurement keys
AU_TO_MEASUREMENT_MAP = {
    "AU101": "inner_brow_eye_dist",
    "AU102": "outer_brow_elevation",
    "AU12": "lip_corner_angle",
    "AU115": "upper_eyelid_opening",
    "AU116": "lower_eyelid_squint",
    "AU117": "eye_closure",
    "AU121": "eye_region_height",
    "EAD102": "ear_forward_angle",
    "EAD103": "ear_flattening_angle",
    "AD19": "tongue_show_proxy",
    "AD37": "nose_lick_proxy",
    "AU26": "jaw_drop_dist",
}


# Universal activation threshold (15% increase)
DEFAULT_ACTIVATION_THRESHOLD = 1.15


@dataclass
class DeltaActionUnit:
    """
    Single Action Unit with delta-based measurement.

    Attributes:
        name: Official DogFACS code (e.g., "AU101", "EAD102")
        ratio: Target/neutral ratio (1.0 = no change, 1.3 = 30% increase)
        delta: Ratio - 1.0 (0.0 = no change, 0.3 = 30% increase)
        is_active: Binary activation based on threshold
        confidence: Confidence score from keypoint visibility
    """
    name: str
    ratio: float
    delta: float
    is_active: bool
    confidence: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "ratio": self.ratio,
            "delta": self.delta,
            "is_active": self.is_active,
            "confidence": self.confidence,
        }


class DeltaActionUnitsExtractor:
    """
    Extracts delta-based Action Units from keypoints.

    Computes AUs as ratios relative to a neutral baseline frame.
    This ensures breed-invariant emotion recognition.

    Example:
        >>> neutral_kp = np.array([...])  # 60 values from neutral frame
        >>> extractor = DeltaActionUnitsExtractor(neutral_kp)
        >>> target_kp = np.array([...])   # 60 values from target frame
        >>> delta_aus = extractor.extract(target_kp)
        >>> print(delta_aus["AU101"].ratio)  # 1.25 = 25% brow raise
    """

    def __init__(
        self,
        neutral_keypoints: np.ndarray,
        activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    ):
        """
        Initialize with neutral baseline frame.

        Args:
            neutral_keypoints: Keypoints from neutral frame [x0,y0,v0,...] (60 values)
            activation_threshold: Ratio threshold for AU activation (default 1.15 = 15% increase)
        """
        if len(neutral_keypoints) != NUM_KEYPOINTS * 3:
            raise ValueError(
                f"Expected {NUM_KEYPOINTS * 3} keypoints values, "
                f"got {len(neutral_keypoints)}"
            )

        self.neutral_keypoints = neutral_keypoints.reshape(NUM_KEYPOINTS, 3)
        self.activation_threshold = activation_threshold

        # Compute and store neutral reference distances
        self.neutral_distances = self._compute_normalized_distances(self.neutral_keypoints)
        self.neutral_eye_distance = self._get_eye_distance(self.neutral_keypoints)

    def extract(self, target_keypoints: np.ndarray) -> dict[str, DeltaActionUnit]:
        """
        Extract delta AUs from target frame.

        Args:
            target_keypoints: Keypoints from target frame [x0,y0,v0,...] (60 values)

        Returns:
            Dictionary of AU_name -> DeltaActionUnit
        """
        if len(target_keypoints) != NUM_KEYPOINTS * 3:
            raise ValueError(
                f"Expected {NUM_KEYPOINTS * 3} keypoints values, "
                f"got {len(target_keypoints)}"
            )

        target_kp = target_keypoints.reshape(NUM_KEYPOINTS, 3)
        target_distances = self._compute_normalized_distances(target_kp)

        delta_aus = {}

        for au_name in ACTION_UNIT_NAMES:
            measurement_key = AU_TO_MEASUREMENT_MAP[au_name]

            neutral_val = self.neutral_distances[measurement_key]
            target_val = target_distances[measurement_key]

            # Compute ratio and delta
            if neutral_val > 1e-6:
                ratio = target_val / neutral_val
                delta = ratio - 1.0
            else:
                ratio = 1.0
                delta = 0.0

            # Determine activation
            is_active = self._is_activated(au_name, ratio, delta)

            # Confidence from keypoint visibility
            confidence = self._compute_au_confidence(au_name, target_kp)

            delta_aus[au_name] = DeltaActionUnit(
                name=au_name,
                ratio=ratio,
                delta=delta,
                is_active=is_active,
                confidence=confidence,
            )

        return delta_aus

    def _compute_normalized_distances(self, keypoints: np.ndarray) -> dict[str, float]:
        """
        Compute all measurements normalized by eye distance.

        This ensures breed-invariant measurements.

        Args:
            keypoints: Keypoints array (20, 3)

        Returns:
            Dictionary of measurement_key -> normalized_value
        """
        coords = keypoints[:, :2]  # (20, 2)
        visibility = keypoints[:, 2]  # (20,)

        # Eye distance for normalization
        eye_distance = self._distance(coords[KP_LEFT_EYE], coords[KP_RIGHT_EYE])
        if eye_distance < 1e-6:
            eye_distance = 1.0  # Fallback

        distances = {}

        # AU101: Inner Brow Raiser - distance(brow, eye)
        left_brow_eye = self._distance(coords[KP_LEFT_BROW], coords[KP_LEFT_EYE])
        right_brow_eye = self._distance(coords[KP_RIGHT_BROW], coords[KP_RIGHT_EYE])
        distances["inner_brow_eye_dist"] = (left_brow_eye + right_brow_eye) / (2 * eye_distance)

        # AU102: Outer Brow Raiser - brow elevation angle
        brow_center_y = (coords[KP_LEFT_BROW][1] + coords[KP_RIGHT_BROW][1]) / 2
        forehead_y = coords[KP_FOREHEAD][1]
        brow_elevation = abs(forehead_y - brow_center_y) / eye_distance
        distances["outer_brow_elevation"] = brow_elevation

        # AU12: Lip Corner Puller (smile) - angle of mouth corners to upper lip
        left_angle = self._angle(coords[KP_UPPER_LIP], coords[KP_LEFT_MOUTH])
        right_angle = self._angle(coords[KP_UPPER_LIP], coords[KP_RIGHT_MOUTH])
        # Normalize angle to [0, 1]
        smile_indicator = (left_angle - right_angle) / math.pi
        distances["lip_corner_angle"] = abs(smile_indicator)

        # AU115: Upper Eyelid Raiser - eye opening from visibility
        upper_eyelid = (visibility[KP_LEFT_EYE] + visibility[KP_RIGHT_EYE]) / 2
        distances["upper_eyelid_opening"] = upper_eyelid

        # AU116: Lower Eyelid Raiser (squint) - inverse of eye opening
        distances["lower_eyelid_squint"] = 1.0 - upper_eyelid

        # AU117: Eye Closure (blink) - 1 - eye visibility
        distances["eye_closure"] = 1.0 - upper_eyelid

        # AU121: Eye Widener - vertical eye region height
        eye_center_y = (coords[KP_LEFT_EYE][1] + coords[KP_RIGHT_EYE][1]) / 2
        forehead_to_eye = abs(forehead_y - eye_center_y) / eye_distance
        distances["eye_region_height"] = forehead_to_eye

        # EAD102: Ears Forward - forward angle of ears
        left_ear_angle = self._angle(coords[KP_LEFT_EAR_BASE], coords[KP_LEFT_EAR_TIP])
        right_ear_angle = self._angle(coords[KP_RIGHT_EAR_BASE], coords[KP_RIGHT_EAR_TIP])
        avg_ear_angle = (abs(left_ear_angle) + abs(right_ear_angle)) / 2
        # Forward = small angle (close to vertical)
        distances["ear_forward_angle"] = 1.0 - (avg_ear_angle / math.pi)

        # EAD103: Ears Flattener - flattening angle (inverse of forward)
        distances["ear_flattening_angle"] = avg_ear_angle / math.pi

        # AD19: Tongue Show - proxy via mouth opening and lip geometry
        mouth_opening = self._distance(coords[KP_UPPER_LIP], coords[KP_LOWER_LIP])
        jaw_drop = self._distance(coords[KP_NOSE], coords[KP_CHIN])
        # Tongue visible when mouth open but jaw not dropped excessively
        tongue_proxy = (mouth_opening / eye_distance) * (1.0 - min(1.0, jaw_drop / eye_distance))
        distances["tongue_show_proxy"] = tongue_proxy

        # AD37: Nose Lick - proxy via nose-lip distance
        nose_lip_dist = self._distance(coords[KP_NOSE], coords[KP_UPPER_LIP])
        # Licking = small distance
        distances["nose_lick_proxy"] = 1.0 - min(1.0, nose_lip_dist / eye_distance)

        # AU26: Jaw Drop - vertical nose-chin distance
        jaw_drop_normalized = jaw_drop / eye_distance
        distances["jaw_drop_dist"] = jaw_drop_normalized

        return distances

    def _is_activated(self, au_name: str, ratio: float, delta: float) -> bool:
        """
        Determine if AU is activated based on ratio threshold.

        Args:
            au_name: AU code (e.g., "AU101")
            ratio: Target/neutral ratio
            delta: Ratio - 1.0

        Returns:
            True if AU is activated (ratio > threshold)
        """
        # Special cases for bidirectional or decrease-based AUs
        if au_name in ["AU115", "AU121"]:  # Eye opening/widening - increase
            return ratio > self.activation_threshold

        elif au_name in ["AU116", "AU117"]:  # Squint/blink - can be decrease or increase
            # Activated if significantly different from neutral (bidirectional)
            return abs(delta) > (self.activation_threshold - 1.0)

        else:  # Most AUs - activation = increase
            return ratio > self.activation_threshold

    def _compute_au_confidence(self, au_name: str, keypoints: np.ndarray) -> float:
        """
        Compute confidence score for AU based on keypoint visibility.

        Args:
            au_name: AU code
            keypoints: Keypoints array (20, 3)

        Returns:
            Confidence score (0.0-1.0)
        """
        # Map AU to relevant keypoints
        keypoint_groups = {
            "AU101": [KP_LEFT_BROW, KP_RIGHT_BROW, KP_LEFT_EYE, KP_RIGHT_EYE],
            "AU102": [KP_LEFT_BROW, KP_RIGHT_BROW, KP_FOREHEAD],
            "AU12": [KP_LEFT_MOUTH, KP_RIGHT_MOUTH, KP_UPPER_LIP],
            "AU115": [KP_LEFT_EYE, KP_RIGHT_EYE],
            "AU116": [KP_LEFT_EYE, KP_RIGHT_EYE],
            "AU117": [KP_LEFT_EYE, KP_RIGHT_EYE],
            "AU121": [KP_LEFT_EYE, KP_RIGHT_EYE, KP_FOREHEAD],
            "EAD102": [KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE, KP_LEFT_EAR_TIP, KP_RIGHT_EAR_TIP],
            "EAD103": [KP_LEFT_EAR_BASE, KP_RIGHT_EAR_BASE, KP_LEFT_EAR_TIP, KP_RIGHT_EAR_TIP],
            "AD19": [KP_UPPER_LIP, KP_LOWER_LIP, KP_NOSE],
            "AD37": [KP_NOSE, KP_UPPER_LIP],
            "AU26": [KP_NOSE, KP_CHIN],
        }

        relevant_kps = keypoint_groups.get(au_name, list(range(NUM_KEYPOINTS)))
        visibilities = [keypoints[idx, 2] for idx in relevant_kps]

        return float(np.mean(visibilities))

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Euclidean distance between two points."""
        return float(np.sqrt(np.sum((p1 - p2) ** 2)))

    def _angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Angle of line p1->p2 relative to horizontal (radians)."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.atan2(dy, dx)

    def _get_eye_distance(self, keypoints: np.ndarray) -> float:
        """Get eye distance for normalization."""
        coords = keypoints[:, :2]
        return self._distance(coords[KP_LEFT_EYE], coords[KP_RIGHT_EYE])


def extract_delta_action_units(
    neutral_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> dict[str, DeltaActionUnit]:
    """
    Convenience function to extract delta AUs.

    Args:
        neutral_keypoints: Neutral baseline keypoints (60 values)
        target_keypoints: Target frame keypoints (60 values)
        activation_threshold: Activation ratio threshold (default 1.15)

    Returns:
        Dictionary of AU_name -> DeltaActionUnit
    """
    extractor = DeltaActionUnitsExtractor(neutral_keypoints, activation_threshold)
    return extractor.extract(target_keypoints)
