"""
Strictly rule-based emotion classification for Dog FACS.

NO MACHINE LEARNING - Uses only scientific rules from DogFACS research.

Architecture:
    Keypoints (20 × 3 = 60) → Delta Action Units (12) → Rule Matching → Emotion (6 classes)

Action Units (AU) are objective measurements of facial muscle movements,
according to Dog Facial Action Coding System (DogFACS).
Emotions are interpretations of AU combinations (poselets).

Scientific basis: Mota-Rojas et al. 2021, https://www.animalfacs.com/dogfacs
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from packages.models.delta_action_units import DeltaActionUnit, ACTION_UNIT_NAMES


# Emotion classes (6 total)
EMOTION_CLASSES = ['happy', 'sad', 'angry', 'fearful', 'relaxed', 'neutral']
NUM_EMOTIONS = len(EMOTION_CLASSES)


@dataclass
class EmotionPrediction:
    """
    Result of emotion prediction.

    Attributes:
        emotion_id: Emotion class ID (0-5)
        emotion: Emotion name (happy, sad, angry, fearful, relaxed, neutral)
        confidence: Prediction confidence (0.0-1.0)
        probabilities: Probabilities for all 6 classes
        action_units: Action Unit values (DeltaActionUnit ratios or absolute values)
        rule_applied: Name of rule that matched (for transparency)
    """
    emotion_id: int
    emotion: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    action_units: dict[str, float] = field(default_factory=dict)
    rule_applied: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert prediction to dictionary."""
        result = {
            "emotion_id": self.emotion_id,
            "emotion": self.emotion,
            "emotion_confidence": self.confidence,
            "probabilities": self.probabilities,
        }
        if self.action_units:
            result["action_units"] = self.action_units
        if self.rule_applied:
            result["rule_applied"] = self.rule_applied
        return result

    def to_coco(self) -> dict:
        """
        Return data in COCO-compatible format.

        Returns:
            Dictionary with emotion and emotion_confidence
        """
        result = {
            "emotion": self.emotion,
            "emotion_confidence": self.confidence,
        }
        if self.rule_applied:
            result["emotion_rule_applied"] = self.rule_applied
        return result


@dataclass
class EmotionRule:
    """
    Single emotion classification rule (poselet matcher).

    Based on DogFACS scientific research - matches specific AU combinations.

    Attributes:
        emotion: Emotion label (e.g., "happy", "angry")
        priority: Rule priority (higher = checked first)
        required_aus: AU_name → min_ratio (must exceed for match)
        inhibitory_aus: AU_name → max_ratio (must stay below for match)
        optional_aus: AU_name → min_ratio (bonus if present)
        min_confidence: Minimum confidence threshold for rule activation
    """
    emotion: str
    priority: int
    required_aus: dict[str, float] = field(default_factory=dict)
    inhibitory_aus: dict[str, float] = field(default_factory=dict)
    optional_aus: dict[str, float] = field(default_factory=dict)
    min_confidence: float = 0.7

    def matches(self, delta_aus: dict[str, DeltaActionUnit]) -> tuple[bool, float]:
        """
        Check if rule matches current AU state.

        Args:
            delta_aus: Dictionary of AU_name → DeltaActionUnit

        Returns:
            (matches, confidence_score)
        """
        # Check required AUs
        for au_name, min_ratio in self.required_aus.items():
            if au_name not in delta_aus:
                return False, 0.0
            if delta_aus[au_name].ratio < min_ratio:
                return False, 0.0

        # Check inhibitory AUs (must stay LOW)
        for au_name, max_ratio in self.inhibitory_aus.items():
            if au_name in delta_aus:
                if delta_aus[au_name].ratio > max_ratio:
                    return False, 0.0

        # Compute confidence
        confidence = self._compute_match_confidence(delta_aus)

        return confidence >= self.min_confidence, confidence

    def _compute_match_confidence(self, delta_aus: dict[str, DeltaActionUnit]) -> float:
        """Compute match confidence based on AU confidences."""
        # Base confidence from required AUs
        if self.required_aus:
            required_confidences = [
                delta_aus[au_name].confidence
                for au_name in self.required_aus.keys()
                if au_name in delta_aus
            ]
            base_confidence = np.mean(required_confidences) if required_confidences else 0.5
        else:
            # Fallback rules (neutral, relaxed) - use overall AU confidence
            all_confidences = [au.confidence for au in delta_aus.values()]
            base_confidence = np.mean(all_confidences) if all_confidences else 0.5

        # Bonus from optional AUs
        bonus = 0.0
        for au_name, min_ratio in self.optional_aus.items():
            if au_name in delta_aus and delta_aus[au_name].ratio >= min_ratio:
                bonus += 0.05  # +5% per optional AU match

        return min(1.0, base_confidence + bonus)


# =============================================================================
# EMOTION RULES DATABASE (Based on official DogFACS codes)
# =============================================================================

EMOTION_RULES = [
    # HAPPY - Priority 100
    EmotionRule(
        emotion="happy",
        priority=100,
        required_aus={
            "AU12": 1.20,      # Lip Corner Puller (smile) 20%+
            "EAD102": 1.10,    # Ears Forward 10%+
        },
        inhibitory_aus={
            "EAD103": 1.10,    # Ears NOT flattened
            "AU26": 1.25,      # Jaw NOT excessively open (not panting)
        },
        optional_aus={
            "AU101": 1.10,     # Bonus: Inner brow raised (playful)
        },
    ),

    # ANGRY - Priority 95
    EmotionRule(
        emotion="angry",
        priority=95,
        required_aus={
            "AU26": 1.25,      # Jaw drop 25%+ (open mouth)
            "AU12": 1.15,      # Lip corners (snarl)
        },
        inhibitory_aus={
            "EAD102": 1.10,    # Ears NOT forward
        },
        optional_aus={
            "AU101": 1.15,     # Tense brows
            "EAD103": 1.10,    # Ears back/flattened
        },
    ),

    # FEARFUL - Priority 90
    EmotionRule(
        emotion="fearful",
        priority=90,
        required_aus={
            "EAD103": 1.15,    # Ears flattened 15%+
            "AU101": 1.10,     # Brows raised (tension)
        },
        inhibitory_aus={
            "AU26": 1.20,      # Mouth NOT wide open
        },
        optional_aus={
            "AD37": 1.10,      # Nose lick (stress indicator)
            "AU117": 1.15,     # Eye closure (blinking)
        },
    ),

    # SAD - Priority 85
    EmotionRule(
        emotion="sad",
        priority=85,
        required_aus={
            "EAD103": 1.10,    # Ears slightly back
        },
        inhibitory_aus={
            "AU26": 1.15,      # Mouth NOT open
            "AU12": 1.10,      # No smile
        },
        optional_aus={},
    ),

    # RELAXED - Priority 70
    EmotionRule(
        emotion="relaxed",
        priority=70,
        required_aus={},  # No strong activations required
        inhibitory_aus={
            "AU26": 1.15,      # Mouth not open
            "EAD103": 1.10,    # Ears not flattened
            "EAD102": 1.10,    # Ears not forward
            "AU101": 1.10,     # Brows not raised
        },
        optional_aus={},
    ),

    # NEUTRAL - Priority 50 (lowest, fallback)
    EmotionRule(
        emotion="neutral",
        priority=50,
        required_aus={},
        inhibitory_aus={},
        optional_aus={},
        min_confidence=0.0,  # Always matches as fallback
    ),
]


class DogFACSRuleEngine:
    """
    Rule-based emotion classifier using strict poselet matching.

    NO MACHINE LEARNING - Uses only DogFACS scientific rules.

    Example:
        >>> engine = DogFACSRuleEngine()
        >>> delta_aus = {...}  # DeltaActionUnit dictionary
        >>> prediction = engine.classify(delta_aus)
        >>> print(f"Emotion: {prediction.emotion} (rule: {prediction.rule_applied})")
    """

    def __init__(self, rules: list[EmotionRule] = None):
        """
        Initialize rule engine.

        Args:
            rules: List of EmotionRule objects (uses EMOTION_RULES if None)
        """
        self.rules = rules if rules is not None else EMOTION_RULES
        # Sort by priority (highest first)
        self.rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

    def classify(
        self,
        delta_aus: dict[str, DeltaActionUnit]
    ) -> EmotionPrediction:
        """
        Classify emotion using priority-based rule matching.

        First matching rule wins (highest priority).

        Args:
            delta_aus: Dictionary of AU_name → DeltaActionUnit

        Returns:
            EmotionPrediction with classified emotion
        """
        # Try each rule in priority order
        for rule in self.rules:
            matches, confidence = rule.matches(delta_aus)

            if matches:
                # Build probabilities dict
                probabilities = {r.emotion: 0.0 for r in self.rules}
                probabilities[rule.emotion] = confidence

                # Add partial matches with lower confidence
                for other_rule in self.rules:
                    if other_rule.emotion != rule.emotion:
                        partial_match, partial_conf = other_rule.matches(delta_aus)
                        if partial_match:
                            probabilities[other_rule.emotion] = partial_conf * 0.5

                # Normalize probabilities
                total = sum(probabilities.values())
                if total > 0:
                    probabilities = {k: v / total for k, v in probabilities.items()}

                # Build rule name for transparency
                rule_name = f"{rule.emotion}_priority_{rule.priority}"

                return EmotionPrediction(
                    emotion_id=EMOTION_CLASSES.index(rule.emotion),
                    emotion=rule.emotion,
                    confidence=confidence,
                    probabilities=probabilities,
                    action_units={au.name: au.ratio for au in delta_aus.values()},
                    rule_applied=rule_name,
                )

        # Should never reach here (neutral always matches)
        # Fallback: neutral with low confidence
        return EmotionPrediction(
            emotion_id=EMOTION_CLASSES.index("neutral"),
            emotion="neutral",
            confidence=0.3,
            probabilities={"neutral": 1.0},
            action_units={},
            rule_applied="fallback_neutral",
        )


def classify_emotion_from_delta_aus(
    delta_aus: dict[str, DeltaActionUnit],
    rule_engine: Optional[DogFACSRuleEngine] = None,
) -> EmotionPrediction:
    """
    Classify emotion from delta Action Units.

    Convenience function using DogFACSRuleEngine.

    Args:
        delta_aus: Dictionary of AU_name → DeltaActionUnit
        rule_engine: Optional custom rule engine (creates default if None)

    Returns:
        EmotionPrediction

    Example:
        >>> from packages.models.delta_action_units import extract_delta_action_units
        >>> neutral_kp = np.array([...])  # Neutral frame keypoints
        >>> target_kp = np.array([...])   # Target frame keypoints
        >>> delta_aus = extract_delta_action_units(neutral_kp, target_kp)
        >>> prediction = classify_emotion_from_delta_aus(delta_aus)
    """
    if rule_engine is None:
        rule_engine = DogFACSRuleEngine()

    return rule_engine.classify(delta_aus)


# =============================================================================
# BACKWARD COMPATIBILITY: Old rule-based classifier
# =============================================================================
# Keep for existing code that uses absolute AU values (not delta)

def classify_emotion_from_au(
    au_values: dict[str, float],
    neutral_threshold: float = 0.35,
) -> EmotionPrediction:
    """
    LEGACY: Rule-based classification from absolute AU values.

    This function uses old AU names (AU_brow_raise, AU_ear_forward, etc.)
    and weighted scoring instead of strict poselet matching.

    Kept for backward compatibility. New code should use:
    classify_emotion_from_delta_aus() with DeltaActionUnits.

    Args:
        au_values: Dictionary of AU_name → value (0.0-1.0) - OLD AU names
        neutral_threshold: Threshold below which emotion = neutral

    Returns:
        EmotionPrediction
    """
    # Map old AU names to new DogFACS codes for compatibility
    au_mapping = {
        'AU_brow_raise': 'AU101',
        'AU_eye_opening': 'AU115',
        'AU_mouth_open': 'AU12',      # Approximation
        'AU_jaw_drop': 'AU26',
        'AU_nose_wrinkle': 'AU301',   # Not in new codes, skip
        'AU_lip_corner_pull': 'AU12',
        'AU_ear_forward': 'EAD102',
        'AU_ear_back': 'EAD103',
        'AU_ear_asymmetry': 'EAD104', # Not in new codes, skip
    }

    # Get values with fallbacks
    brow_raise = au_values.get('AU_brow_raise', 0.0)
    eye_opening = au_values.get('AU_eye_opening', 0.5)
    mouth_open = au_values.get('AU_mouth_open', 0.0)
    jaw_drop = au_values.get('AU_jaw_drop', 0.0)
    nose_wrinkle = au_values.get('AU_nose_wrinkle', 0.0)
    lip_corner_pull = au_values.get('AU_lip_corner_pull', 0.0)
    ear_forward = au_values.get('AU_ear_forward', 0.0)
    ear_back = au_values.get('AU_ear_back', 0.0)
    ear_asymmetry = au_values.get('AU_ear_asymmetry', 0.0)

    blink = 1.0 - eye_opening
    nose_lick = max(0.0, mouth_open * 0.5 - jaw_drop * 0.3)

    all_au_values = [
        brow_raise, blink, mouth_open, jaw_drop, nose_wrinkle,
        lip_corner_pull, ear_forward, ear_back, ear_asymmetry, nose_lick
    ]
    mean_activation = sum(all_au_values) / len(all_au_values)

    # Scoring (weighted)
    happy_score = (
        mouth_open * 0.35 +
        ear_forward * 0.25 +
        brow_raise * 0.15 +
        (1 - ear_back) * 0.15 +
        (1 - nose_lick) * 0.10
    )

    sad_score = (
        ear_back * 0.40 +
        blink * 0.15 +
        (1 - brow_raise) * 0.15 +
        (1 - mouth_open) * 0.15 +
        nose_lick * 0.15
    )

    angry_score = (
        ((mouth_open + jaw_drop) / 2) * 0.30 +
        lip_corner_pull * 0.25 +
        nose_wrinkle * 0.15 +
        ((ear_back + ear_asymmetry) / 2) * 0.15 +
        blink * 0.15
    )

    fearful_score = (
        ear_back * 0.30 +
        nose_lick * 0.25 +
        blink * 0.20 +
        brow_raise * 0.12 +
        (1 - mouth_open) * 0.13
    )

    relaxed_score = (
        (1 - mean_activation) * 0.50 +
        (1 - ear_back) * 0.15 +
        (1 - ear_forward) * 0.15 +
        (1 - nose_lick) * 0.10 +
        (1 - nose_wrinkle) * 0.10
    )

    neutral_score = (
        (1 - mean_activation) * 0.70 +
        (1 - (mouth_open + jaw_drop) / 2) * 0.15 +
        (1 - (ear_back + ear_forward) / 2) * 0.15
    )

    scores = {
        'happy': happy_score,
        'sad': sad_score,
        'angry': angry_score,
        'fearful': fearful_score,
        'relaxed': relaxed_score,
        'neutral': neutral_score,
    }

    # Softmax normalization
    temperature = 2.0
    exp_scores = {k: np.exp(v * temperature) for k, v in scores.items()}
    total_exp = sum(exp_scores.values())
    probabilities = {k: v / total_exp for k, v in exp_scores.items()}

    best_emotion = max(probabilities, key=probabilities.get)
    best_prob = probabilities[best_emotion]

    if best_prob < neutral_threshold and best_emotion != 'neutral':
        best_emotion = 'neutral'
        best_prob = probabilities['neutral']

    emotion_id = EMOTION_CLASSES.index(best_emotion)

    return EmotionPrediction(
        emotion_id=emotion_id,
        emotion=best_emotion,
        confidence=best_prob,
        probabilities=probabilities,
        action_units=au_values,
        rule_applied="legacy_weighted_scoring",
    )


def classify_emotion_from_keypoints(
    keypoints_flat: np.ndarray,
    neutral_threshold: float = 0.35,
) -> EmotionPrediction:
    """
    LEGACY: Rule-based classification directly from keypoints.

    Uses old absolute AU extraction (not delta-based).
    Kept for backward compatibility.

    Args:
        keypoints_flat: Array [x0, y0, v0, ...] (60 values)
        neutral_threshold: Threshold for neutral

    Returns:
        EmotionPrediction
    """
    from .action_units import extract_action_units, ACTION_UNIT_NAMES

    # Extract absolute AUs (old method)
    au_prediction = extract_action_units(keypoints_flat)

    # Convert to dict
    au_dict = {
        ACTION_UNIT_NAMES[i]: float(au_prediction[i])
        for i in range(len(ACTION_UNIT_NAMES))
    }

    # Classify using legacy method
    return classify_emotion_from_au(au_dict, neutral_threshold)
