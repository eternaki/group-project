"""
Testy dla rule-based emotion classification (EmotionRule, DogFACSRuleEngine).

Uruchomienie:
    pytest tests/test_models/test_emotion_rules.py -v
"""

import numpy as np
import pytest

from packages.models.emotion import (
    EmotionRule,
    DogFACSRuleEngine,
    EmotionPrediction,
    EMOTION_RULES,
    EMOTION_CLASSES,
    classify_emotion_from_delta_aus,
)
from packages.models.delta_action_units import DeltaActionUnit


class TestEmotionRule:
    """Testy dla klasy EmotionRule."""

    def test_creation(self) -> None:
        """Test tworzenia EmotionRule."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20, "EAD102": 1.10},
            inhibitory_aus={"EAD103": 1.10},
            optional_aus={"AU101": 1.10},
            min_confidence=0.7,
        )

        assert rule.emotion == "happy"
        assert rule.priority == 100
        assert len(rule.required_aus) == 2
        assert len(rule.inhibitory_aus) == 1
        assert len(rule.optional_aus) == 1

    def test_matches_success(self) -> None:
        """Test: rule match gdy wszystkie required AU są spełnione."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20, "EAD102": 1.10},
            inhibitory_aus={"EAD103": 1.10},
        )

        # Twórz delta AUs spełniające rule
        delta_aus = {
            "AU12": DeltaActionUnit("AU12", 1.30, 0.30, True, 0.9),
            "EAD102": DeltaActionUnit("EAD102", 1.15, 0.15, True, 0.85),
            "EAD103": DeltaActionUnit("EAD103", 1.05, 0.05, False, 0.9),
        }

        matches, confidence = rule.matches(delta_aus)

        assert matches is True
        assert confidence >= 0.7

    def test_matches_fail_required_au_low(self) -> None:
        """Test: rule nie match gdy required AU jest za niskie."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20, "EAD102": 1.10},
        )

        # AU12 za niskie (1.10 < 1.20)
        delta_aus = {
            "AU12": DeltaActionUnit("AU12", 1.10, 0.10, False, 0.9),
            "EAD102": DeltaActionUnit("EAD102", 1.15, 0.15, True, 0.85),
        }

        matches, confidence = rule.matches(delta_aus)

        assert matches is False

    def test_matches_fail_inhibitory_au_high(self) -> None:
        """Test: rule nie match gdy inhibitory AU jest za wysokie."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20},
            inhibitory_aus={"EAD103": 1.10},  # EAD103 musi być < 1.10
        )

        # EAD103 za wysokie (1.20 > 1.10)
        delta_aus = {
            "AU12": DeltaActionUnit("AU12", 1.30, 0.30, True, 0.9),
            "EAD103": DeltaActionUnit("EAD103", 1.20, 0.20, True, 0.85),
        }

        matches, confidence = rule.matches(delta_aus)

        assert matches is False

    def test_matches_missing_required_au(self) -> None:
        """Test: rule nie match gdy brakuje required AU."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20, "EAD102": 1.10},
        )

        # Brak EAD102
        delta_aus = {
            "AU12": DeltaActionUnit("AU12", 1.30, 0.30, True, 0.9),
        }

        matches, confidence = rule.matches(delta_aus)

        assert matches is False

    def test_optional_aus_increase_confidence(self) -> None:
        """Test: optional AUs zwiększają confidence."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20},
            optional_aus={"AU101": 1.10, "EAD102": 1.10},
        )

        # Bez optional AUs
        delta_aus_without = {
            "AU12": DeltaActionUnit("AU12", 1.30, 0.30, True, 0.8),
            "AU101": DeltaActionUnit("AU101", 1.00, 0.00, False, 0.8),
            "EAD102": DeltaActionUnit("EAD102", 1.00, 0.00, False, 0.8),
        }

        # Z optional AUs
        delta_aus_with = {
            "AU12": DeltaActionUnit("AU12", 1.30, 0.30, True, 0.8),
            "AU101": DeltaActionUnit("AU101", 1.15, 0.15, True, 0.8),
            "EAD102": DeltaActionUnit("EAD102", 1.15, 0.15, True, 0.8),
        }

        matches_without, conf_without = rule.matches(delta_aus_without)
        matches_with, conf_with = rule.matches(delta_aus_with)

        assert matches_without is True
        assert matches_with is True
        assert conf_with > conf_without  # Confidence z optional AUs powinna być wyższa


class TestDogFACSRuleEngine:
    """Testy dla klasy DogFACSRuleEngine."""

    @pytest.fixture
    def engine(self) -> DogFACSRuleEngine:
        """Fixture dla rule engine z domyślnymi regułami."""
        return DogFACSRuleEngine()

    def test_engine_initialization(self, engine: DogFACSRuleEngine) -> None:
        """Test inicjalizacji engine."""
        assert len(engine.rules) == 6  # 6 emotions
        # Rules powinny być posortowane po priorytecie (malejąco)
        priorities = [rule.priority for rule in engine.rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_classify_happy(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja happy emotion."""
        # Delta AUs charakterystyczne dla happy
        delta_aus = {
            "AU101": DeltaActionUnit("AU101", 1.05, 0.05, False, 0.9),
            "AU102": DeltaActionUnit("AU102", 1.03, 0.03, False, 0.9),
            "AU12": DeltaActionUnit("AU12", 1.25, 0.25, True, 0.9),  # Smile
            "AU115": DeltaActionUnit("AU115", 1.00, 0.00, False, 0.9),
            "AU116": DeltaActionUnit("AU116", 1.00, 0.00, False, 0.9),
            "AU117": DeltaActionUnit("AU117", 1.00, 0.00, False, 0.9),
            "AU121": DeltaActionUnit("AU121", 1.00, 0.00, False, 0.9),
            "EAD102": DeltaActionUnit("EAD102", 1.15, 0.15, True, 0.9),  # Ears forward
            "EAD103": DeltaActionUnit("EAD103", 1.00, 0.00, False, 0.9),
            "AD19": DeltaActionUnit("AD19", 1.00, 0.00, False, 0.9),
            "AD37": DeltaActionUnit("AD37", 1.00, 0.00, False, 0.9),
            "AU26": DeltaActionUnit("AU26", 1.05, 0.05, False, 0.9),
        }

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "happy"
        assert prediction.confidence > 0.7

    def test_classify_angry(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja angry emotion."""
        # Delta AUs charakterystyczne dla angry
        delta_aus = {
            "AU101": DeltaActionUnit("AU101", 1.10, 0.10, True, 0.9),
            "AU102": DeltaActionUnit("AU102", 1.00, 0.00, False, 0.9),
            "AU12": DeltaActionUnit("AU12", 1.20, 0.20, True, 0.9),  # Snarl
            "AU115": DeltaActionUnit("AU115", 1.00, 0.00, False, 0.9),
            "AU116": DeltaActionUnit("AU116", 1.00, 0.00, False, 0.9),
            "AU117": DeltaActionUnit("AU117", 1.00, 0.00, False, 0.9),
            "AU121": DeltaActionUnit("AU121", 1.00, 0.00, False, 0.9),
            "EAD102": DeltaActionUnit("EAD102", 1.00, 0.00, False, 0.9),
            "EAD103": DeltaActionUnit("EAD103", 1.15, 0.15, True, 0.9),  # Ears back
            "AD19": DeltaActionUnit("AD19", 1.00, 0.00, False, 0.9),
            "AD37": DeltaActionUnit("AD37", 1.00, 0.00, False, 0.9),
            "AU26": DeltaActionUnit("AU26", 1.30, 0.30, True, 0.9),  # Jaw drop
        }

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "angry"

    def test_classify_fearful(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja fearful emotion."""
        # Delta AUs charakterystyczne dla fearful
        delta_aus = {
            "AU101": DeltaActionUnit("AU101", 1.12, 0.12, True, 0.9),  # Brows raised
            "AU102": DeltaActionUnit("AU102", 1.00, 0.00, False, 0.9),
            "AU12": DeltaActionUnit("AU12", 1.00, 0.00, False, 0.9),
            "AU115": DeltaActionUnit("AU115", 1.00, 0.00, False, 0.9),
            "AU116": DeltaActionUnit("AU116", 1.00, 0.00, False, 0.9),
            "AU117": DeltaActionUnit("AU117", 1.15, 0.15, True, 0.9),  # Eye closure
            "AU121": DeltaActionUnit("AU121", 1.00, 0.00, False, 0.9),
            "EAD102": DeltaActionUnit("EAD102", 1.00, 0.00, False, 0.9),
            "EAD103": DeltaActionUnit("EAD103", 1.20, 0.20, True, 0.9),  # Ears flattened
            "AD19": DeltaActionUnit("AD19", 1.00, 0.00, False, 0.9),
            "AD37": DeltaActionUnit("AD37", 1.12, 0.12, True, 0.9),  # Nose lick (stress)
            "AU26": DeltaActionUnit("AU26", 1.05, 0.05, False, 0.9),
        }

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "fearful"

    def test_classify_sad(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja sad emotion."""
        # Delta AUs charakterystyczne dla sad
        delta_aus = {
            "AU101": DeltaActionUnit("AU101", 1.00, 0.00, False, 0.9),
            "AU102": DeltaActionUnit("AU102", 1.00, 0.00, False, 0.9),
            "AU12": DeltaActionUnit("AU12", 1.00, 0.00, False, 0.9),
            "AU115": DeltaActionUnit("AU115", 1.00, 0.00, False, 0.9),
            "AU116": DeltaActionUnit("AU116", 1.00, 0.00, False, 0.9),
            "AU117": DeltaActionUnit("AU117", 1.00, 0.00, False, 0.9),
            "AU121": DeltaActionUnit("AU121", 1.00, 0.00, False, 0.9),
            "EAD102": DeltaActionUnit("EAD102", 1.00, 0.00, False, 0.9),
            "EAD103": DeltaActionUnit("EAD103", 1.12, 0.12, True, 0.9),  # Ears back
            "AD19": DeltaActionUnit("AD19", 1.00, 0.00, False, 0.9),
            "AD37": DeltaActionUnit("AD37", 1.00, 0.00, False, 0.9),
            "AU26": DeltaActionUnit("AU26", 1.00, 0.00, False, 0.9),
        }

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "sad"

    def test_classify_relaxed(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja relaxed emotion."""
        # Delta AUs - minimalna aktywacja
        delta_aus = {
            "AU101": DeltaActionUnit("AU101", 1.00, 0.00, False, 0.9),
            "AU102": DeltaActionUnit("AU102", 1.00, 0.00, False, 0.9),
            "AU12": DeltaActionUnit("AU12", 1.00, 0.00, False, 0.9),
            "AU115": DeltaActionUnit("AU115", 1.00, 0.00, False, 0.9),
            "AU116": DeltaActionUnit("AU116", 1.00, 0.00, False, 0.9),
            "AU117": DeltaActionUnit("AU117", 1.00, 0.00, False, 0.9),
            "AU121": DeltaActionUnit("AU121", 1.00, 0.00, False, 0.9),
            "EAD102": DeltaActionUnit("EAD102", 1.00, 0.00, False, 0.9),
            "EAD103": DeltaActionUnit("EAD103", 1.00, 0.00, False, 0.9),
            "AD19": DeltaActionUnit("AD19", 1.00, 0.00, False, 0.9),
            "AD37": DeltaActionUnit("AD37", 1.00, 0.00, False, 0.9),
            "AU26": DeltaActionUnit("AU26", 1.00, 0.00, False, 0.9),
        }

        prediction = engine.classify(delta_aus)

        # Może być relaxed lub neutral (oba mają minimalna aktywację)
        assert prediction.emotion in ["relaxed", "neutral"]

    def test_classify_neutral_fallback(self, engine: DogFACSRuleEngine) -> None:
        """Test: neutral jako fallback gdy nic nie pasuje."""
        # Słabe AUs, które nie pasują do żadnej reguły
        delta_aus = {
            "AU101": DeltaActionUnit("AU101", 1.05, 0.05, False, 0.3),  # Niska conf
            "AU102": DeltaActionUnit("AU102", 1.05, 0.05, False, 0.3),
            "AU12": DeltaActionUnit("AU12", 1.05, 0.05, False, 0.3),
            "AU115": DeltaActionUnit("AU115", 1.00, 0.00, False, 0.3),
            "AU116": DeltaActionUnit("AU116", 1.00, 0.00, False, 0.3),
            "AU117": DeltaActionUnit("AU117", 1.00, 0.00, False, 0.3),
            "AU121": DeltaActionUnit("AU121", 1.00, 0.00, False, 0.3),
            "EAD102": DeltaActionUnit("EAD102", 1.00, 0.00, False, 0.3),
            "EAD103": DeltaActionUnit("EAD103", 1.00, 0.00, False, 0.3),
            "AD19": DeltaActionUnit("AD19", 1.00, 0.00, False, 0.3),
            "AD37": DeltaActionUnit("AD37", 1.00, 0.00, False, 0.3),
            "AU26": DeltaActionUnit("AU26", 1.00, 0.00, False, 0.3),
        }

        prediction = engine.classify(delta_aus)

        # Powinno pasować do neutral lub relaxed
        assert prediction.emotion in ["neutral", "relaxed"]

    def test_priority_order_matters(self) -> None:
        """Test: priorytet reguł ma znaczenie."""
        # Twórz custom rules z różnymi priorytetami
        rule_high = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.10},
        )
        rule_low = EmotionRule(
            emotion="relaxed",
            priority=50,
            required_aus={"AU12": 1.10},
        )

        engine = DogFACSRuleEngine(rules=[rule_high, rule_low])

        # Delta AUs pasujące do obu reguł
        delta_aus = {
            "AU12": DeltaActionUnit("AU12", 1.20, 0.20, True, 0.9),
        }

        prediction = engine.classify(delta_aus)

        # Powinna wybrać rule z wyższym priorytetem
        assert prediction.emotion == "happy"

    def test_prediction_includes_rule_applied(self, engine: DogFACSRuleEngine) -> None:
        """Test: prediction zawiera nazwę zastosowanej reguły."""
        delta_aus = {
            "AU101": DeltaActionUnit("AU101", 1.00, 0.00, False, 0.9),
            "AU102": DeltaActionUnit("AU102", 1.00, 0.00, False, 0.9),
            "AU12": DeltaActionUnit("AU12", 1.25, 0.25, True, 0.9),
            "AU115": DeltaActionUnit("AU115", 1.00, 0.00, False, 0.9),
            "AU116": DeltaActionUnit("AU116", 1.00, 0.00, False, 0.9),
            "AU117": DeltaActionUnit("AU117", 1.00, 0.00, False, 0.9),
            "AU121": DeltaActionUnit("AU121", 1.00, 0.00, False, 0.9),
            "EAD102": DeltaActionUnit("EAD102", 1.15, 0.15, True, 0.9),
            "EAD103": DeltaActionUnit("EAD103", 1.00, 0.00, False, 0.9),
            "AD19": DeltaActionUnit("AD19", 1.00, 0.00, False, 0.9),
            "AD37": DeltaActionUnit("AD37", 1.00, 0.00, False, 0.9),
            "AU26": DeltaActionUnit("AU26", 1.00, 0.00, False, 0.9),
        }

        prediction = engine.classify(delta_aus)

        assert prediction.rule_applied is not None
        assert "happy" in prediction.rule_applied
        assert "priority" in prediction.rule_applied


class TestConvenienceFunctions:
    """Testy dla funkcji pomocniczych."""

    def test_classify_emotion_from_delta_aus(self) -> None:
        """Test funkcji classify_emotion_from_delta_aus."""
        delta_aus = {
            "AU101": DeltaActionUnit("AU101", 1.00, 0.00, False, 0.9),
            "AU102": DeltaActionUnit("AU102", 1.00, 0.00, False, 0.9),
            "AU12": DeltaActionUnit("AU12", 1.25, 0.25, True, 0.9),
            "AU115": DeltaActionUnit("AU115", 1.00, 0.00, False, 0.9),
            "AU116": DeltaActionUnit("AU116", 1.00, 0.00, False, 0.9),
            "AU117": DeltaActionUnit("AU117", 1.00, 0.00, False, 0.9),
            "AU121": DeltaActionUnit("AU121", 1.00, 0.00, False, 0.9),
            "EAD102": DeltaActionUnit("EAD102", 1.15, 0.15, True, 0.9),
            "EAD103": DeltaActionUnit("EAD103", 1.00, 0.00, False, 0.9),
            "AD19": DeltaActionUnit("AD19", 1.00, 0.00, False, 0.9),
            "AD37": DeltaActionUnit("AD37", 1.00, 0.00, False, 0.9),
            "AU26": DeltaActionUnit("AU26", 1.00, 0.00, False, 0.9),
        }

        prediction = classify_emotion_from_delta_aus(delta_aus)

        assert isinstance(prediction, EmotionPrediction)
        assert prediction.emotion in EMOTION_CLASSES


class TestEmotionRulesDatabase:
    """Testy dla bazy reguł EMOTION_RULES."""

    def test_emotion_rules_count(self) -> None:
        """Test: powinno być 6 reguł (dla 6 emocji)."""
        assert len(EMOTION_RULES) == 6

    def test_emotion_rules_unique_emotions(self) -> None:
        """Test: każda reguła ma unikalną emocję."""
        emotions = [rule.emotion for rule in EMOTION_RULES]
        assert len(emotions) == len(set(emotions))

    def test_emotion_rules_cover_all_classes(self) -> None:
        """Test: reguły pokrywają wszystkie EMOTION_CLASSES."""
        rule_emotions = {rule.emotion for rule in EMOTION_RULES}
        expected_emotions = set(EMOTION_CLASSES)

        assert rule_emotions == expected_emotions

    def test_emotion_rules_priorities_unique(self) -> None:
        """Test: każda reguła ma unikalny priorytet."""
        priorities = [rule.priority for rule in EMOTION_RULES]
        assert len(priorities) == len(set(priorities))

    def test_neutral_rule_lowest_priority(self) -> None:
        """Test: neutral powinien mieć najniższy priorytet (fallback)."""
        neutral_rule = next(r for r in EMOTION_RULES if r.emotion == "neutral")
        other_priorities = [r.priority for r in EMOTION_RULES if r.emotion != "neutral"]

        assert neutral_rule.priority < min(other_priorities)

    def test_happy_rule_highest_priority(self) -> None:
        """Test: happy powinien mieć najwyższy priorytet."""
        happy_rule = next(r for r in EMOTION_RULES if r.emotion == "happy")
        other_priorities = [r.priority for r in EMOTION_RULES if r.emotion != "happy"]

        assert happy_rule.priority > max(other_priorities)
