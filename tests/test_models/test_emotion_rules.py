"""
Testy dla klasyfikatora emocji opartego na regułach DogFACS.

Uruchomienie:
    pytest tests/test_models/test_emotion_rules.py -v
"""

import pytest

from packages.models.delta_action_units import DeltaActionUnit
from packages.models.emotion import (
    EMOTION_CLASSES,
    EMOTION_RULES,
    NUM_EMOTIONS,
    DogFACSRuleEngine,
    EmotionPrediction,
    EmotionRule,
    classify_emotion_from_delta_aus,
)

# =============================================================================
# Helperу: tworzenie zestawów AU
# =============================================================================

def make_neutral_aus() -> dict[str, DeltaActionUnit]:
    """Tworzy zestaw neutralnych AU (ratio=1.0, brak aktywacji)."""
    all_aus = [
        "AU101", "AU143", "AU145",
        "AU109", "AU110", "AU12", "AU116", "AU118", "AU25", "AU26", "AU27",
        "AD19", "AD33", "AD35", "AD37", "AD137",
        "EAD101", "EAD102", "EAD103", "EAD104", "EAD105",
    ]
    return {
        name: DeltaActionUnit(name, 1.0, 0.0, False, 0.9)
        for name in all_aus
    }


def make_aus_with(**overrides: float) -> dict[str, DeltaActionUnit]:
    """
    Tworzy zestaw AU z nadpisaniem wybranych wartości ratio.

    Przykład:
        make_aus_with(AU12=1.30, EAD101=1.20)
    """
    aus = make_neutral_aus()
    for au_name, ratio in overrides.items():
        delta = ratio - 1.0
        aus[au_name] = DeltaActionUnit(au_name, ratio, delta, ratio > 1.15, 0.9)
    return aus


# =============================================================================
# Testy klasy EmotionRule
# =============================================================================

class TestEmotionRule:
    """Testy dla klasy EmotionRule."""

    def test_creation(self) -> None:
        """Test tworzenia EmotionRule z poprawnymi polami."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20, "EAD101": 1.10},
            inhibitory_aus={"EAD103": 1.10},
            optional_aus={"AU101": 1.10},
            min_confidence=0.7,
        )

        assert rule.emotion == "happy"
        assert rule.priority == 100
        assert len(rule.required_aus) == 2
        assert len(rule.inhibitory_aus) == 1
        assert len(rule.optional_aus) == 1

    def test_matches_when_required_aus_met(self) -> None:
        """Test: reguła pasuje gdy wszystkie wymagane AU są spełnione."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20, "EAD101": 1.10},
            inhibitory_aus={"EAD103": 1.10},
        )

        delta_aus = make_aus_with(AU12=1.30, EAD101=1.15)
        matches, confidence = rule.matches(delta_aus)

        assert matches is True
        assert confidence >= 0.7

    def test_no_match_when_required_au_too_low(self) -> None:
        """Test: reguła nie pasuje gdy required AU ma za niskie ratio."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20, "EAD101": 1.10},
        )

        # AU12 za niskie (1.10 < wymagane 1.20)
        delta_aus = make_aus_with(AU12=1.10, EAD101=1.15)
        matches, _ = rule.matches(delta_aus)

        assert matches is False

    def test_no_match_when_inhibitory_au_too_high(self) -> None:
        """Test: reguła nie pasuje gdy inhibitory AU jest za wysokie."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20},
            inhibitory_aus={"EAD103": 1.10},
        )

        # EAD103 za wysokie (1.25 > max 1.10)
        delta_aus = make_aus_with(AU12=1.30, EAD103=1.25)
        matches, _ = rule.matches(delta_aus)

        assert matches is False

    def test_no_match_when_required_au_missing(self) -> None:
        """Test: reguła nie pasuje gdy brakuje required AU w słowniku."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20, "EAD101": 1.10},
        )

        # Brak EAD101 w słowniku
        delta_aus = {"AU12": DeltaActionUnit("AU12", 1.30, 0.30, True, 0.9)}
        matches, _ = rule.matches(delta_aus)

        assert matches is False

    def test_optional_aus_increase_confidence(self) -> None:
        """Test: opcjonalne AU zwiększają pewność dopasowania."""
        rule = EmotionRule(
            emotion="happy",
            priority=100,
            required_aus={"AU12": 1.20},
            optional_aus={"AU101": 1.10, "EAD101": 1.10},
        )

        aus_without_optional = make_aus_with(AU12=1.30)
        aus_with_optional = make_aus_with(AU12=1.30, AU101=1.15, EAD101=1.20)

        _, conf_without = rule.matches(aus_without_optional)
        _, conf_with = rule.matches(aus_with_optional)

        assert conf_with > conf_without


# =============================================================================
# Testy klasy DogFACSRuleEngine
# =============================================================================

class TestDogFACSRuleEngine:
    """Testy dla klasy DogFACSRuleEngine."""

    @pytest.fixture
    def engine(self) -> DogFACSRuleEngine:
        """Fixture: silnik reguł z domyślnymi regułami."""
        return DogFACSRuleEngine()

    def test_engine_has_9_rules(self, engine: DogFACSRuleEngine) -> None:
        """Test: silnik powinien mieć 9 reguł (dla 9 emocji)."""
        assert len(engine.rules) == 9

    def test_rules_sorted_by_priority_descending(
        self, engine: DogFACSRuleEngine
    ) -> None:
        """Test: reguły posortowane malejąco według priorytetu."""
        priorities = [rule.priority for rule in engine.rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_classify_happy(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja emocji happy."""
        # Uśmiech (AU12) + uszy do przodu (EAD101)
        delta_aus = make_aus_with(AU12=1.25, EAD101=1.15)

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "happy"
        assert prediction.confidence > 0.7

    def test_classify_angry(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja emocji angry."""
        # Opad żuchwy (AU26) + zmarszczony nos (AU109)
        delta_aus = make_aus_with(AU26=1.30, AU109=1.20)

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "angry"

    def test_classify_fearful(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja emocji fearful."""
        # Uszy płaskie (EAD103) + napięte brwi (AU101)
        delta_aus = make_aus_with(EAD103=1.20, AU101=1.15)

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "fearful"

    def test_classify_sad(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja emocji sad."""
        # Uszy lekko do tyłu (EAD103 słabo), zamknięty pysk
        delta_aus = make_aus_with(EAD103=1.12)

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "sad"

    def test_classify_surprise(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja emocji surprise."""
        # Wargi rozchylone (AU25) + uszy do przodu (EAD101) + brwi (AU101)
        delta_aus = make_aus_with(AU25=1.25, EAD101=1.20, AU101=1.15)

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "surprise"

    def test_classify_pain(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja emocji pain."""
        # Zmrużone oczy (AU143 niska wartość) + napięte brwi + opuszczona warga
        aus = make_neutral_aus()
        aus["AU143"] = DeltaActionUnit("AU143", 0.65, -0.35, True, 0.9)
        aus["AU101"] = DeltaActionUnit("AU101", 1.20, 0.20, True, 0.9)
        aus["AU116"] = DeltaActionUnit("AU116", 1.20, 0.20, True, 0.9)

        prediction = engine.classify(aus)

        assert prediction.emotion == "pain"

    def test_classify_submission(self, engine: DogFACSRuleEngine) -> None:
        """Test: klasyfikacja emocji submission."""
        # Uszy bardzo płaskie (EAD103 mocno) + addukcja uszu (EAD102)
        delta_aus = make_aus_with(EAD103=1.25, EAD102=1.15)

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "submission"

    def test_classify_relaxed_or_neutral_when_no_activation(
        self, engine: DogFACSRuleEngine
    ) -> None:
        """Test: brak aktywacji → relaxed lub neutral."""
        delta_aus = make_neutral_aus()

        prediction = engine.classify(delta_aus)

        assert prediction.emotion in ("relaxed", "neutral")

    def test_priority_determines_winner_when_both_rules_match(self) -> None:
        """Test: wyższy priorytet wygrywa gdy obie reguły pasują."""
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
        delta_aus = {"AU12": DeltaActionUnit("AU12", 1.20, 0.20, True, 0.9)}

        prediction = engine.classify(delta_aus)

        assert prediction.emotion == "happy"

    def test_prediction_includes_rule_applied(
        self, engine: DogFACSRuleEngine
    ) -> None:
        """Test: predykcja zawiera informację o zastosowanej regule."""
        delta_aus = make_aus_with(AU12=1.25, EAD101=1.15)

        prediction = engine.classify(delta_aus)

        assert prediction.rule_applied is not None
        assert "priority" in prediction.rule_applied

    def test_prediction_probabilities_sum_to_one(
        self, engine: DogFACSRuleEngine
    ) -> None:
        """Test: suma prawdopodobieństw = 1.0."""
        delta_aus = make_aus_with(AU12=1.25, EAD101=1.15)

        prediction = engine.classify(delta_aus)

        assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_prediction_emotion_id_matches_emotion_class(
        self, engine: DogFACSRuleEngine
    ) -> None:
        """Test: emotion_id odpowiada indeksowi w EMOTION_CLASSES."""
        delta_aus = make_neutral_aus()

        prediction = engine.classify(delta_aus)

        assert EMOTION_CLASSES[prediction.emotion_id] == prediction.emotion


# =============================================================================
# Testy funkcji pomocniczej classify_emotion_from_delta_aus
# =============================================================================

class TestClassifyEmotionFromDeltaAus:
    """Testy dla funkcji classify_emotion_from_delta_aus."""

    def test_returns_emotion_prediction(self) -> None:
        """Test: funkcja zwraca obiekt EmotionPrediction."""
        delta_aus = make_neutral_aus()

        result = classify_emotion_from_delta_aus(delta_aus)

        assert isinstance(result, EmotionPrediction)

    def test_emotion_in_valid_classes(self) -> None:
        """Test: wynikowa emocja należy do EMOTION_CLASSES."""
        delta_aus = make_neutral_aus()

        result = classify_emotion_from_delta_aus(delta_aus)

        assert result.emotion in EMOTION_CLASSES

    def test_accepts_custom_rule_engine(self) -> None:
        """Test: funkcja akceptuje niestandardowy silnik reguł."""
        custom_engine = DogFACSRuleEngine(rules=[
            EmotionRule(emotion="neutral", priority=50, min_confidence=0.0)
        ])

        delta_aus = make_neutral_aus()
        result = classify_emotion_from_delta_aus(delta_aus, rule_engine=custom_engine)

        assert result.emotion == "neutral"


# =============================================================================
# Testy bazy reguł EMOTION_RULES i stałych
# =============================================================================

class TestEmotionRulesDatabase:
    """Testy dla bazy reguł EMOTION_RULES i stałych modułu."""

    def test_emotion_rules_count_is_9(self) -> None:
        """Test: baza powinna mieć 9 reguł (dla 9 emocji)."""
        assert len(EMOTION_RULES) == 9

    def test_emotion_classes_count_is_9(self) -> None:
        """Test: EMOTION_CLASSES powinno mieć 9 elementów."""
        assert len(EMOTION_CLASSES) == 9
        assert NUM_EMOTIONS == 9

    def test_all_9_emotions_present(self) -> None:
        """Test: wszystkie 9 emocji z Mota-Rojas et al. 2021 są w EMOTION_CLASSES."""
        expected = {
            "happy", "sad", "angry", "fearful", "relaxed",
            "neutral", "surprise", "pain", "submission",
        }
        assert set(EMOTION_CLASSES) == expected

    def test_emotion_rules_unique_emotions(self) -> None:
        """Test: każda reguła ma unikalną emocję."""
        emotions = [rule.emotion for rule in EMOTION_RULES]
        assert len(emotions) == len(set(emotions))

    def test_emotion_rules_cover_all_classes(self) -> None:
        """Test: reguły pokrywają wszystkie EMOTION_CLASSES."""
        rule_emotions = {rule.emotion for rule in EMOTION_RULES}
        assert rule_emotions == set(EMOTION_CLASSES)

    def test_emotion_rules_priorities_unique(self) -> None:
        """Test: każda reguła ma unikalny priorytet."""
        priorities = [rule.priority for rule in EMOTION_RULES]
        assert len(priorities) == len(set(priorities))

    def test_neutral_rule_has_lowest_priority(self) -> None:
        """Test: neutral powinien mieć najniższy priorytet (fallback)."""
        neutral_rule = next(r for r in EMOTION_RULES if r.emotion == "neutral")
        other_priorities = [r.priority for r in EMOTION_RULES if r.emotion != "neutral"]
        assert neutral_rule.priority < min(other_priorities)

    def test_neutral_rule_always_matches(self) -> None:
        """Test: reguła neutral dopasowuje przy dowolnych AU (fallback)."""
        neutral_rule = next(r for r in EMOTION_RULES if r.emotion == "neutral")
        assert neutral_rule.min_confidence == 0.0
        assert not neutral_rule.required_aus

    def test_happy_rule_has_highest_priority(self) -> None:
        """Test: happy powinien mieć najwyższy priorytet."""
        happy_rule = next(r for r in EMOTION_RULES if r.emotion == "happy")
        other_priorities = [r.priority for r in EMOTION_RULES if r.emotion != "happy"]
        assert happy_rule.priority > max(other_priorities)

    def test_no_old_invalid_au_codes_in_rules(self) -> None:
        """Test: stare, nieistniejące AU nie pojawiają się w żadnej regule."""
        invalid_codes = {"AU102", "AU115", "AU117", "AU121", "EAD102_old"}
        for rule in EMOTION_RULES:
            all_rule_aus = (
                set(rule.required_aus)
                | set(rule.inhibitory_aus)
                | set(rule.optional_aus)
            )
            for invalid in invalid_codes:
                assert invalid not in all_rule_aus, (
                    f"Nieprawidłowy AU {invalid} znaleziony w regule {rule.emotion}"
                )
