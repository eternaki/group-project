"""
Klasyfikator emocji psów oparty wyłącznie na regułach DogFACS.

BRAK UCZENIA MASZYNOWEGO — używa wyłącznie reguł naukowych z badań DogFACS.

Architektura:
    Keypoints (46×3=138) → Delta Action Units (21) → Dopasowanie reguł → Emocja (9 klas)

Action Units (AU) to obiektywne pomiary ruchów mięśni twarzy według
Dog Facial Action Coding System (DogFACS). Emocje są interpretacją
kombinacji AU (tzw. poselets).

9 klas emocji według Mota-Rojas et al. 2021:
    happy, sad, angry, fearful, relaxed, neutral, surprise, pain, submission

Źródło naukowe: Mota-Rojas et al. 2021, https://www.animalfacs.com/dogfacs
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from packages.data.schemas import EMOTION_CLASSES
from packages.models.delta_action_units import MIN_AU_CONFIDENCE, DeltaActionUnit

NUM_EMOTIONS: int = len(EMOTION_CLASSES)


@dataclass
class EmotionPrediction:
    """
    Wynik klasyfikacji emocji.

    Attributes:
        emotion_id: ID klasy emocji (0-8)
        emotion: Nazwa emocji (happy, sad, angry, fearful, relaxed,
                 neutral, surprise, pain, submission)
        confidence: Pewność predykcji (0.0-1.0)
        probabilities: Prawdopodobieństwa dla wszystkich 9 klas
        action_units: Wartości Action Units (ratio)
        rule_applied: Nazwa reguły która dopasowała (dla przejrzystości)
    """

    emotion_id: int
    emotion: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    action_units: dict[str, float] = field(default_factory=dict)
    rule_applied: Optional[str] = None

    def to_dict(self) -> dict:
        """Konwertuje predykcję do słownika."""
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
        Zwraca dane w formacie kompatybilnym z COCO.

        Returns:
            Słownik z emotion i emotion_confidence
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
    Pojedyncza reguła klasyfikacji emocji (poselet matcher).

    Oparta na badaniach naukowych DogFACS — dopasowuje konkretne kombinacje AU.

    Attributes:
        emotion: Etykieta emocji (np. "happy", "angry")
        priority: Priorytet reguły (wyższy = sprawdzany wcześniej)
        required_aus: AU_name → min_ratio (musi być przekroczony)
        inhibitory_aus: AU_name → max_ratio (musi pozostać poniżej)
        optional_aus: AU_name → min_ratio (bonus jeśli obecny)
        min_confidence: Minimalny próg pewności dla aktywacji reguły
    """

    emotion: str
    priority: int
    required_aus: dict[str, float] = field(default_factory=dict)
    inhibitory_aus: dict[str, float] = field(default_factory=dict)
    optional_aus: dict[str, float] = field(default_factory=dict)
    min_confidence: float = 0.7

    def matches(
        self,
        delta_aus: dict[str, DeltaActionUnit],
    ) -> tuple[bool, float]:
        """
        Sprawdza czy reguła pasuje do aktualnego stanu AU.

        Args:
            delta_aus: Słownik AU_name → DeltaActionUnit

        Returns:
            (dopasowanie, wynik_pewności)
        """
        if not self._required_aus_met(delta_aus):
            return False, 0.0
        if not self._inhibitory_aus_clear(delta_aus):
            return False, 0.0

        confidence = self._compute_confidence(delta_aus)
        return confidence >= self.min_confidence, confidence

    def _required_aus_met(self, delta_aus: dict[str, DeltaActionUnit]) -> bool:
        """
        Sprawdza czy wszystkie wymagane AU są aktywowane.

        Konwencja progu:
        - threshold >= 1.0: AU aktywowany przez wzrost (ratio >= threshold)
        - threshold < 1.0: AU aktywowany przez spadek (ratio <= threshold),
          np. AU143 (zmrużenie) wymaga ratio <= 0.75
        """
        for au_name, threshold in self.required_aus.items():
            if au_name not in delta_aus:
                return False
            au = delta_aus[au_name]
            # Niewiarygodny pomiar (np. klamrowane ucho, confidence 0) nie może
            # potwierdzić wymaganego AU — nie budujemy na nim emocji.
            if au.confidence < MIN_AU_CONFIDENCE:
                return False
            ratio = au.ratio
            if threshold < 1.0:
                if ratio > threshold:
                    return False
            else:
                if ratio < threshold:
                    return False
        return True

    def _inhibitory_aus_clear(self, delta_aus: dict[str, DeltaActionUnit]) -> bool:
        """Sprawdza czy żaden hamujący AU nie jest aktywowany.

        Niewiarygodny pomiar (confidence < próg, np. klamrowane ucho) nie blokuje
        reguły — nie można na jego podstawie twierdzić, że AU jest aktywny.
        """
        return all(
            au_name not in delta_aus
            or delta_aus[au_name].confidence < MIN_AU_CONFIDENCE
            or delta_aus[au_name].ratio <= max_ratio
            for au_name, max_ratio in self.inhibitory_aus.items()
        )

    def _compute_confidence(
        self,
        delta_aus: dict[str, DeltaActionUnit],
    ) -> float:
        """Oblicza pewność dopasowania na podstawie widoczności AU."""
        if self.required_aus:
            confidences = [
                delta_aus[au_name].confidence
                for au_name in self.required_aus
                if au_name in delta_aus
            ]
            base = float(np.mean(confidences)) if confidences else 0.5
        else:
            all_conf = [au.confidence for au in delta_aus.values()]
            base = float(np.mean(all_conf)) if all_conf else 0.5

        bonus = sum(
            0.05
            for au_name, threshold in self.optional_aus.items()
            if au_name in delta_aus and (
                (threshold < 1.0 and delta_aus[au_name].ratio <= threshold)
                or (threshold >= 1.0 and delta_aus[au_name].ratio >= threshold)
            )
        )
        return min(1.0, base + bonus)


# =============================================================================
# BAZA REGUŁ EMOCJI (na podstawie oficjalnych kodów DogFACS)
# Źródło: Mota-Rojas et al. 2021, Waller et al. 2013
# =============================================================================

EMOTION_RULES: list[EmotionRule] = [

    # HAPPY — priorytet 100
    # Kąciki ust uniesione (AU12), uszy do przodu (EAD101), pysk otwarty (AU25)
    EmotionRule(
        emotion="happy",
        priority=100,
        required_aus={
            "AU12":   1.20,  # Lip Corner Puller: uśmiech 20%+
            "EAD101": 1.10,  # Ears Forward: uszy do przodu 10%+
        },
        inhibitory_aus={
            "EAD103": 1.10,  # NIE: uszy przyciśnięte
            "AU26":   1.30,  # NIE: ekstremalny opad żuchwy
        },
        optional_aus={
            "AU25":  1.10,  # Bonus: wargi rozchylone
            "AU101": 1.10,  # Bonus: uniesiona brew (zabawa)
            "AD19":  1.15,  # Bonus: widoczny język
        },
    ),

    # ANGRY — priorytet 95
    # Otwarty pysk (AU26/AU27), zmarszczony nos (AU109/AU110), uszy do tyłu
    EmotionRule(
        emotion="angry",
        priority=95,
        required_aus={
            "AU26":  1.25,  # Jaw Drop: opad żuchwy 25%+
            "AU109": 1.15,  # Nose Wrinkler Left: zmarszczenie nosa
        },
        inhibitory_aus={
            "EAD101": 1.10,  # NIE: uszy do przodu (to byłoby happy)
        },
        optional_aus={
            "AU110":  1.15,  # Bonus: zmarszczenie nosa prawa
            "AU101":  1.15,  # Bonus: napięcie brwi
            "EAD103": 1.10,  # Bonus: uszy przyciśnięte
            "AU27":   1.20,  # Bonus: szerokie otwarcie pyska
        },
    ),

    # SURPRISE — priorytet 92
    # Szeroko otwarte oczy (AU143 mała wartość = NIE zmrużone), usta otwarte (AU25), uszy do przodu
    EmotionRule(
        emotion="surprise",
        priority=92,
        required_aus={
            "AU25":   1.20,  # Lips Part: rozchylone wargi 20%+
            "EAD101": 1.15,  # Ears Forward: uszy do przodu 15%+
            "AU101":  1.10,  # Inner Brow Raiser: uniesione brwi
        },
        inhibitory_aus={
            "EAD103": 1.10,  # NIE: uszy płaskie
            "AU109":  1.15,  # NIE: zmarszczony nos (byłoby angry)
        },
        optional_aus={
            "AU26":  1.15,  # Bonus: opad żuchwy
        },
    ),

    # FEARFUL — priorytet 90
    # Uszy przyciśnięte (EAD103), napięte brwi (AU101), pysk zamknięty
    EmotionRule(
        emotion="fearful",
        priority=90,
        required_aus={
            "EAD103": 1.15,  # Ears Flattener: uszy płaskie 15%+
            "AU101":  1.10,  # Inner Brow Raiser: napięcie brwi
        },
        inhibitory_aus={
            "AU26":   1.20,  # NIE: szeroko otwarty pysk
            "EAD101": 1.10,  # NIE: uszy do przodu
        },
        optional_aus={
            "AD137":  1.10,  # Bonus: lizanie nosa (stres)
            "AU145":  0.85,  # Bonus: mrugnięcie (obniżona widoczność)
            "AU118":  1.10,  # Bonus: rozciągnięte wargi
        },
    ),

    # PAIN — priorytet 88
    # Zmrużone oczy (AU143), napięte brwi (AU101), opuszczona dolna warga (AU116)
    EmotionRule(
        emotion="pain",
        priority=88,
        required_aus={
            "AU143": 0.75,  # Lid Tightener: zmrużenie (ratio < 1 = mniejsze otwarcie)
            "AU101": 1.15,  # Inner Brow Raiser: napięcie/uniesienie brwi
            "AU116": 1.15,  # Lower Lip Depressor: opuszczona dolna warga
        },
        inhibitory_aus={
            "EAD101": 1.10,  # NIE: uszy do przodu
            "AU12":   1.15,  # NIE: uśmiech
        },
        optional_aus={
            "EAD103": 1.10,  # Bonus: uszy płaskie
            "AD137":  1.10,  # Bonus: lizanie nosa
        },
    ),

    # SUBMISSION — priorytet 85
    # Uszy płaskie (EAD103), oczy zmrużone (AU145), ciało zniżone (proxy: uszy bardzo nisko)
    EmotionRule(
        emotion="submission",
        priority=85,
        required_aus={
            "EAD103": 1.20,  # Ears Flattener: uszy mocno przyciśnięte 20%+
            "EAD102": 1.10,  # Ears Adductor: uszy zbliżone do siebie
        },
        inhibitory_aus={
            "AU26":   1.15,  # NIE: otwarty pysk
            "AU12":   1.15,  # NIE: uśmiech
            "AU109":  1.15,  # NIE: zmarszczony nos
        },
        optional_aus={
            "AU145":  0.85,  # Bonus: mrugnięcie / przymknięte oczy
            "AD37":   1.10,  # Bonus: oblizywanie warg
        },
    ),

    # SAD — priorytet 80
    # Uszy lekko do tyłu (EAD103 słabo), brak aktywności reszty twarzy
    EmotionRule(
        emotion="sad",
        priority=80,
        required_aus={
            "EAD103": 1.10,  # Ears Flattener: uszy lekko przyciśnięte
        },
        inhibitory_aus={
            "AU26":   1.15,  # NIE: otwarty pysk
            "AU12":   1.10,  # NIE: uśmiech
            "EAD101": 1.10,  # NIE: uszy do przodu
        },
        optional_aus={
            "AU116": 1.10,  # Bonus: opuszczona dolna warga
        },
    ),

    # RELAXED — priorytet 70
    # Brak silnych aktywacji, uszy w normalnej pozycji
    EmotionRule(
        emotion="relaxed",
        priority=70,
        required_aus={},
        inhibitory_aus={
            "AU26":   1.15,  # NIE: otwarty pysk
            "EAD103": 1.10,  # NIE: uszy płaskie
            "EAD101": 1.10,  # NIE: uszy napięte do przodu
            "AU101":  1.10,  # NIE: napięte brwi
            "AU109":  1.15,  # NIE: zmarszczony nos
        },
        optional_aus={},
    ),

    # NEUTRAL — priorytet 50 (fallback, zawsze dopasowuje)
    EmotionRule(
        emotion="neutral",
        priority=50,
        required_aus={},
        inhibitory_aus={},
        optional_aus={},
        min_confidence=0.0,
    ),
]


class DogFACSRuleEngine:
    """
    Silnik reguł do klasyfikacji emocji psów metodą poselet matching.

    BRAK UCZENIA MASZYNOWEGO — używa wyłącznie reguł naukowych DogFACS.
    Reguły sprawdzane są w kolejności priorytetu (najwyższy pierwszy).
    Pierwsza pasująca reguła wygrywa.

    Przykład:
        >>> engine = DogFACSRuleEngine()
        >>> delta_aus = {...}  # słownik DeltaActionUnit
        >>> prediction = engine.classify(delta_aus)
        >>> print(f"Emocja: {prediction.emotion} (reguła: {prediction.rule_applied})")
    """

    def __init__(self, rules: Optional[list[EmotionRule]] = None) -> None:
        """
        Inicjalizuje silnik reguł.

        Args:
            rules: Lista obiektów EmotionRule (używa EMOTION_RULES jeśli None)
        """
        source_rules = rules if rules is not None else EMOTION_RULES
        self.rules = sorted(source_rules, key=lambda r: r.priority, reverse=True)

    def classify(
        self,
        delta_aus: dict[str, DeltaActionUnit],
    ) -> EmotionPrediction:
        """
        Klasyfikuje emocję używając dopasowania reguł według priorytetu.

        Args:
            delta_aus: Słownik AU_name → DeltaActionUnit

        Returns:
            EmotionPrediction z klasyfikowaną emocją
        """
        for rule in self.rules:
            matches, confidence = rule.matches(delta_aus)
            if matches:
                return self._build_prediction(rule, confidence, delta_aus)

        # Fallback — neutral z niską pewnością (nie powinno wystąpić)
        return EmotionPrediction(
            emotion_id=EMOTION_CLASSES.index("neutral"),
            emotion="neutral",
            confidence=0.3,
            probabilities={"neutral": 1.0},
            action_units={},
            rule_applied="fallback_neutral",
        )

    def _build_prediction(
        self,
        matched_rule: EmotionRule,
        confidence: float,
        delta_aus: dict[str, DeltaActionUnit],
    ) -> EmotionPrediction:
        """
        Buduje obiekt EmotionPrediction dla dopasowanej reguły.

        Args:
            matched_rule: Reguła która dopasowała
            confidence: Pewność dopasowania
            delta_aus: Słownik AU do obliczenia prawdopodobieństw

        Returns:
            EmotionPrediction
        """
        probabilities = {r.emotion: 0.0 for r in self.rules}
        probabilities[matched_rule.emotion] = confidence

        # Częściowe dopasowania innych reguł z obniżoną pewnością
        for other_rule in self.rules:
            if other_rule.emotion == matched_rule.emotion:
                continue
            partial_match, partial_conf = other_rule.matches(delta_aus)
            if partial_match:
                probabilities[other_rule.emotion] = partial_conf * 0.5

        # Normalizacja prawdopodobieństw
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        return EmotionPrediction(
            emotion_id=EMOTION_CLASSES.index(matched_rule.emotion),
            emotion=matched_rule.emotion,
            confidence=confidence,
            probabilities=probabilities,
            action_units={au.name: au.ratio for au in delta_aus.values()},
            rule_applied=f"{matched_rule.emotion}_priority_{matched_rule.priority}",
        )


def classify_emotion_from_delta_aus(
    delta_aus: dict[str, DeltaActionUnit],
    rule_engine: Optional[DogFACSRuleEngine] = None,
) -> EmotionPrediction:
    """
    Klasyfikuje emocję na podstawie delta Action Units.

    Funkcja pomocnicza używająca DogFACSRuleEngine.

    Args:
        delta_aus: Słownik AU_name → DeltaActionUnit
        rule_engine: Opcjonalny niestandardowy silnik reguł (tworzy domyślny jeśli None)

    Returns:
        EmotionPrediction

    Przykład:
        >>> from packages.models.delta_action_units import extract_delta_action_units
        >>> neutral_kp = np.zeros(138)
        >>> target_kp = np.zeros(138)
        >>> delta_aus = extract_delta_action_units(neutral_kp, target_kp)
        >>> prediction = classify_emotion_from_delta_aus(delta_aus)
    """
    engine = rule_engine if rule_engine is not None else DogFACSRuleEngine()
    return engine.classify(delta_aus)
