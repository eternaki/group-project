"""
Model klasyfikacji emocji psów oparty na keypoints i Action Units.

Architektura zgodna z DogFACS:
    Keypoints (20 * 3 = 60) → Action Units (12) → łącznie 72 features
    72 features → MLP → 5 klas trenowanych
    Neutral jest wykrywany gdy żadna emocja nie ma wysokiej pewności.

Action Units (AU) to obiektywne pomiary ruchów mięśni twarzy,
zgodnie z Dog Facial Action Coding System (DogFACS).
Emocje są interpretacjami kombinacji AU.

Źródło: https://www.animalfacs.com/dogfacs
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .base import BaseModel, ModelConfig
from packages.data.schemas import NUM_KEYPOINTS
from .action_units import (
    ActionUnitsExtractor,
    NUM_ACTION_UNITS,
    ACTION_UNIT_NAMES,
    extract_action_units,
)


# Klasy emocji trenowane przez model (5 klas z dostępnych datasetów)
EMOTION_CLASSES_TRAINED = ['happy', 'sad', 'angry', 'fearful', 'relaxed']
NUM_EMOTIONS_TRAINED = len(EMOTION_CLASSES_TRAINED)

# Wszystkie klasy emocji (6 klas - neutral wykrywany przez próg)
EMOTION_CLASSES = EMOTION_CLASSES_TRAINED + ['neutral']
NUM_EMOTIONS = len(EMOTION_CLASSES)

# Indeks klasy neutral
NEUTRAL_ID = EMOTION_CLASSES.index('neutral')  # 5

# Liczba cech wejściowych
KEYPOINTS_FEATURES = NUM_KEYPOINTS * 3  # 20 * 3 = 60
AU_FEATURES = NUM_ACTION_UNITS           # 12
INPUT_FEATURES = KEYPOINTS_FEATURES + AU_FEATURES  # 60 + 12 = 72

# Dla kompatybilności wstecznej (tylko keypoints)
INPUT_FEATURES_LEGACY = KEYPOINTS_FEATURES  # 60


@dataclass
class EmotionConfig(ModelConfig):
    """
    Konfiguracja modelu klasyfikacji emocji.

    Attributes:
        weights_path: Sciezka do wag modelu (.pt)
        device: Urzadzenie ('cuda', 'cpu', etc.)
        hidden_dims: Wymiary warstw ukrytych MLP
        dropout: Prawdopodobienstwo dropout
        neutral_threshold: Próg pewności poniżej którego emocja = neutral
        use_action_units: Czy używać Action Units jako dodatkowych cech
    """

    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3
    neutral_threshold: float = 0.35  # Jeśli max_prob < 0.35 → neutral
    use_action_units: bool = True    # Używaj AU (zgodne z DogFACS)


@dataclass
class EmotionPrediction:
    """
    Wynik predykcji emocji.

    Attributes:
        emotion_id: ID przewidywanej emocji
        emotion: Nazwa emocji
        confidence: Pewnosc predykcji
        probabilities: Prawdopodobienstwa wszystkich klas
        action_units: Wartosci Action Units (opcjonalne)
    """

    emotion_id: int
    emotion: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    action_units: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Konwertuje predykcje do slownika."""
        result = {
            "emotion_id": self.emotion_id,
            "emotion": self.emotion,
            "emotion_confidence": self.confidence,
            "probabilities": self.probabilities,
        }
        if self.action_units:
            result["action_units"] = self.action_units
        return result

    def to_coco(self) -> dict:
        """
        Zwraca dane w formacie kompatybilnym z COCO.

        Returns:
            Slownik z polami emotion i emotion_confidence
        """
        return {
            "emotion": self.emotion,
            "emotion_confidence": self.confidence,
        }


class KeypointsEmotionMLP(nn.Module):
    """
    MLP do klasyfikacji emocji na podstawie keypoints + Action Units.

    Architektura (z AU):
        Input (72) → FC(256) → BN → ReLU → Dropout
                   → FC(128) → BN → ReLU → Dropout
                   → FC(64) → BN → ReLU → Dropout
                   → FC → Output (5 klas trenowanych)

    Gdzie Input = Keypoints (60) + Action Units (12) = 72 features

    Neutral jest wykrywany przez EmotionModel gdy max(prob) < threshold.
    """

    def __init__(
        self,
        input_dim: int = INPUT_FEATURES,  # 72 z AU, 60 bez
        hidden_dims: list[int] = None,
        num_classes: int = NUM_EMOTIONS_TRAINED,  # 5 klas trenowanych
        dropout: float = 0.3,
    ) -> None:
        """
        Inicjalizuje MLP.

        Args:
            input_dim: Wymiar wejsciowy (72 z AU, 60 bez AU)
            hidden_dims: Wymiary warstw ukrytych
            num_classes: Liczba klas wyjsciowych
            dropout: Prawdopodobienstwo dropout
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        # Warstwy ukryte
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Warstwa wyjsciowa
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor o ksztalcie (batch, 138)

        Returns:
            Logity o ksztalcie (batch, 6)
        """
        return self.network(x)


class EmotionModel(BaseModel[np.ndarray, EmotionPrediction]):
    """
    Model klasyfikacji emocji psów oparty na keypoints.

    Uzywa MLP do klasyfikacji emocji na podstawie 46 keypoints.
    Jest to zgodne z podejsciem DogFACS.

    Example:
        >>> config = EmotionConfig(weights_path="models/emotion_keypoints.pt")
        >>> model = EmotionModel(config)
        >>> model.load()
        >>> # keypoints_flat: [x0, y0, v0, x1, y1, v1, ..., x45, y45, v45]
        >>> prediction = model.predict(keypoints_flat)
        >>> print(f"Emotion: {prediction.emotion} ({prediction.confidence:.2%})")
    """

    def __init__(self, config: EmotionConfig) -> None:
        """
        Inicjalizuje model klasyfikacji emocji.

        Args:
            config: Konfiguracja modelu
        """
        super().__init__(config)
        self.config: EmotionConfig = config
        self._model: Optional[nn.Module] = None

    def load(self) -> None:
        """
        Laduje model MLP.

        Raises:
            FileNotFoundError: Gdy plik z wagami nie istnieje
        """
        # Inicjalizuj ekstraktor Action Units
        self._au_extractor = ActionUnitsExtractor() if self.config.use_action_units else None

        # Określ wymiar wejściowy
        if self.config.use_action_units:
            input_dim = INPUT_FEATURES  # 72 (keypoints + AU)
        else:
            input_dim = INPUT_FEATURES_LEGACY  # 60 (tylko keypoints)

        # Utworz model (5 klas trenowanych, neutral przez próg)
        self._model = KeypointsEmotionMLP(
            input_dim=input_dim,
            hidden_dims=self.config.hidden_dims,
            num_classes=NUM_EMOTIONS_TRAINED,  # 5 klas
            dropout=self.config.dropout,
        )

        # Zaladuj wagi jesli istnieja
        if self.config.weights_path.exists():
            device = torch.device(
                self.config.device if torch.cuda.is_available() else "cpu"
            )
            state_dict = torch.load(self.config.weights_path, map_location=device)
            self._model.load_state_dict(state_dict)
            self._model = self._model.to(device)
            au_status = "z Action Units" if self.config.use_action_units else "bez Action Units"
            print(f"Model emocji zaladowany ({au_status}): {self.config.weights_path}")
        else:
            # Model bez wag - uzyjemy losowych wag (do treningu)
            device = torch.device(
                self.config.device if torch.cuda.is_available() else "cpu"
            )
            self._model = self._model.to(device)
            au_status = "z Action Units" if self.config.use_action_units else "bez Action Units"
            print(f"Model emocji zainicjalizowany ({au_status}, bez wag)")
            print(f"  ! UWAGA: Model wymaga treningu przed uzyciem produkcyjnym")

        self._model.eval()
        self._loaded = True

    def preprocess(self, keypoints_flat: list[float] | np.ndarray) -> torch.Tensor:
        """
        Przetwarza keypoints do tensora, opcjonalnie dodając Action Units.

        Args:
            keypoints_flat: Lista lub array [x0, y0, v0, x1, y1, v1, ...]
                           Dlugosc: 60 (20 keypoints * 3)

        Returns:
            Tensor gotowy do inference (60 lub 72 features)

        Raises:
            ValueError: Gdy keypoints maja nieprawidlowy format
        """
        if keypoints_flat is None:
            raise ValueError("Keypoints nie moga byc None")

        if isinstance(keypoints_flat, list):
            keypoints_flat = np.array(keypoints_flat, dtype=np.float32)

        if len(keypoints_flat) != KEYPOINTS_FEATURES:
            raise ValueError(
                f"Oczekiwano {KEYPOINTS_FEATURES} wartosci keypoints, "
                f"otrzymano: {len(keypoints_flat)}"
            )

        # Dodaj Action Units jeśli włączone
        if self.config.use_action_units and self._au_extractor is not None:
            au_features = extract_action_units(keypoints_flat)
            features = np.concatenate([keypoints_flat, au_features])
        else:
            features = keypoints_flat

        tensor = torch.from_numpy(features).float()
        return tensor.unsqueeze(0)  # Dodaj batch dimension

    def predict(self, keypoints_flat: list[float] | np.ndarray) -> EmotionPrediction:
        """
        Klasyfikuje emocje psa na podstawie keypoints i Action Units.

        Model zwraca 5 klas trenowanych. Neutral jest wykrywany gdy
        żadna emocja nie ma pewności powyżej neutral_threshold.

        Args:
            keypoints_flat: Keypoints w formacie flat [x0, y0, v0, ...] (60 wartości)

        Returns:
            Obiekt EmotionPrediction z wynikami (jedna z 6 klas) + Action Units

        Raises:
            RuntimeError: Gdy model nie zostal zaladowany
        """
        if not self._loaded or self._model is None:
            raise RuntimeError(
                "Model nie zostal zaladowany. Wywolaj load() przed predict()."
            )

        if isinstance(keypoints_flat, list):
            keypoints_flat = np.array(keypoints_flat, dtype=np.float32)

        # Ekstrahuj Action Units (dla wyników, niezależnie od use_action_units)
        au_values = {}
        if self._au_extractor is not None:
            au_prediction = self._au_extractor.extract(keypoints_flat)
            au_values = au_prediction.values

        # Preprocess (dodaje AU do features jeśli use_action_units=True)
        tensor = self.preprocess(keypoints_flat)

        # Przenies na device
        device = next(self._model.parameters()).device
        tensor = tensor.to(device)

        # Inference
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]  # 5 prawdopodobieństw

        # Najlepsza predykcja z 5 trenowanych klas
        top_prob, top_idx = probs.max(0)
        top_idx = int(top_idx.cpu().numpy())
        top_prob = float(top_prob.cpu().numpy())

        # Prawdopodobieństwa dla 5 trenowanych klas
        probabilities = {
            EMOTION_CLASSES_TRAINED[i]: float(probs[i].cpu().numpy())
            for i in range(NUM_EMOTIONS_TRAINED)
        }

        # Wykryj neutral: jeśli max(prob) < threshold → neutral
        if top_prob < self.config.neutral_threshold:
            # Neutral - oblicz "pewność" jako odwrotność max_prob
            neutral_confidence = 1.0 - top_prob
            probabilities['neutral'] = neutral_confidence
            # Przeskaluj pozostałe prawdopodobieństwa
            scale = (1.0 - neutral_confidence)
            for emotion in EMOTION_CLASSES_TRAINED:
                probabilities[emotion] *= scale

            return EmotionPrediction(
                emotion_id=NEUTRAL_ID,
                emotion='neutral',
                confidence=neutral_confidence,
                probabilities=probabilities,
                action_units=au_values,
            )

        # Nie neutral - dodaj zerową wartość dla neutral
        probabilities['neutral'] = 0.0

        return EmotionPrediction(
            emotion_id=top_idx,
            emotion=EMOTION_CLASSES_TRAINED[top_idx],
            confidence=top_prob,
            probabilities=probabilities,
            action_units=au_values,
        )

    def predict_from_keypoints_prediction(
        self,
        keypoints_prediction: "KeypointsPrediction",
    ) -> EmotionPrediction:
        """
        Klasyfikuje emocje na podstawie obiektu KeypointsPrediction.

        Args:
            keypoints_prediction: Obiekt KeypointsPrediction z modelu keypoints

        Returns:
            Obiekt EmotionPrediction z wynikami
        """
        # Konwertuj do formatu flat
        keypoints_flat = keypoints_prediction.to_coco_format()
        return self.predict(keypoints_flat)

    def postprocess(self, prediction: EmotionPrediction) -> dict:
        """
        Konwertuje predykcje do slownika.

        Args:
            prediction: Obiekt EmotionPrediction

        Returns:
            Slownik z wynikami
        """
        return prediction.to_dict()

    def get_emotion_name(self, emotion_id: int) -> str:
        """
        Zwraca nazwe emocji dla danego ID.

        Args:
            emotion_id: ID emocji

        Returns:
            Nazwa emocji
        """
        if 0 <= emotion_id < NUM_EMOTIONS:
            return EMOTION_CLASSES[emotion_id]
        return f"Unknown_{emotion_id}"


# Import dla type hints
try:
    from .keypoints import KeypointsPrediction
except ImportError:
    pass


# =============================================================================
# RULE-BASED EMOTION CLASSIFICATION (DogFACS approach)
# =============================================================================

# Mapowanie naszych AU na AU z planu (QUICK_IMPLEMENTATION_PLAN.md)
# Nasze AU → Plan AU:
#   AU_brow_raise → AU101 (Inner Brow Raiser)
#   AU_eye_opening → AU145 inverted (Blink = 1 - eye_opening)
#   AU_mouth_open → AU25 (Lips Part)
#   AU_jaw_drop → AU26 (Jaw Drop)
#   AU_nose_wrinkle → AU301 (Nose Wrinkler)
#   AU_lip_corner_pull → AU401 (Upper Lip Raiser)
#   AU_ear_forward → EAD102 (Ears Forward)
#   AU_ear_back → EAD103 (Ears Flattener/Back)
#   AU_ear_asymmetry → EAD104 (Ears Rotator)
#   AD137 (Nose Lick) - trudne do detekcji z keypoints, używamy proxy


def classify_emotion_from_au(
    au_values: dict[str, float],
    neutral_threshold: float = 0.35,
) -> EmotionPrediction:
    """
    Rule-based klasyfikacja emocji na podstawie Action Units.

    Używa wzorów z QUICK_IMPLEMENTATION_PLAN.md opartych na badaniach
    DogFACS (Mota-Rojas et al. 2021).

    NIE wymaga treningu - oparta na regułach naukowych.

    Args:
        au_values: Słownik AU_name → wartość (0-1)
        neutral_threshold: Próg poniżej którego emocja = neutral

    Returns:
        EmotionPrediction z przewidywaną emocją

    Example:
        >>> au_values = {
        ...     'AU_brow_raise': 0.3,
        ...     'AU_ear_forward': 0.8,
        ...     'AU_mouth_open': 0.6,
        ...     # ... pozostałe AU
        ... }
        >>> prediction = classify_emotion_from_au(au_values)
        >>> print(f"Emotion: {prediction.emotion}")
    """
    # Pobierz wartości AU (z domyślnym 0.0 jeśli brak)
    brow_raise = au_values.get('AU_brow_raise', 0.0)       # AU101
    eye_opening = au_values.get('AU_eye_opening', 0.5)     # AU145 = 1 - eye_opening
    mouth_open = au_values.get('AU_mouth_open', 0.0)       # AU25
    jaw_drop = au_values.get('AU_jaw_drop', 0.0)           # AU26
    nose_wrinkle = au_values.get('AU_nose_wrinkle', 0.0)   # AU301
    lip_corner_pull = au_values.get('AU_lip_corner_pull', 0.0)  # AU401
    ear_forward = au_values.get('AU_ear_forward', 0.0)     # EAD102
    ear_back = au_values.get('AU_ear_back', 0.0)           # EAD103
    ear_asymmetry = au_values.get('AU_ear_asymmetry', 0.0) # EAD104

    # AU145 (Blink) = 1 - eye_opening
    blink = 1.0 - eye_opening

    # AD137 (Nose Lick) - proxy: używamy kombinacji mouth_open i jaw_drop
    # Lizanie zwykle wiąże się z ruchem ust bez pełnego otwarcia szczęki
    nose_lick = max(0.0, mouth_open * 0.5 - jaw_drop * 0.3)

    # Oblicz średnią aktywację AU (dla relaxed i neutral)
    all_au_values = [
        brow_raise, blink, mouth_open, jaw_drop, nose_wrinkle,
        lip_corner_pull, ear_forward, ear_back, ear_asymmetry, nose_lick
    ]
    mean_activation = sum(all_au_values) / len(all_au_values)

    # ==========================================================================
    # SCORING EMOCJI (wg QUICK_IMPLEMENTATION_PLAN.md)
    # ==========================================================================

    # HAPPY: Otwarty puch, uszy do przodu, brwi uniesione
    happy_score = (
        mouth_open * 0.35 +           # AU25 - Rót otwarty
        ear_forward * 0.25 +          # EAD102 - Uszy do przodu
        brow_raise * 0.15 +           # AU101 - Brwi uniesione
        (1 - ear_back) * 0.15 +       # Uszy NIE są płaskie
        (1 - nose_lick) * 0.10        # Brak stresu (lizania)
    )

    # SAD: Uszy płaskie, zamknięte oczy, zamknięty pysk
    sad_score = (
        ear_back * 0.40 +             # EAD103 - Uszy płaskie
        blink * 0.15 +                # AU145 - Przymknięte oczy
        (1 - brow_raise) * 0.15 +     # Brwi NIE uniesione
        (1 - mouth_open) * 0.15 +     # Pysk zamknięty
        nose_lick * 0.15              # Lizanie (stres)
    )

    # ANGRY: Otwarta paszcza, obnażone zęby, zmarszczony nos
    angry_score = (
        ((mouth_open + jaw_drop) / 2) * 0.30 +  # Otwarta paszcza
        lip_corner_pull * 0.25 +       # AU401 - Obnażone zęby
        nose_wrinkle * 0.15 +          # AU301 - Zmarszczony nos
        ((ear_back + ear_asymmetry) / 2) * 0.15 +  # Uszy w pozycji agresji
        blink * 0.15                   # Zmrużone oczy
    )

    # FEARFUL: Uszy płaskie, lizanie (główny wskaźnik stresu), mruganie
    fearful_score = (
        ear_back * 0.30 +              # EAD103 - Uszy płaskie
        nose_lick * 0.25 +             # AD137 - Lizanie (GŁÓWNY wskaźnik stresu)
        blink * 0.20 +                 # AU145 - Częste mruganie
        brow_raise * 0.12 +            # AU101 - Brwi uniesione (zaskoczenie)
        (1 - mouth_open) * 0.13        # Pysk zamknięty
    )

    # RELAXED: Minimalna aktywacja wszystkich AU
    relaxed_score = (
        (1 - mean_activation) * 0.50 +  # Minimalna aktywacja
        (1 - ear_back) * 0.15 +         # Uszy NIE płaskie
        (1 - ear_forward) * 0.15 +      # Uszy NIE do przodu (neutralne)
        (1 - nose_lick) * 0.10 +        # Brak stresu
        (1 - nose_wrinkle) * 0.10       # Nos NIE zmarszczony
    )

    # NEUTRAL: Baseline - bardzo niska aktywacja
    neutral_score = (
        (1 - mean_activation) * 0.70 +
        (1 - (mouth_open + jaw_drop) / 2) * 0.15 +
        (1 - (ear_back + ear_forward) / 2) * 0.15
    )

    # ==========================================================================
    # NORMALIZACJA I WYBÓR EMOCJI
    # ==========================================================================

    scores = {
        'happy': happy_score,
        'sad': sad_score,
        'angry': angry_score,
        'fearful': fearful_score,
        'relaxed': relaxed_score,
        'neutral': neutral_score,
    }

    # Softmax-like normalizacja do prawdopodobieństw
    # Używamy temperature=2.0 dla bardziej "pewnych" predykcji
    temperature = 2.0
    exp_scores = {k: np.exp(v * temperature) for k, v in scores.items()}
    total_exp = sum(exp_scores.values())
    probabilities = {k: v / total_exp for k, v in exp_scores.items()}

    # Znajdź najlepszą emocję
    best_emotion = max(probabilities, key=probabilities.get)
    best_prob = probabilities[best_emotion]

    # Jeśli pewność jest za niska → neutral
    if best_prob < neutral_threshold and best_emotion != 'neutral':
        best_emotion = 'neutral'
        best_prob = probabilities['neutral']

    # Znajdź emotion_id
    emotion_id = EMOTION_CLASSES.index(best_emotion)

    return EmotionPrediction(
        emotion_id=emotion_id,
        emotion=best_emotion,
        confidence=best_prob,
        probabilities=probabilities,
        action_units=au_values,
    )


def classify_emotion_from_keypoints(
    keypoints_flat: np.ndarray,
    neutral_threshold: float = 0.35,
) -> EmotionPrediction:
    """
    Rule-based klasyfikacja emocji bezpośrednio z keypoints.

    Convenience function łącząca ekstrakcję AU i klasyfikację.

    Args:
        keypoints_flat: Array [x0, y0, v0, ...] (60 wartości)
        neutral_threshold: Próg dla neutral

    Returns:
        EmotionPrediction

    Example:
        >>> keypoints = np.array([...])  # 60 wartości
        >>> prediction = classify_emotion_from_keypoints(keypoints)
    """
    # Ekstrahuj AU z keypoints
    au_prediction = extract_action_units(keypoints_flat)

    # Konwertuj numpy array na dict
    au_dict = {
        ACTION_UNIT_NAMES[i]: float(au_prediction[i])
        for i in range(len(ACTION_UNIT_NAMES))
    }

    # Klasyfikuj
    return classify_emotion_from_au(au_dict, neutral_threshold)
