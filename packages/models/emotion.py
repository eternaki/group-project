"""
Model klasyfikacji emocji psów oparty na keypoints.

Klasyfikuje emocje psów na podstawie 46 keypoints z modelu KeypointsModel.
Zwraca 6 klas emocji: happy, sad, angry, fearful, relaxed, neutral.

Architektura:
    Keypoints (46 * 3 = 138 features) → MLP → 6 klas emocji

Jest to zgodne z podejściem DogFACS, gdzie emocje są pochodną
ruchów mięśni twarzy (Action Units), a nie bezpośrednio pikseli.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .base import BaseModel, ModelConfig
from packages.data.schemas import NUM_KEYPOINTS


# Klasy emocji (6 klas zgodnie z dokumentacją DogFACS)
EMOTION_CLASSES = ['happy', 'sad', 'angry', 'fearful', 'relaxed', 'neutral']
NUM_EMOTIONS = len(EMOTION_CLASSES)

# Liczba cech wejściowych: x, y, visibility dla każdego keypointa
INPUT_FEATURES = NUM_KEYPOINTS * 3  # 46 * 3 = 138


@dataclass
class EmotionConfig(ModelConfig):
    """
    Konfiguracja modelu klasyfikacji emocji.

    Attributes:
        weights_path: Sciezka do wag modelu (.pt)
        device: Urzadzenie ('cuda', 'cpu', etc.)
        hidden_dims: Wymiary warstw ukrytych MLP
        dropout: Prawdopodobienstwo dropout
    """

    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3


@dataclass
class EmotionPrediction:
    """
    Wynik predykcji emocji.

    Attributes:
        emotion_id: ID przewidywanej emocji
        emotion: Nazwa emocji
        confidence: Pewnosc predykcji
        probabilities: Prawdopodobienstwa wszystkich klas
    """

    emotion_id: int
    emotion: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Konwertuje predykcje do slownika."""
        return {
            "emotion_id": self.emotion_id,
            "emotion": self.emotion,
            "emotion_confidence": self.confidence,
            "probabilities": self.probabilities,
        }

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
    MLP do klasyfikacji emocji na podstawie keypoints.

    Architektura:
        Input (138) → FC → ReLU → Dropout → FC → ReLU → Dropout → FC → Output (6)
    """

    def __init__(
        self,
        input_dim: int = INPUT_FEATURES,
        hidden_dims: list[int] = None,
        num_classes: int = NUM_EMOTIONS,
        dropout: float = 0.3,
    ) -> None:
        """
        Inicjalizuje MLP.

        Args:
            input_dim: Wymiar wejsciowy (domyslnie 138)
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
        # Utworz model
        self._model = KeypointsEmotionMLP(
            input_dim=INPUT_FEATURES,
            hidden_dims=self.config.hidden_dims,
            num_classes=NUM_EMOTIONS,
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
            print(f"Model emocji zaladowany: {self.config.weights_path}")
        else:
            # Model bez wag - uzyjemy losowych wag (do treningu)
            device = torch.device(
                self.config.device if torch.cuda.is_available() else "cpu"
            )
            self._model = self._model.to(device)
            print(f"Model emocji zainicjalizowany (bez wag)")
            print(f"  ! UWAGA: Model wymaga treningu przed uzyciem produkcyjnym")

        self._model.eval()
        self._loaded = True

    def preprocess(self, keypoints_flat: list[float] | np.ndarray) -> torch.Tensor:
        """
        Przetwarza keypoints do tensora.

        Args:
            keypoints_flat: Lista lub array [x0, y0, v0, x1, y1, v1, ...]
                           Dlugosc: 138 (46 keypoints * 3)

        Returns:
            Tensor gotowy do inference

        Raises:
            ValueError: Gdy keypoints maja nieprawidlowy format
        """
        if keypoints_flat is None:
            raise ValueError("Keypoints nie moga byc None")

        if isinstance(keypoints_flat, list):
            keypoints_flat = np.array(keypoints_flat, dtype=np.float32)

        if len(keypoints_flat) != INPUT_FEATURES:
            raise ValueError(
                f"Oczekiwano {INPUT_FEATURES} wartosci, otrzymano: {len(keypoints_flat)}"
            )

        # Normalizacja (opcjonalna - mozna dostosowac)
        # Zakladamy ze x, y sa w zakresie [0, crop_size]
        # visibility jest w zakresie [0, 1]
        tensor = torch.from_numpy(keypoints_flat).float()

        return tensor.unsqueeze(0)  # Dodaj batch dimension

    def predict(self, keypoints_flat: list[float] | np.ndarray) -> EmotionPrediction:
        """
        Klasyfikuje emocje psa na podstawie keypoints.

        Args:
            keypoints_flat: Keypoints w formacie flat [x0, y0, v0, ...]

        Returns:
            Obiekt EmotionPrediction z wynikami

        Raises:
            RuntimeError: Gdy model nie zostal zaladowany
        """
        if not self._loaded or self._model is None:
            raise RuntimeError(
                "Model nie zostal zaladowany. Wywolaj load() przed predict()."
            )

        # Preprocess
        tensor = self.preprocess(keypoints_flat)

        # Przenies na device
        device = next(self._model.parameters()).device
        tensor = tensor.to(device)

        # Inference
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # Najlepsza predykcja
        top_prob, top_idx = probs.max(0)
        top_idx = int(top_idx.cpu().numpy())
        top_prob = float(top_prob.cpu().numpy())

        # Wszystkie prawdopodobienstwa
        probabilities = {
            EMOTION_CLASSES[i]: float(probs[i].cpu().numpy())
            for i in range(NUM_EMOTIONS)
        }

        return EmotionPrediction(
            emotion_id=top_idx,
            emotion=EMOTION_CLASSES[top_idx],
            confidence=top_prob,
            probabilities=probabilities,
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
