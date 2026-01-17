"""
Model klasyfikacji emocji psów oparty na EfficientNet-B0.

Klasyfikuje emocje psów na podstawie wyciętego obrazu psa.
Zwraca 4 klasy emocji: sad, angry, relaxed, happy.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .base import BaseModel, ModelConfig


# Klasy emocji
EMOTION_CLASSES = ['sad', 'angry', 'relaxed', 'happy']
NUM_EMOTIONS = len(EMOTION_CLASSES)


@dataclass
class EmotionConfig(ModelConfig):
    """
    Konfiguracja modelu klasyfikacji emocji.

    Attributes:
        weights_path: Ścieżka do wag modelu (.pt)
        device: Urządzenie ('cuda', 'cpu', etc.)
        model_name: Nazwa architektury z timm
        img_size: Rozmiar obrazu wejściowego
    """

    model_name: str = "efficientnet_b0"
    img_size: int = 224


@dataclass
class EmotionPrediction:
    """
    Wynik predykcji emocji.

    Attributes:
        emotion_id: ID przewidywanej emocji
        emotion: Nazwa emocji
        confidence: Pewność predykcji
        probabilities: Prawdopodobieństwa wszystkich klas
    """

    emotion_id: int
    emotion: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Konwertuje predykcję do słownika."""
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
            Słownik z polami emotion i emotion_confidence
        """
        return {
            "emotion": self.emotion,
            "emotion_confidence": self.confidence,
        }


class EmotionModel(BaseModel[np.ndarray, EmotionPrediction]):
    """
    Model klasyfikacji emocji psów oparty na EfficientNet-B0.

    Używa modelu fine-tunowanego na datasecie Dog Emotion.

    Example:
        >>> config = EmotionConfig(weights_path="models/emotion.pt")
        >>> model = EmotionModel(config)
        >>> model.load()
        >>> cropped_dog = image[y:y+h, x:x+w]
        >>> prediction = model.predict(cropped_dog)
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
        self._transform: Optional[object] = None

    def load(self) -> None:
        """
        Ładuje model EfficientNet.

        Raises:
            FileNotFoundError: Gdy plik z wagami nie istnieje
            ImportError: Gdy timm nie jest zainstalowany
        """
        if not self.config.weights_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono wag modelu: {self.config.weights_path}"
            )

        # Import timm
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "Biblioteka timm nie jest zainstalowana. "
                "Zainstaluj: pip install timm"
            ) from e

        # Import torchvision transforms
        try:
            from torchvision import transforms
        except ImportError as e:
            raise ImportError(
                "Biblioteka torchvision nie jest zainstalowana. "
                "Zainstaluj: pip install torchvision"
            ) from e

        # Utwórz model
        self._model = timm.create_model(
            self.config.model_name,
            pretrained=False,
            num_classes=NUM_EMOTIONS,
        )

        # Załaduj wagi
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(self.config.weights_path, map_location=device)
        self._model.load_state_dict(state_dict)
        self._model = self._model.to(device)
        self._model.eval()

        # Przygotuj transformacje
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._loaded = True
        print(f"Model emocji załadowany: {self.config.weights_path}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Przetwarza obraz do tensora.

        Args:
            image: Obraz jako numpy array (BGR lub RGB)

        Returns:
            Tensor gotowy do inference

        Raises:
            ValueError: Gdy obraz jest nieprawidłowy
        """
        if image is None or image.size == 0:
            raise ValueError("Obraz nie może być pusty")

        if len(image.shape) != 3:
            raise ValueError(
                f"Obraz musi mieć 3 wymiary (H, W, C), otrzymano: {image.shape}"
            )

        # Zastosuj transformacje
        tensor = self._transform(image)

        return tensor.unsqueeze(0)  # Dodaj batch dimension

    def predict(self, image: np.ndarray) -> EmotionPrediction:
        """
        Klasyfikuje emocję psa na obrazie.

        Args:
            image: Wycięty obraz psa jako numpy array

        Returns:
            Obiekt EmotionPrediction z wynikami

        Raises:
            RuntimeError: Gdy model nie został załadowany
        """
        if not self._loaded or self._model is None:
            raise RuntimeError(
                "Model nie został załadowany. Wywołaj load() przed predict()."
            )

        # Preprocess
        tensor = self.preprocess(image)

        # Przenieś na device
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

        # Wszystkie prawdopodobieństwa
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

    def postprocess(self, prediction: EmotionPrediction) -> dict:
        """
        Konwertuje predykcję do słownika.

        Args:
            prediction: Obiekt EmotionPrediction

        Returns:
            Słownik z wynikami
        """
        return prediction.to_dict()

    def get_emotion_name(self, emotion_id: int) -> str:
        """
        Zwraca nazwę emocji dla danego ID.

        Args:
            emotion_id: ID emocji

        Returns:
            Nazwa emocji
        """
        if 0 <= emotion_id < NUM_EMOTIONS:
            return EMOTION_CLASSES[emotion_id]
        return f"Unknown_{emotion_id}"
