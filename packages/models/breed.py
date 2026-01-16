"""
Model klasyfikacji ras psów oparty na EfficientNet-B4.

Klasyfikuje rasy psów na podstawie wyciętego obrazu psa.
Zwraca Top-5 predykcji z prawdopodobieństwami.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .base import BaseModel, ModelConfig


@dataclass
class BreedConfig(ModelConfig):
    """
    Konfiguracja modelu klasyfikacji ras.

    Attributes:
        weights_path: Ścieżka do wag modelu (.pt)
        labels_path: Ścieżka do pliku breeds.json z mapowaniem
        device: Urządzenie ('cuda', 'cpu', etc.)
        model_name: Nazwa architektury z timm
        img_size: Rozmiar obrazu wejściowego
        top_k: Liczba najlepszych predykcji do zwrócenia
    """

    labels_path: Path | str = Path("packages/models/breeds.json")
    model_name: str = "efficientnet_b4"
    img_size: int = 224
    top_k: int = 5

    def __post_init__(self) -> None:
        """Konwertuje ścieżki do Path."""
        super().__post_init__()
        if isinstance(self.labels_path, str):
            self.labels_path = Path(self.labels_path)


@dataclass
class BreedPrediction:
    """
    Wynik predykcji rasy.

    Attributes:
        class_id: ID przewidywanej klasy (Top-1)
        class_name: Nazwa rasy (Top-1)
        confidence: Pewność predykcji (Top-1)
        top_k: Lista Top-K predykcji jako (class_id, class_name, confidence)
    """

    class_id: int
    class_name: str
    confidence: float
    top_k: list[tuple[int, str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Konwertuje predykcję do słownika."""
        return {
            "breed_id": self.class_id,
            "breed": self.class_name,
            "breed_confidence": self.confidence,
            "top_k": [
                {"id": id_, "name": name, "confidence": conf}
                for id_, name, conf in self.top_k
            ],
        }

    def to_coco(self) -> dict:
        """
        Zwraca dane w formacie kompatybilnym z COCO.

        Returns:
            Słownik z polami breed i breed_confidence
        """
        return {
            "breed": self.class_name,
            "breed_confidence": self.confidence,
        }


class BreedModel(BaseModel[np.ndarray, BreedPrediction]):
    """
    Model klasyfikacji ras psów oparty na EfficientNet-B4.

    Używa pretrenowanego lub fine-tunowanego modelu EfficientNet
    do klasyfikacji ras psów na wyciętych obrazach.

    Example:
        >>> config = BreedConfig(
        ...     weights_path="models/breed.pt",
        ...     labels_path="packages/models/breeds.json"
        ... )
        >>> model = BreedModel(config)
        >>> model.load()
        >>> cropped_dog = image[y:y+h, x:x+w]
        >>> prediction = model.predict(cropped_dog)
        >>> print(f"Breed: {prediction.class_name} ({prediction.confidence:.2%})")
    """

    def __init__(self, config: BreedConfig) -> None:
        """
        Inicjalizuje model klasyfikacji ras.

        Args:
            config: Konfiguracja modelu
        """
        super().__init__(config)
        self.config: BreedConfig = config
        self._model: Optional[nn.Module] = None
        self._labels: Optional[dict[str, str]] = None
        self._transform: Optional[object] = None

    @property
    def labels(self) -> dict[str, str]:
        """Zwraca mapowanie ID -> nazwa rasy."""
        if self._labels is None:
            raise RuntimeError("Model nie został załadowany")
        return self._labels

    @property
    def num_classes(self) -> int:
        """Zwraca liczbę klas."""
        return len(self._labels) if self._labels else 0

    def load(self) -> None:
        """
        Ładuje model i mapowanie ras.

        Raises:
            FileNotFoundError: Gdy plik z wagami lub labels nie istnieje
            ImportError: Gdy timm nie jest zainstalowany
        """
        if not self.config.weights_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono wag modelu: {self.config.weights_path}"
            )

        if not self.config.labels_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono mapowania ras: {self.config.labels_path}"
            )

        # Wczytaj mapowanie ras
        with open(self.config.labels_path, encoding="utf-8") as f:
            self._labels = json.load(f)

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
        num_classes = len(self._labels)
        self._model = timm.create_model(
            self.config.model_name,
            pretrained=False,
            num_classes=num_classes,
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
            transforms.Resize(int(self.config.img_size * 1.14)),
            transforms.CenterCrop(self.config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._loaded = True

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

        # Konwertuj BGR do RGB jeśli potrzeba (OpenCV używa BGR)
        if image.shape[2] == 3:
            # Zakładamy że może być BGR, ale transform i tak to obsłuży
            pass

        # Zastosuj transformacje
        tensor = self._transform(image)

        return tensor.unsqueeze(0)  # Dodaj batch dimension

    def predict(self, image: np.ndarray) -> BreedPrediction:
        """
        Klasyfikuje rasę psa na obrazie.

        Args:
            image: Wycięty obraz psa jako numpy array

        Returns:
            Obiekt BreedPrediction z wynikami

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

        # Top-K
        top_k_probs, top_k_indices = probs.topk(self.config.top_k)

        top_k_results = []
        for idx, prob in zip(top_k_indices.cpu().numpy(), top_k_probs.cpu().numpy()):
            class_name = self._labels.get(str(idx), f"Unknown_{idx}")
            top_k_results.append((int(idx), class_name, float(prob)))

        # Top-1
        top1_id, top1_name, top1_conf = top_k_results[0]

        return BreedPrediction(
            class_id=top1_id,
            class_name=top1_name,
            confidence=top1_conf,
            top_k=top_k_results,
        )

    def postprocess(self, prediction: BreedPrediction) -> dict:
        """
        Konwertuje predykcję do słownika.

        Args:
            prediction: Obiekt BreedPrediction

        Returns:
            Słownik z wynikami
        """
        return prediction.to_dict()

    def predict_batch(self, images: list[np.ndarray]) -> list[BreedPrediction]:
        """
        Klasyfikuje batch obrazów.

        Args:
            images: Lista wyciętych obrazów psów

        Returns:
            Lista predykcji
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model nie został załadowany.")

        # Preprocess batch
        tensors = torch.stack([self.preprocess(img).squeeze(0) for img in images])

        device = next(self._model.parameters()).device
        tensors = tensors.to(device)

        # Inference
        with torch.no_grad():
            logits = self._model(tensors)
            probs = torch.softmax(logits, dim=1)

        # Wyniki
        predictions = []
        for i in range(len(images)):
            top_k_probs, top_k_indices = probs[i].topk(self.config.top_k)

            top_k_results = []
            for idx, prob in zip(top_k_indices.cpu().numpy(), top_k_probs.cpu().numpy()):
                class_name = self._labels.get(str(idx), f"Unknown_{idx}")
                top_k_results.append((int(idx), class_name, float(prob)))

            top1_id, top1_name, top1_conf = top_k_results[0]

            predictions.append(BreedPrediction(
                class_id=top1_id,
                class_name=top1_name,
                confidence=top1_conf,
                top_k=top_k_results,
            ))

        return predictions

    def get_breed_name(self, class_id: int) -> str:
        """
        Zwraca nazwę rasy dla danego ID.

        Args:
            class_id: ID klasy

        Returns:
            Nazwa rasy
        """
        if self._labels is None:
            raise RuntimeError("Model nie został załadowany")
        return self._labels.get(str(class_id), f"Unknown_{class_id}")
