"""
Bazowe klasy dla modeli AI w projekcie Dog FACS.

Definiuje interfejs dla wszystkich modeli w pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import numpy as np

# Typ dla danych wejściowych i wyjściowych
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class ModelConfig:
    """
    Bazowa konfiguracja modelu.

    Attributes:
        weights_path: Ścieżka do pliku z wagami modelu
        device: Urządzenie do inference ('cuda', 'cpu', 'mps', '0', '1', etc.)
        half: Czy używać FP16 (tylko GPU)
    """

    weights_path: Path | str
    device: str = "cuda"
    half: bool = False

    def __post_init__(self) -> None:
        """Konwertuje ścieżkę do Path."""
        if isinstance(self.weights_path, str):
            self.weights_path = Path(self.weights_path)


class BaseModel(ABC, Generic[InputT, OutputT]):
    """
    Abstrakcyjna klasa bazowa dla wszystkich modeli.

    Definiuje interfejs: load(), preprocess(), predict(), postprocess().
    Każdy model musi implementować te metody.

    Example:
        >>> config = BBoxConfig(weights_path="models/bbox.pt")
        >>> model = BBoxModel(config)
        >>> model.load()
        >>> detections = model.predict(image)
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Inicjalizuje model z konfiguracją.

        Args:
            config: Konfiguracja modelu
        """
        self.config = config
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Sprawdza czy model został załadowany."""
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """
        Ładuje wagi modelu.

        Raises:
            FileNotFoundError: Gdy plik z wagami nie istnieje
            RuntimeError: Gdy ładowanie się nie powiedzie
        """
        pass

    @abstractmethod
    def preprocess(self, data: InputT) -> Any:
        """
        Przetwarza dane wejściowe przed inference.

        Args:
            data: Surowe dane wejściowe

        Returns:
            Dane przygotowane do inference
        """
        pass

    @abstractmethod
    def predict(self, data: InputT) -> OutputT:
        """
        Wykonuje predykcję na danych.

        Args:
            data: Dane wejściowe (surowe lub przetworzone)

        Returns:
            Wynik predykcji

        Raises:
            RuntimeError: Gdy model nie został załadowany
        """
        pass

    @abstractmethod
    def postprocess(self, output: OutputT) -> dict | list[dict]:
        """
        Przetwarza wynik predykcji do formatu słownika.

        Args:
            output: Surowy wynik predykcji

        Returns:
            Przetworzony wynik jako słownik lub lista słowników
        """
        pass

    def __call__(self, data: InputT) -> dict | list[dict]:
        """
        Wykonuje pełny pipeline: preprocess -> predict -> postprocess.

        Args:
            data: Dane wejściowe

        Returns:
            Przetworzony wynik predykcji
        """
        if not self.is_loaded:
            self.load()

        processed = self.preprocess(data)
        output = self.predict(processed)
        return self.postprocess(output)

    def __repr__(self) -> str:
        """Reprezentacja tekstowa modelu."""
        return (
            f"{self.__class__.__name__}("
            f"weights={self.config.weights_path}, "
            f"device={self.config.device}, "
            f"loaded={self._loaded})"
        )
