"""
Model detekcji psów (Bounding Box) oparty na YOLOv8.

Wykrywa psy na obrazach i zwraca bounding boxy z confidence scores.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseModel, ModelConfig


@dataclass
class BBoxConfig(ModelConfig):
    """
    Konfiguracja modelu detekcji.

    Attributes:
        weights_path: Ścieżka do wag modelu (.pt)
        device: Urządzenie ('cuda', 'cpu', '0', etc.)
        confidence_threshold: Minimalny próg pewności detekcji
        iou_threshold: Próg IoU dla Non-Maximum Suppression
        max_detections: Maksymalna liczba detekcji na obraz
        imgsz: Rozmiar obrazu wejściowego (domyślnie 640)
    """

    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 10
    imgsz: int = 640


@dataclass
class Detection:
    """
    Pojedyncza detekcja psa.

    Attributes:
        bbox: Bounding box jako (x, y, width, height) w pikselach
        confidence: Pewność detekcji [0.0, 1.0]
        class_id: ID klasy (zawsze 0 dla psa)
        class_name: Nazwa klasy (zawsze "dog")
    """

    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    class_id: int = 0
    class_name: str = "dog"

    def to_dict(self) -> dict:
        """Konwertuje detekcję do słownika."""
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    def to_coco(self, image_id: int, annotation_id: int) -> dict:
        """
        Konwertuje detekcję do formatu COCO.

        Args:
            image_id: ID obrazu w datasecie
            annotation_id: Unikalny ID anotacji

        Returns:
            Słownik w formacie COCO annotation
        """
        x, y, w, h = self.bbox
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # dog
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
            "score": self.confidence,
        }


class BBoxModel(BaseModel[np.ndarray, list[Detection]]):
    """
    Model detekcji psów oparty na YOLOv8.

    Używa pretrenowanego lub fine-tunowanego modelu YOLOv8
    do wykrywania psów na obrazach.

    Example:
        >>> config = BBoxConfig(weights_path="models/bbox.pt")
        >>> model = BBoxModel(config)
        >>> model.load()
        >>> image = cv2.imread("dog.jpg")
        >>> detections = model.predict(image)
        >>> for det in detections:
        ...     print(f"Dog at {det.bbox} with confidence {det.confidence:.2f}")
    """

    def __init__(self, config: BBoxConfig) -> None:
        """
        Inicjalizuje model detekcji.

        Args:
            config: Konfiguracja modelu
        """
        super().__init__(config)
        self.config: BBoxConfig = config
        self._model: Optional[object] = None  # YOLO model

    def load(self) -> None:
        """
        Ładuje model YOLOv8.

        Raises:
            FileNotFoundError: Gdy plik z wagami nie istnieje
            ImportError: Gdy ultralytics nie jest zainstalowane
        """
        if not self.config.weights_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono wag modelu: {self.config.weights_path}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Biblioteka ultralytics nie jest zainstalowana. "
                "Zainstaluj: pip install ultralytics"
            ) from e

        self._model = YOLO(str(self.config.weights_path))

        # Ustaw device
        if self.config.device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    # Model sam wybierze device podczas inference
                    pass
            except ImportError:
                pass

        self._loaded = True

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Przetwarza obraz przed inference.

        YOLOv8 obsługuje preprocessing wewnętrznie, więc ta metoda
        wykonuje tylko walidację.

        Args:
            image: Obraz jako numpy array (BGR lub RGB)

        Returns:
            Obraz (bez zmian - YOLO obsługuje preprocessing)

        Raises:
            ValueError: Gdy obraz jest nieprawidłowy
        """
        if image is None or image.size == 0:
            raise ValueError("Obraz nie może być pusty")

        if len(image.shape) != 3:
            raise ValueError(
                f"Obraz musi mieć 3 wymiary (H, W, C), otrzymano: {image.shape}"
            )

        if image.shape[2] not in [3, 4]:
            raise ValueError(
                f"Obraz musi mieć 3 lub 4 kanały, otrzymano: {image.shape[2]}"
            )

        return image

    def predict(self, image: np.ndarray) -> list[Detection]:
        """
        Wykrywa psy na obrazie.

        Args:
            image: Obraz jako numpy array (BGR)

        Returns:
            Lista wykrytych psów jako obiekty Detection

        Raises:
            RuntimeError: Gdy model nie został załadowany
        """
        if not self._loaded or self._model is None:
            raise RuntimeError(
                "Model nie został załadowany. Wywołaj load() przed predict()."
            )

        # Wywołaj YOLO
        results = self._model(
            image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            device=self.config.device,
            verbose=False,
            imgsz=self.config.imgsz,
        )[0]

        # Konwertuj wyniki do Detection
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                # Pobierz współrzędne (xyxy -> xywh)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1

                # Confidence
                conf = float(box.conf[0].cpu().numpy())

                # Class ID (filtrujemy tylko psy jeśli model ma wiele klas)
                cls_id = int(box.cls[0].cpu().numpy())

                # Dla modelu fine-tunowanego na psach, class_id = 0
                # Dla pretrenowanego YOLO COCO, psy to class_id = 16
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(w), int(h)),
                        confidence=conf,
                        class_id=cls_id,
                        class_name="dog",
                    )
                )

        return detections

    def postprocess(self, detections: list[Detection]) -> list[dict]:
        """
        Konwertuje detekcje do listy słowników.

        Args:
            detections: Lista obiektów Detection

        Returns:
            Lista słowników z informacjami o detekcjach
        """
        return [det.to_dict() for det in detections]

    def predict_with_visualization(
        self,
        image: np.ndarray,
    ) -> tuple[list[Detection], np.ndarray]:
        """
        Wykrywa psy i zwraca obraz z narysowanymi bounding boxami.

        Args:
            image: Obraz wejściowy

        Returns:
            Tuple (lista detekcji, obraz z wizualizacją)
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model nie został załadowany.")

        results = self._model(
            image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            device=self.config.device,
            verbose=False,
        )[0]

        # Detekcje
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        confidence=float(box.conf[0]),
                    )
                )

        # Wizualizacja
        annotated = results.plot()

        return detections, annotated

    def filter_dogs_only(
        self,
        image: np.ndarray,
        coco_dog_class_id: int = 16,
    ) -> list[Detection]:
        """
        Wykrywa tylko psy używając pretrenowanego modelu COCO.

        Przydatne gdy używamy standardowego yolov8m.pt zamiast
        fine-tunowanego modelu.

        Args:
            image: Obraz wejściowy
            coco_dog_class_id: ID klasy psa w COCO (domyślnie 16)

        Returns:
            Lista detekcji tylko psów
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model nie został załadowany.")

        results = self._model(
            image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections * 5,  # Więcej, bo filtrujemy
            device=self.config.device,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].cpu().numpy())

                # Filtruj tylko psy
                if cls_id == coco_dog_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append(
                        Detection(
                            bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                            confidence=float(box.conf[0]),
                            class_id=0,  # Normalizuj do 0
                            class_name="dog",
                        )
                    )

        # Ogranicz do max_detections
        detections = sorted(
            detections, key=lambda d: d.confidence, reverse=True
        )[: self.config.max_detections]

        return detections
