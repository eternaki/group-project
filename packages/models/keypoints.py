"""
Model detekcji keypoints na twarzy psa.

Wykrywa 20 kluczowych punktów na twarzy psa używając architektury
SimpleBaseline (ResNet + Deconv Head) lub HRNet.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from packages.data.schemas import (
    NUM_KEYPOINTS,
    KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
    Keypoint,
    KeypointsAnnotation,
    get_keypoint_color,
)
from packages.models.base import BaseModel, ModelConfig


@dataclass
class KeypointsConfig(ModelConfig):
    """Konfiguracja modelu keypoints."""

    model_name: str = "resnet50"
    img_size: int = 256
    heatmap_size: int = 64
    confidence_threshold: float = 0.3


@dataclass
class KeypointsPrediction:
    """Wynik predykcji keypoints dla jednego obrazu."""

    keypoints: list[Keypoint]  # 20 keypoints
    confidence: float  # Średnia pewność
    num_detected: int  # Liczba wykrytych punktów (visibility > threshold)

    def to_annotation(self, image_id: str) -> KeypointsAnnotation:
        """Konwertuje do KeypointsAnnotation."""
        return KeypointsAnnotation(image_id=image_id, keypoints=self.keypoints)

    def get_keypoint(self, name: str) -> Keypoint | None:
        """
        Pobiera keypoint po nazwie.

        Args:
            name: Nazwa keypointa (np. 'left_eye', 'nose')

        Returns:
            Keypoint lub None jeśli niewidoczny
        """
        try:
            idx = KEYPOINT_NAMES.index(name)
            kp = self.keypoints[idx]
            if kp.visibility > 0.5:
                return kp
            return None
        except ValueError:
            return None


class SimpleBaselineModel(nn.Module):
    """
    Simple Baseline dla pose estimation.

    Architektura: ResNet/HRNet backbone + Deconvolution head
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_keypoints: int = NUM_KEYPOINTS,
        pretrained: bool = True,
    ):
        super().__init__()

        import timm

        # Backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],
        )

        # Określ kanały
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            features = self.backbone(dummy)
            backbone_channels = features[-1].shape[1]

        # Deconv head
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(backbone_channels, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final conv
        self.final = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - zwraca heatmapy."""
        features = self.backbone(x)[-1]
        x = self.deconv(features)
        return self.final(x)


class KeypointsModel(BaseModel[np.ndarray, KeypointsPrediction]):
    """
    Model do detekcji keypoints na twarzy psa.

    Użycie:
        config = KeypointsConfig(weights_path="models/keypoints.pt")
        model = KeypointsModel(config)
        model.load()

        prediction = model.predict(image)
        print(f"Wykryto {prediction.num_detected} keypoints")
    """

    def __init__(self, config: KeypointsConfig) -> None:
        """
        Inicjalizuje model.

        Args:
            config: Konfiguracja modelu
        """
        super().__init__(config)
        self.config: KeypointsConfig = config
        self.model: SimpleBaselineModel | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self) -> None:
        """Ładuje model z wag."""
        self.model = SimpleBaselineModel(
            backbone=self.config.model_name,
            num_keypoints=NUM_KEYPOINTS,
            pretrained=False,
        )

        weights_path = Path(self.config.weights_path)
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"⚠️ Wagi nie znalezione: {weights_path}")
            print("   Model używa losowych wag (tylko do testów)")

        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        """
        Przetwarza obraz do formatu modelu.

        Args:
            data: Obraz numpy [H, W, 3] BGR lub RGB

        Returns:
            Tensor [1, 3, img_size, img_size]
        """
        # Zakładamy RGB, jeśli BGR to konwertuj
        if data.shape[2] == 3:
            image = data
        else:
            image = data[:, :, :3]

        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: np.ndarray) -> KeypointsPrediction:
        """
        Wykrywa keypoints na obrazie.

        Args:
            image: Obraz numpy [H, W, 3]

        Returns:
            KeypointsPrediction z 20 keypoints
        """
        if not self._loaded:
            raise RuntimeError("Model nie załadowany. Wywołaj load() najpierw.")

        original_h, original_w = image.shape[:2]

        # Preprocess
        tensor = self.preprocess(image)

        # Inference
        with torch.no_grad():
            heatmaps = self.model(tensor)

        # Decode heatmaps to keypoints
        keypoints = self._decode_heatmaps(
            heatmaps[0],
            original_w,
            original_h,
        )

        # Oblicz statystyki
        visible_count = sum(1 for kp in keypoints if kp.visibility > self.config.confidence_threshold)
        avg_confidence = np.mean([kp.visibility for kp in keypoints])

        return KeypointsPrediction(
            keypoints=keypoints,
            confidence=float(avg_confidence),
            num_detected=visible_count,
        )

    def _decode_heatmaps(
        self,
        heatmaps: torch.Tensor,
        target_width: int,
        target_height: int,
    ) -> list[Keypoint]:
        """
        Dekoduje heatmapy do keypoints.

        Args:
            heatmaps: Tensor [NUM_KEYPOINTS, H, W]
            target_width: Docelowa szerokość
            target_height: Docelowa wysokość

        Returns:
            Lista 20 Keypoint
        """
        hm_height, hm_width = heatmaps.shape[1], heatmaps.shape[2]
        scale_x = target_width / hm_width
        scale_y = target_height / hm_height

        keypoints = []

        for k in range(NUM_KEYPOINTS):
            hm = heatmaps[k].cpu().numpy()

            # Znajdź maksimum
            max_val = hm.max()
            max_idx = hm.argmax()
            y_hm = max_idx // hm_width
            x_hm = max_idx % hm_width

            # Skaluj do oryginalnego rozmiaru
            x = float(x_hm * scale_x)
            y = float(y_hm * scale_y)

            # Visibility = wartość heatmapy (0-1)
            visibility = float(max_val)

            keypoints.append(Keypoint(x=x, y=y, visibility=visibility))

        return keypoints

    def postprocess(self, output: KeypointsPrediction) -> dict:
        """
        Konwertuje wynik do słownika.

        Args:
            output: KeypointsPrediction

        Returns:
            Słownik z keypoints
        """
        return {
            "keypoints": [
                {
                    "name": KEYPOINT_NAMES[i],
                    "x": kp.x,
                    "y": kp.y,
                    "visibility": kp.visibility,
                }
                for i, kp in enumerate(output.keypoints)
            ],
            "num_detected": output.num_detected,
            "confidence": output.confidence,
        }

    def draw_keypoints(
        self,
        image: np.ndarray,
        prediction: KeypointsPrediction,
        draw_skeleton: bool = True,
        radius: int = 3,
    ) -> np.ndarray:
        """
        Rysuje keypoints na obrazie.

        Args:
            image: Obraz numpy [H, W, 3]
            prediction: Wynik predykcji
            draw_skeleton: Czy rysować połączenia
            radius: Promień punktów

        Returns:
            Obraz z narysowanymi keypoints
        """
        from PIL import Image, ImageDraw

        # Konwertuj do PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        kps = prediction.keypoints
        threshold = self.config.confidence_threshold

        # Rysuj skeleton
        if draw_skeleton:
            for i, j in SKELETON_CONNECTIONS:
                if kps[i].visibility > threshold and kps[j].visibility > threshold:
                    draw.line(
                        [(kps[i].x, kps[i].y), (kps[j].x, kps[j].y)],
                        fill=(100, 100, 100),
                        width=1,
                    )

        # Rysuj keypoints
        for k, kp in enumerate(kps):
            if kp.visibility > threshold:
                color = get_keypoint_color(k)
                draw.ellipse(
                    [
                        (kp.x - radius, kp.y - radius),
                        (kp.x + radius, kp.y + radius),
                    ],
                    fill=color,
                    outline=(255, 255, 255),
                )

        return np.array(pil_image)
