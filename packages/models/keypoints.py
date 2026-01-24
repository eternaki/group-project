"""
Model detekcji keypoints na twarzy psa.

Wykrywa 46 kluczowych punktów na twarzy psa używając architektury
SimpleBaseline (ResNet50 + Deconv Head), a następnie mapuje je
do 20 keypoints zgodnie ze specyfikacją projektu.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from packages.data.schemas import (
    NUM_KEYPOINTS,
    NUM_KEYPOINTS_DOGFLW,
    KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
    PROJECT_TO_DOGFLW_MAPPING,
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
    """Wynik predykcji keypoints dla jednego obrazu (20 keypoints)."""

    keypoints: list[Keypoint]
    confidence: float
    num_detected: int

    def to_annotation(self, image_id: str) -> KeypointsAnnotation:
        """Konwertuje do KeypointsAnnotation."""
        return KeypointsAnnotation(image_id=image_id, keypoints=self.keypoints)

    def to_coco_format(self) -> list[float]:
        """Konwertuje do formatu COCO: [x1, y1, v1, x2, y2, v2, ...]."""
        result = []
        for kp in self.keypoints:
            result.extend([kp.x, kp.y, kp.visibility])
        return result


class SimpleBaselineModel(nn.Module):
    """
    Simple Baseline dla pose estimation.
    Architektura zgodna z wytrenowanym modelem na Kaggle (DogFLW - 46 keypoints).
    """

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS_DOGFLW):
        super().__init__()
        import timm

        # Backbone - używamy nazwy 'bb' jak w treningu
        self.bb = timm.create_model(
            "resnet50",
            pretrained=False,
            features_only=True,
            out_indices=[-1],
        )

        # Deconv head - nazwy zgodne z treningiem
        self.head = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # Final conv - 46 keypoints z DogFLW
        self.out = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - zwraca heatmapy."""
        x = self.bb(x)[-1]
        x = self.head(x)
        return self.out(x)


class KeypointsModel(BaseModel[np.ndarray, KeypointsPrediction]):
    """
    Model do detekcji keypoints na twarzy psa.

    Wewnętrznie używa modelu DogFLW (46 keypoints),
    ale zwraca 20 keypoints zgodnie ze specyfikacją projektu.

    Użycie:
        config = KeypointsConfig(weights_path="models/keypoints_best.pt")
        model = KeypointsModel(config)
        model.load()

        prediction = model.predict(image)
        print(f"Wykryto {prediction.num_detected} keypoints")
        print(f"Nazwy: {[KEYPOINT_NAMES[i] for i in range(20)]}")
    """

    def __init__(self, config: KeypointsConfig) -> None:
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
        # Model DogFLW ma 46 keypoints
        self.model = SimpleBaselineModel(num_keypoints=NUM_KEYPOINTS_DOGFLW)

        weights_path = Path(self.config.weights_path)
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Wagi załadowane: {weights_path}")
        else:
            print(f"Wagi nie znalezione: {weights_path}")

        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        """Przetwarza obraz do formatu modelu."""
        if data.shape[2] == 3:
            image = data
        else:
            image = data[:, :, :3]

        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: np.ndarray) -> KeypointsPrediction:
        """
        Wykrywa keypoints na obrazie.

        Zwraca 20 keypoints zgodnie ze specyfikacją projektu
        (zmapowane z 46 keypoints DogFLW).
        """
        if not self._loaded:
            raise RuntimeError("Model nie załadowany. Wywołaj load() najpierw.")

        original_h, original_w = image.shape[:2]

        tensor = self.preprocess(image)

        with torch.no_grad():
            heatmaps = self.model(tensor)

        # Dekoduj wszystkie 46 keypoints z DogFLW
        dogflw_keypoints = self._decode_heatmaps(
            heatmaps[0],
            original_w,
            original_h,
            num_keypoints=NUM_KEYPOINTS_DOGFLW,
        )

        # Mapuj 46 -> 20 keypoints projektu
        project_keypoints = self._map_to_project_keypoints(dogflw_keypoints)

        visible_count = sum(
            1 for kp in project_keypoints
            if kp.visibility > self.config.confidence_threshold
        )
        avg_confidence = np.mean([kp.visibility for kp in project_keypoints])

        return KeypointsPrediction(
            keypoints=project_keypoints,
            confidence=float(avg_confidence),
            num_detected=visible_count,
        )

    def _map_to_project_keypoints(
        self,
        dogflw_keypoints: list[Keypoint],
    ) -> list[Keypoint]:
        """
        Mapuje 46 keypoints DogFLW do 20 keypoints projektu.

        Args:
            dogflw_keypoints: Lista 46 keypoints z modelu DogFLW

        Returns:
            Lista 20 keypoints zgodnie ze specyfikacją projektu
        """
        project_keypoints = []
        for project_idx in range(NUM_KEYPOINTS):
            dogflw_idx = PROJECT_TO_DOGFLW_MAPPING[project_idx]
            project_keypoints.append(dogflw_keypoints[dogflw_idx])
        return project_keypoints

    def postprocess(self, output: KeypointsPrediction) -> dict:
        """Przetwarza wynik predykcji do formatu słownika."""
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
            "confidence": output.confidence,
            "num_detected": output.num_detected,
        }

    def _decode_heatmaps(
        self,
        heatmaps: torch.Tensor,
        target_width: int,
        target_height: int,
        num_keypoints: int = NUM_KEYPOINTS_DOGFLW,
    ) -> list[Keypoint]:
        """Dekoduje heatmapy do keypoints."""
        hm_height, hm_width = heatmaps.shape[1], heatmaps.shape[2]
        scale_x = target_width / hm_width
        scale_y = target_height / hm_height

        keypoints = []

        for k in range(num_keypoints):
            hm = heatmaps[k].cpu().numpy()

            max_val = hm.max()
            max_idx = hm.argmax()
            y_hm = max_idx // hm_width
            x_hm = max_idx % hm_width

            x = float(x_hm * scale_x)
            y = float(y_hm * scale_y)

            visibility = float(max_val)

            keypoints.append(Keypoint(x=x, y=y, visibility=visibility))

        return keypoints

    def draw_keypoints(
        self,
        image: np.ndarray,
        prediction: KeypointsPrediction,
        draw_skeleton: bool = True,
        radius: int = 3,
        show_names: bool = False,
    ) -> np.ndarray:
        """Rysuje keypoints na obrazie."""
        from PIL import Image, ImageDraw

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        kps = prediction.keypoints
        threshold = self.config.confidence_threshold

        # Rysuj skeleton
        if draw_skeleton:
            for i, j in SKELETON_CONNECTIONS:
                if i < len(kps) and j < len(kps):
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

                # Opcjonalnie: rysuj nazwy
                if show_names and k < len(KEYPOINT_NAMES):
                    draw.text(
                        (kp.x + radius + 2, kp.y - 5),
                        KEYPOINT_NAMES[k],
                        fill=color,
                    )

        return np.array(pil_image)
