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
    confidence_threshold: float = 0.15  # Niższy próg dla lepszej wizualizacji
    use_tta: bool = True  # Test-Time Augmentation (flip + average)


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

    # Mapping dla flip (zamiana lewych i prawych keypoints po odbiciu)
    # Format: project_idx -> flipped_project_idx
    FLIP_MAPPING: dict[int, int] = {
        0: 1, 1: 0,    # left_eye <-> right_eye
        3: 4, 4: 3,    # left_ear_base <-> right_ear_base
        5: 6, 6: 5,    # left_ear_tip <-> right_ear_tip
        7: 8, 8: 7,    # left_mouth_corner <-> right_mouth_corner
        12: 13, 13: 12,  # left_cheek <-> right_cheek
        15: 16, 16: 15,  # left_eyebrow <-> right_eyebrow
        18: 19, 19: 18,  # muzzle_left <-> muzzle_right
        # Środkowe punkty pozostają bez zmian:
        2: 2, 9: 9, 10: 10, 11: 11, 14: 14, 17: 17,
    }

    def predict(self, image: np.ndarray) -> KeypointsPrediction:
        """
        Wykrywa keypoints na obrazie.

        Zwraca 20 keypoints zgodnie ze specyfikacją projektu.
        Jeśli use_tta=True, używa Test-Time Augmentation (flip + average).
        """
        if not self._loaded:
            raise RuntimeError("Model nie załadowany. Wywołaj load() najpierw.")

        original_h, original_w = image.shape[:2]

        # Podstawowa predykcja
        project_keypoints = self._predict_single(image, original_w, original_h)

        # Test-Time Augmentation: flip + average
        if self.config.use_tta:
            # Odwróć obraz poziomo
            flipped_image = np.fliplr(image).copy()
            flipped_keypoints = self._predict_single(
                flipped_image, original_w, original_h
            )

            # Odwróć współrzędne i zamień lewe/prawe keypoints
            project_keypoints = self._merge_with_flipped(
                project_keypoints, flipped_keypoints, original_w
            )

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

    def _predict_single(
        self,
        image: np.ndarray,
        original_w: int,
        original_h: int,
    ) -> list[Keypoint]:
        """Pojedyncza predykcja bez TTA."""
        tensor = self.preprocess(image)

        with torch.no_grad():
            heatmaps = self.model(tensor)

        dogflw_keypoints = self._decode_heatmaps(
            heatmaps[0],
            original_w,
            original_h,
            num_keypoints=NUM_KEYPOINTS_DOGFLW,
        )

        return self._map_to_project_keypoints(dogflw_keypoints)

    def _merge_with_flipped(
        self,
        original: list[Keypoint],
        flipped: list[Keypoint],
        image_width: int,
    ) -> list[Keypoint]:
        """
        Łączy keypoints z oryginalnego i odwróconego obrazu.

        Używa średniej ważonej confidence dla lepszych wyników.
        """
        merged = []

        for idx in range(NUM_KEYPOINTS):
            orig_kp = original[idx]
            flip_idx = self.FLIP_MAPPING.get(idx, idx)
            flip_kp = flipped[flip_idx]

            # Odwróć współrzędną X dla flipped keypoint
            flip_x = image_width - flip_kp.x

            # Średnia ważona na podstawie confidence
            total_conf = orig_kp.visibility + flip_kp.visibility
            if total_conf > 0:
                w1 = orig_kp.visibility / total_conf
                w2 = flip_kp.visibility / total_conf
                avg_x = w1 * orig_kp.x + w2 * flip_x
                avg_y = w1 * orig_kp.y + w2 * flip_kp.y
                avg_vis = max(orig_kp.visibility, flip_kp.visibility)
            else:
                avg_x = (orig_kp.x + flip_x) / 2
                avg_y = (orig_kp.y + flip_kp.y) / 2
                avg_vis = 0.0

            merged.append(Keypoint(x=avg_x, y=avg_y, visibility=avg_vis))

        return merged

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
        radius: int = 4,
        show_names: bool = False,
        skeleton_width: int = 2,
        show_low_confidence: bool = True,
    ) -> np.ndarray:
        """
        Rysuje keypoints i skeleton na obrazie.

        Args:
            image: Obraz do wizualizacji
            prediction: Wynik predykcji keypoints
            draw_skeleton: Czy rysować połączenia skeleton
            radius: Promień punktów keypoints
            show_names: Czy pokazywać nazwy keypoints
            skeleton_width: Grubość linii skeleton
            show_low_confidence: Czy pokazywać punkty z niskim confidence
        """
        from PIL import Image, ImageDraw

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        kps = prediction.keypoints
        threshold = self.config.confidence_threshold
        low_threshold = threshold * 0.5  # Próg dla pół-przezroczystych punktów

        # Kolory dla różnych grup skeleton
        skeleton_colors = {
            "eyes": (0, 255, 0),       # Zielony - oczy
            "brows": (128, 0, 255),    # Fioletowy - brwi/czoło
            "ears": (255, 165, 0),     # Pomarańczowy - uszy
            "nose": (0, 128, 255),     # Niebieski - nos/pysk
            "cheeks": (255, 0, 0),     # Czerwony - policzki
            "mouth": (255, 255, 0),    # Żółty - usta
            "chin": (255, 0, 255),     # Magenta - podbródek
        }

        def get_skeleton_color(i: int, j: int) -> tuple[int, int, int]:
            """Zwraca kolor dla połączenia skeleton."""
            pts = {i, j}
            if pts & {0, 1}:  # Oczy
                return skeleton_colors["eyes"]
            if pts & {15, 16, 14}:  # Brwi/czoło
                return skeleton_colors["brows"]
            if pts & {3, 4, 5, 6}:  # Uszy
                return skeleton_colors["ears"]
            if pts & {2, 17, 18, 19}:  # Nos/pysk
                return skeleton_colors["nose"]
            if pts & {12, 13}:  # Policzki
                return skeleton_colors["cheeks"]
            if pts & {7, 8, 9, 10}:  # Usta
                return skeleton_colors["mouth"]
            if pts & {11}:  # Podbródek
                return skeleton_colors["chin"]
            return (150, 150, 150)  # Domyślny szary

        # Rysuj skeleton
        if draw_skeleton:
            for i, j in SKELETON_CONNECTIONS:
                if i < len(kps) and j < len(kps):
                    kp_i, kp_j = kps[i], kps[j]
                    min_vis = min(kp_i.visibility, kp_j.visibility)

                    # Rysuj jeśli przynajmniej jedna ma wystarczający confidence
                    if min_vis > low_threshold or (
                        show_low_confidence and min_vis > 0.05
                    ):
                        color = get_skeleton_color(i, j)
                        # Przezroczystość dla niskiego confidence
                        if min_vis < threshold:
                            color = tuple(c // 2 for c in color)
                        draw.line(
                            [(kp_i.x, kp_i.y), (kp_j.x, kp_j.y)],
                            fill=color,
                            width=skeleton_width,
                        )

        # Rysuj keypoints
        for k, kp in enumerate(kps):
            draw_point = kp.visibility > threshold or (
                show_low_confidence and kp.visibility > low_threshold
            )

            if draw_point:
                color = get_keypoint_color(k)

                # Zmniejsz jasność dla niskiego confidence
                if kp.visibility < threshold:
                    color = tuple(c // 2 for c in color)
                    r = radius - 1
                else:
                    r = radius

                draw.ellipse(
                    [
                        (kp.x - r, kp.y - r),
                        (kp.x + r, kp.y + r),
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
