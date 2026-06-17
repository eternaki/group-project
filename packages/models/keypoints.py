"""
Model detekcji keypoints na twarzy psa.

Wykrywa 46 kluczowych punktów na twarzy psa zgodnie ze schematem DogFLW,
używając heatmap regression na backbone HRNet-W32 (konfigurowalny).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from packages.data.schemas import (
    KEYPOINT_NAMES,
    KP,
    NUM_KEYPOINTS,
    SKELETON_CONNECTIONS,
    Keypoint,
    KeypointsAnnotation,
    get_keypoint_color,
)
from packages.models.base import BaseModel, ModelConfig

# Pary lewo/prawo w KANONICZNEJ kolejności DogFLW (do odbicia poziomego).
# Używane zarówno w treningu (poprawny flip z przestawieniem par), jak i w
# inferencji (flip-TTA). Punkty centralne (24,25,32,35,38,41,42,45) mapują się
# na siebie (domyślnie przez .get(i, i)).
FLIP_PAIRS: list[tuple[int, int]] = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13),  # uszy
    (14, 15),                                                     # brwi
    (16, 17), (18, 19), (20, 21), (22, 23),                       # oczy
    (26, 27), (28, 29), (30, 31), (33, 34), (36, 37),             # nos/pysk/policzki
    (39, 40), (43, 44),                                           # wargi
]
FLIP_MAPPING: dict[int, int] = {
    **{a: b for a, b in FLIP_PAIRS},
    **{b: a for a, b in FLIP_PAIRS},
}


@dataclass
class KeypointsConfig(ModelConfig):
    """Konfiguracja modelu keypoints."""

    model_name: str = "hrnet_w32"  # HRNet-W32: stride-4 → heatmapy bez deconv
    img_size: int = 320  # wejście 320 → heatmapa 80 (lepsza lokalizacja niż 256/64)
    heatmap_size: int = 80
    confidence_threshold: float = 0.15  # Niższy próg dla lepszej wizualizacji
    use_tta: bool = True  # Test-Time Augmentation (flip + average)
    use_dark: bool = True  # Subpikselowe dekodowanie heatmap (DARK)


@dataclass
class KeypointsPrediction:
    """Wynik predykcji keypoints dla jednego obrazu (46 keypoints DogFLW)."""

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
    Model detekcji keypoints (heatmap regression) z konfigurowalnym backbone.

    Domyślnie HRNet-W32: warstwa stride-4 daje mapy 64×64 bez deconv, co przy
    wejściu 256² odpowiada rozmiarowi heatmap (najlepsza lokalizacja dla
    landmarków twarzy). Dla backbone'ów o niskiej rozdzielczości (np. ResNet,
    stride-32) automatycznie dokładany jest deconv-head do rozmiaru heatmap.

    UWAGA: ta sama klasa musi być użyta w treningu (notebook Kaggle) i w
    inferencji — inaczej wagi się nie wczytają.
    """

    def __init__(
        self,
        num_keypoints: int = NUM_KEYPOINTS,
        backbone: str = "hrnet_w32",
        img_size: int = 256,
        heatmap_size: int = 64,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        import timm

        # pretrained=True tylko przy treningu (init z ImageNet); w inferencji
        # False, bo i tak wczytujemy własne wagi z pliku.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )
        reductions = self.backbone.feature_info.reduction()
        channels = self.backbone.feature_info.channels()

        # Wybierz warstwę o reduction == img_size/heatmap_size (np. stride-4
        # dla 256→64). Jeśli brak — bierz najgłębszą i dołóż deconv.
        target_reduction = img_size // heatmap_size
        if target_reduction in reductions:
            self.feat_idx = reductions.index(target_reduction)
            in_channels = channels[self.feat_idx]
            self.deconv: nn.Module = nn.Identity()
            head_in = in_channels
        else:
            self.feat_idx = len(reductions) - 1
            in_channels = channels[self.feat_idx]
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            head_in = 256

        # Head: conv 3×3 (uściślenie) → conv 1×1 → 46 heatmap
        self.head = nn.Sequential(
            nn.Conv2d(head_in, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, num_keypoints, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - zwraca heatmapy (B, num_keypoints, H, W)."""
        features = self.backbone(x)[self.feat_idx]
        x = self.deconv(features)
        return self.head(x)


class KeypointsModel(BaseModel[np.ndarray, KeypointsPrediction]):
    """
    Model do detekcji 46 keypoints na twarzy psa (schemat DogFLW).

    Użycie:
        config = KeypointsConfig(weights_path="models/keypoints_dogflw.pt")
        model = KeypointsModel(config)
        model.load()

        prediction = model.predict(image)
        print(f"Wykryto {prediction.num_detected} keypoints")
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
        # Model DogFLW: 46 keypoints, backbone z konfiguracji (domyślnie HRNet-W32)
        self.model = SimpleBaselineModel(
            num_keypoints=NUM_KEYPOINTS,
            backbone=self.config.model_name,
            img_size=self.config.img_size,
            heatmap_size=self.config.heatmap_size,
        )

        weights_path = Path(self.config.weights_path)
        if weights_path.exists():
            state_dict = torch.load(
                weights_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
            print(f"Wagi załadowane: {weights_path} (backbone: {self.config.model_name})")
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

    # Mapping dla flip TTA — wskazuje na stałe modułowe FLIP_PAIRS/FLIP_MAPPING.
    FLIP_MAPPING: dict[int, int] = FLIP_MAPPING

    def predict(self, image: np.ndarray) -> KeypointsPrediction:
        """
        Wykrywa keypoints na obrazie.

        Zwraca 46 keypoints zgodnie ze schematem DogFLW.
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

        return self._decode_heatmaps(
            heatmaps[0],
            original_w,
            original_h,
        )

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
        num_keypoints: int = NUM_KEYPOINTS,
    ) -> list[Keypoint]:
        """
        Dekoduje heatmapy do keypoints.

        Gdy ``config.use_dark`` jest włączone, stosuje subpikselowe uściślenie
        DARK (modulacja Gaussem + rozwinięcie Taylora log-likelihood), co
        eliminuje kwantyzację argmax na siatce heatmap (krok 4 px przy 64×64).
        """
        hm_all = heatmaps.cpu().numpy()
        hm_height, hm_width = hm_all.shape[1], hm_all.shape[2]
        scale_x = target_width / hm_width
        scale_y = target_height / hm_height

        keypoints = []
        for k in range(num_keypoints):
            hm = hm_all[k]
            visibility = float(hm.max())

            if self.config.use_dark:
                x_hm, y_hm = self._dark_refine(hm)
            else:
                max_idx = int(hm.argmax())
                y_hm = max_idx // hm_width
                x_hm = max_idx % hm_width

            keypoints.append(
                Keypoint(
                    x=float(x_hm * scale_x),
                    y=float(y_hm * scale_y),
                    visibility=visibility,
                )
            )

        return keypoints

    @staticmethod
    def _dark_refine(hm: np.ndarray, blur_sigma: float = 1.0) -> tuple[float, float]:
        """
        Subpikselowe uściślenie pozycji piku heatmapy metodą DARK.

        Args:
            hm: Pojedyncza heatmapa (H, W)
            blur_sigma: Sigma rozmycia Gaussa stabilizującego pochodne

        Returns:
            (x, y) — współrzędne piku w układzie heatmapy (subpikselowo)
        """
        import cv2

        h, w = hm.shape
        hm_b = cv2.GaussianBlur(hm, (0, 0), blur_sigma)
        hm_b = np.clip(hm_b, 1e-10, None)
        lp = np.log(hm_b)

        idx = int(hm_b.argmax())
        py, px = idx // w, idx % w
        x, y = float(px), float(py)

        if 1 <= px < w - 1 and 1 <= py < h - 1:
            dx = 0.5 * (lp[py, px + 1] - lp[py, px - 1])
            dy = 0.5 * (lp[py + 1, px] - lp[py - 1, px])
            dxx = lp[py, px + 1] - 2 * lp[py, px] + lp[py, px - 1]
            dyy = lp[py + 1, px] - 2 * lp[py, px] + lp[py - 1, px]
            dxy = 0.25 * (
                lp[py + 1, px + 1] - lp[py + 1, px - 1]
                - lp[py - 1, px + 1] + lp[py - 1, px - 1]
            )
            det = dxx * dyy - dxy * dxy
            if abs(det) > 1e-6:
                ox = -(dyy * dx - dxy * dy) / det
                oy = -(-dxy * dx + dxx * dy) / det
                if abs(ox) < 1.0 and abs(oy) < 1.0:
                    x, y = px + ox, py + oy

        return x, y

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

        _eye_kps = {
            KP.LEFT_EYE_INNER, KP.LEFT_EYE_TOP, KP.LEFT_EYE_OUTER, KP.LEFT_EYE_BOTTOM,
            KP.RIGHT_EYE_INNER, KP.RIGHT_EYE_TOP, KP.RIGHT_EYE_OUTER, KP.RIGHT_EYE_BOTTOM,
        }
        _brow_kps = {
            KP.LEFT_BROW_INNER, KP.LEFT_BROW_CENTER, KP.LEFT_BROW_OUTER,
            KP.RIGHT_BROW_INNER, KP.RIGHT_BROW_CENTER, KP.RIGHT_BROW_OUTER,
            KP.FOREHEAD_CENTER, KP.FOREHEAD_LEFT, KP.FOREHEAD_RIGHT,
        }
        _ear_kps = {
            KP.LEFT_EAR_BASE_FRONT, KP.LEFT_EAR_BASE_BACK, KP.LEFT_EAR_MID, KP.LEFT_EAR_TIP,
            KP.RIGHT_EAR_BASE_FRONT, KP.RIGHT_EAR_BASE_BACK, KP.RIGHT_EAR_MID, KP.RIGHT_EAR_TIP,
        }
        _nose_kps = {KP.NOSE_TIP, KP.NOSE_LEFT_WING, KP.NOSE_RIGHT_WING, KP.NOSE_BRIDGE}
        _cheek_kps = {
            KP.LEFT_CHEEK_UPPER, KP.LEFT_CHEEK_LOWER,
            KP.RIGHT_CHEEK_UPPER, KP.RIGHT_CHEEK_LOWER,
        }
        _mouth_kps = {
            KP.MOUTH_LEFT_CORNER, KP.UPPER_LIP_LEFT, KP.UPPER_LIP_CENTER, KP.UPPER_LIP_RIGHT,
            KP.MOUTH_RIGHT_CORNER, KP.LOWER_LIP_RIGHT, KP.LOWER_LIP_CENTER, KP.LOWER_LIP_LEFT,
            KP.MUZZLE_TOP, KP.MUZZLE_LEFT, KP.MUZZLE_RIGHT,
        }
        _chin_kps = {KP.CHIN, KP.JAW_CENTER}

        def get_skeleton_color(i: int, j: int) -> tuple[int, int, int]:
            """Zwraca kolor dla połączenia skeleton."""
            pts = {i, j}
            if pts & _eye_kps:
                return skeleton_colors["eyes"]
            if pts & _brow_kps:
                return skeleton_colors["brows"]
            if pts & _ear_kps:
                return skeleton_colors["ears"]
            if pts & _nose_kps:
                return skeleton_colors["nose"]
            if pts & _cheek_kps:
                return skeleton_colors["cheeks"]
            if pts & _mouth_kps:
                return skeleton_colors["mouth"]
            if pts & _chin_kps:
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
