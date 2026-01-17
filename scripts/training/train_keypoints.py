#!/usr/bin/env python3
"""
Skrypt do trenowania modelu detekcji keypoints (HRNet lub SimpleBaseline).

Użycie:
    python scripts/training/train_keypoints.py
    python scripts/training/train_keypoints.py --model hrnet_w32 --epochs 100
    python scripts/training/train_keypoints.py --model resnet50 --batch 16
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


# Liczba keypoints
NUM_KEYPOINTS = 20


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Trening modelu detekcji keypoints"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="hrnet_w32",
        choices=["hrnet_w32", "hrnet_w48", "resnet50", "resnet101"],
        help="Architektura modelu (default: hrnet_w32)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Użyj pretrained weights",
    )

    # Dane
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/keypoints_training"),
        help="Katalog z danymi",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Rozmiar obrazu (default: 256)",
    )

    # Trening
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Liczba epok (default: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Wyjście
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/keypoints"),
        help="Katalog wyjściowy",
    )

    return parser.parse_args()


class KeypointsDataset(Dataset):
    """Dataset dla keypoints w formacie COCO."""

    def __init__(
        self,
        annotations_path: Path,
        images_dir: Path,
        img_size: int = 256,
        is_train: bool = True,
    ):
        """
        Args:
            annotations_path: Ścieżka do pliku JSON z anotacjami COCO
            images_dir: Katalog z obrazami
            img_size: Rozmiar obrazu wyjściowego
            is_train: Czy to zbiór treningowy (włącza augmentacje)
        """
        with open(annotations_path) as f:
            self.coco = json.load(f)

        self.images_dir = images_dir
        self.img_size = img_size
        self.is_train = is_train

        # Mapuj image_id do file_name
        self.id_to_filename = {
            img["id"]: img["file_name"] for img in self.coco["images"]
        }
        self.id_to_size = {
            img["id"]: (img["width"], img["height"]) for img in self.coco["images"]
        }

        self.annotations = self.coco["annotations"]

        # Transformacje
        self.transform = self._get_transforms()

    def _get_transforms(self) -> transforms.Compose:
        """Zwraca transformacje dla obrazów."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if self.is_train:
            return transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                ),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor [3, H, W]
            keypoints: Tensor [NUM_KEYPOINTS, 2] - (x, y) w pikselach
            visibility: Tensor [NUM_KEYPOINTS] - widoczność
        """
        ann = self.annotations[idx]
        image_id = ann["image_id"]
        filename = self.id_to_filename[image_id]
        orig_width, orig_height = self.id_to_size[image_id]

        # Wczytaj obraz
        img_path = self.images_dir / filename
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            # Fallback na szary obraz
            image = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        # Resize
        image = image.resize((self.img_size, self.img_size))

        # Parsuj keypoints
        kps_flat = ann["keypoints"]
        keypoints = []
        visibility = []

        for i in range(NUM_KEYPOINTS):
            x = kps_flat[i * 3] / orig_width * self.img_size
            y = kps_flat[i * 3 + 1] / orig_height * self.img_size
            v = kps_flat[i * 3 + 2]
            keypoints.append([x, y])
            visibility.append(v)

        # Augmentacje dla keypoints (horizontal flip)
        if self.is_train and np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # Flip keypoints
            keypoints = [[self.img_size - x, y] for x, y in keypoints]
            # Swap left/right pairs
            swap_pairs = [(0, 1), (3, 4), (5, 6), (7, 8), (12, 13), (15, 16), (18, 19)]
            for i, j in swap_pairs:
                keypoints[i], keypoints[j] = keypoints[j], keypoints[i]
                visibility[i], visibility[j] = visibility[j], visibility[i]

        # Transformacje
        image = self.transform(image)

        return (
            image,
            torch.tensor(keypoints, dtype=torch.float32),
            torch.tensor(visibility, dtype=torch.float32),
        )


class SimpleBaseline(nn.Module):
    """
    Simple Baseline model dla pose estimation.

    Architektura: ResNet backbone + deconv head
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_keypoints: int = 20,
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

        # Określ liczbę kanałów wyjściowych backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            features = self.backbone(dummy)
            backbone_channels = features[-1].shape[1]

        # Deconv head
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(backbone_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final conv
        self.final = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, 3, H, W]

        Returns:
            heatmaps: Tensor [B, NUM_KEYPOINTS, H/4, W/4]
        """
        features = self.backbone(x)[-1]
        x = self.deconv(features)
        heatmaps = self.final(x)
        return heatmaps


def create_model(model_name: str, num_keypoints: int, pretrained: bool) -> nn.Module:
    """Tworzy model do detekcji keypoints."""
    if model_name.startswith("hrnet"):
        # Próbuj użyć HRNet z timm
        try:
            import timm
            backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
            # Dodaj head
            model = SimpleBaseline(model_name.replace("hrnet_", "hrnet_"), num_keypoints, pretrained)
        except Exception:
            print(f"HRNet niedostępny, używam ResNet50")
            model = SimpleBaseline("resnet50", num_keypoints, pretrained)
    else:
        model = SimpleBaseline(model_name, num_keypoints, pretrained)

    return model


def generate_heatmaps(
    keypoints: torch.Tensor,
    visibility: torch.Tensor,
    heatmap_size: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    Generuje ground truth heatmapy dla keypoints.

    Args:
        keypoints: Tensor [B, NUM_KEYPOINTS, 2]
        visibility: Tensor [B, NUM_KEYPOINTS]
        heatmap_size: Rozmiar heatmapy
        sigma: Sigma dla Gaussiana

    Returns:
        heatmaps: Tensor [B, NUM_KEYPOINTS, heatmap_size, heatmap_size]
    """
    batch_size = keypoints.shape[0]
    num_kps = keypoints.shape[1]

    heatmaps = torch.zeros(batch_size, num_kps, heatmap_size, heatmap_size)

    # Skaluj keypoints do rozmiaru heatmapy
    scale = heatmap_size / 256.0

    for b in range(batch_size):
        for k in range(num_kps):
            if visibility[b, k] > 0:
                x, y = keypoints[b, k]
                x = x * scale
                y = y * scale

                # Generuj Gaussiana
                xx, yy = torch.meshgrid(
                    torch.arange(heatmap_size),
                    torch.arange(heatmap_size),
                    indexing='xy',
                )
                xx = xx.float()
                yy = yy.float()

                heatmap = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
                heatmaps[b, k] = heatmap

    return heatmaps


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Trenuje model przez jedną epokę."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, keypoints, visibility in pbar:
        images = images.to(device)
        keypoints = keypoints.to(device)
        visibility = visibility.to(device)

        # Forward
        heatmaps_pred = model(images)
        heatmap_size = heatmaps_pred.shape[-1]

        # Ground truth heatmaps
        heatmaps_gt = generate_heatmaps(keypoints, visibility, heatmap_size)
        heatmaps_gt = heatmaps_gt.to(device)

        # Loss
        loss = criterion(heatmaps_pred, heatmaps_gt)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Waliduje model.

    Returns:
        Tuple (loss, PCK@0.1)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    threshold = 0.1  # PCK threshold (10% of bbox size)

    for images, keypoints_gt, visibility in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        keypoints_gt = keypoints_gt.to(device)
        visibility = visibility.to(device)

        # Forward
        heatmaps_pred = model(images)
        heatmap_size = heatmaps_pred.shape[-1]

        # GT heatmaps
        heatmaps_gt = generate_heatmaps(keypoints_gt, visibility, heatmap_size)
        heatmaps_gt = heatmaps_gt.to(device)

        loss = criterion(heatmaps_pred, heatmaps_gt)
        total_loss += loss.item() * images.size(0)

        # Decode keypoints from heatmaps
        batch_size = images.shape[0]
        scale = 256.0 / heatmap_size

        for b in range(batch_size):
            for k in range(NUM_KEYPOINTS):
                if visibility[b, k] > 0:
                    # Znajdź maksimum w heatmapie
                    hm = heatmaps_pred[b, k]
                    max_idx = hm.argmax()
                    pred_y = (max_idx // heatmap_size).float() * scale
                    pred_x = (max_idx % heatmap_size).float() * scale

                    # Ground truth
                    gt_x, gt_y = keypoints_gt[b, k]

                    # Dystans
                    dist = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

                    # PCK threshold (10% of image size)
                    if dist < threshold * 256:
                        correct += 1
                    total += 1

    pck = correct / total if total > 0 else 0.0
    return total_loss / len(loader.dataset), pck


def main() -> None:
    """Główna funkcja treningu."""
    args = parse_args()

    print("=" * 60)
    print("Trening modelu detekcji keypoints")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print()

    device = torch.device(args.device)

    # Sprawdź dane
    train_json = args.data_dir / "train.json"
    val_json = args.data_dir / "val.json"
    images_dir = args.data_dir / "images"

    if not train_json.exists():
        print(f"❌ Nie znaleziono: {train_json}")
        print("Uruchom najpierw: python scripts/training/prepare_keypoints_data.py")
        return

    # Datasety
    train_dataset = KeypointsDataset(train_json, images_dir, args.img_size, is_train=True)
    val_dataset = KeypointsDataset(val_json, images_dir, args.img_size, is_train=False)

    print(f"Train: {len(train_dataset)} obrazów")
    print(f"Val: {len(val_dataset)} obrazów")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = create_model(args.model, NUM_KEYPOINTS, args.pretrained)
    model = model.to(device)

    # Optymalizator
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Trening
    best_pck = 0.0
    history = {"train_loss": [], "val_loss": [], "val_pck": []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pck = validate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pck"].append(val_pck)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, PCK@0.1: {val_pck * 100:.2f}%")

        if val_pck > best_pck:
            best_pck = val_pck
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"  ⭐ Nowy najlepszy model! PCK@0.1: {best_pck * 100:.2f}%")

    # Zapisz ostatni model i historię
    torch.save(model.state_dict(), output_dir / "last.pt")
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Trening zakończony!")
    print("=" * 60)
    print(f"Najlepszy PCK@0.1: {best_pck * 100:.2f}%")
    print(f"Target: > 75%")
    print(f"Status: {'PASS' if best_pck > 0.75 else 'FAIL'}")
    print(f"\nWyniki: {output_dir}")


if __name__ == "__main__":
    main()
