#!/usr/bin/env python3
"""
Skrypt do ewaluacji modelu detekcji keypoints.

Generuje:
- PCK@0.1 metrykę
- Per-keypoint accuracy
- Wizualizacje predykcji

Użycie:
    python scripts/training/evaluate_keypoints.py --weights models/keypoints.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw

# Import lokalnych modułów
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from packages.data.schemas import (
    NUM_KEYPOINTS,
    KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
    get_keypoint_color,
)


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Ewaluacja modelu detekcji keypoints"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Ścieżka do wag modelu",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Architektura modelu (default: resnet50)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/keypoints_training"),
        help="Katalog z danymi",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Zbiór do ewaluacji",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/reports"),
        help="Katalog na raport",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=10,
        help="Liczba wizualizacji (default: 10)",
    )
    return parser.parse_args()


def decode_heatmaps(heatmaps: torch.Tensor, img_size: int = 256) -> torch.Tensor:
    """
    Dekoduje keypoints z heatmap.

    Args:
        heatmaps: Tensor [B, NUM_KEYPOINTS, H, W]
        img_size: Rozmiar oryginalnego obrazu

    Returns:
        keypoints: Tensor [B, NUM_KEYPOINTS, 2] - (x, y) w pikselach
    """
    batch_size = heatmaps.shape[0]
    num_kps = heatmaps.shape[1]
    hm_size = heatmaps.shape[-1]
    scale = img_size / hm_size

    keypoints = torch.zeros(batch_size, num_kps, 2)

    for b in range(batch_size):
        for k in range(num_kps):
            hm = heatmaps[b, k]
            max_idx = hm.argmax()
            y = (max_idx // hm_size).float() * scale
            x = (max_idx % hm_size).float() * scale
            keypoints[b, k] = torch.tensor([x, y])

    return keypoints


def compute_pck(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    visibility: torch.Tensor,
    threshold: float = 0.1,
    img_size: int = 256,
) -> tuple[float, dict[int, float]]:
    """
    Oblicza PCK (Percentage of Correct Keypoints).

    Args:
        pred_keypoints: [B, NUM_KEYPOINTS, 2]
        gt_keypoints: [B, NUM_KEYPOINTS, 2]
        visibility: [B, NUM_KEYPOINTS]
        threshold: Próg jako procent rozmiaru obrazu
        img_size: Rozmiar obrazu

    Returns:
        overall_pck: Ogólny PCK
        per_keypoint_pck: PCK dla każdego keypointa
    """
    batch_size = pred_keypoints.shape[0]
    num_kps = pred_keypoints.shape[1]

    correct_per_kp = {k: 0 for k in range(num_kps)}
    total_per_kp = {k: 0 for k in range(num_kps)}

    threshold_px = threshold * img_size

    for b in range(batch_size):
        for k in range(num_kps):
            if visibility[b, k] > 0:
                pred = pred_keypoints[b, k]
                gt = gt_keypoints[b, k]
                dist = torch.sqrt(((pred - gt) ** 2).sum())

                total_per_kp[k] += 1
                if dist < threshold_px:
                    correct_per_kp[k] += 1

    # Oblicz PCK
    per_keypoint_pck = {}
    for k in range(num_kps):
        if total_per_kp[k] > 0:
            per_keypoint_pck[k] = correct_per_kp[k] / total_per_kp[k]
        else:
            per_keypoint_pck[k] = 0.0

    overall_correct = sum(correct_per_kp.values())
    overall_total = sum(total_per_kp.values())
    overall_pck = overall_correct / overall_total if overall_total > 0 else 0.0

    return overall_pck, per_keypoint_pck


def visualize_keypoints(
    image: Image.Image,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    output_path: Path,
    gt_keypoints: np.ndarray | None = None,
) -> None:
    """
    Wizualizuje keypoints na obrazie.

    Args:
        image: Obraz PIL
        keypoints: [NUM_KEYPOINTS, 2] - predykcje
        visibility: [NUM_KEYPOINTS] - widoczność
        output_path: Ścieżka do zapisu
        gt_keypoints: Opcjonalne ground truth keypoints
    """
    draw = ImageDraw.Draw(image)
    radius = 3

    # Rysuj skeleton
    for i, j in SKELETON_CONNECTIONS:
        if visibility[i] > 0 and visibility[j] > 0:
            x1, y1 = keypoints[i]
            x2, y2 = keypoints[j]
            draw.line([(x1, y1), (x2, y2)], fill=(100, 100, 100), width=1)

    # Rysuj keypoints
    for k in range(NUM_KEYPOINTS):
        if visibility[k] > 0:
            x, y = keypoints[k]
            color = get_keypoint_color(k)

            # Predykcja
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=color,
                outline=(255, 255, 255),
            )

            # GT (jeśli dostępne) - mniejszy, inny kolor
            if gt_keypoints is not None:
                gt_x, gt_y = gt_keypoints[k]
                draw.ellipse(
                    [(gt_x - 2, gt_y - 2), (gt_x + 2, gt_y + 2)],
                    fill=(255, 0, 0),
                    outline=(255, 255, 255),
                )

    image.save(output_path)


def generate_report(
    overall_pck: float,
    per_keypoint_pck: dict[int, float],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """Generuje raport ewaluacji."""
    from datetime import datetime

    # Sortuj keypoints wg PCK
    sorted_kps = sorted(per_keypoint_pck.items(), key=lambda x: x[1], reverse=True)

    report = f"""# Raport Ewaluacji - Model Detekcji Keypoints

**Data:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Model:** {args.model}
**Wagi:** {args.weights}
**Zbiór:** {args.split}

---

## 1. Podsumowanie

| Metryka | Wartość | Target |
|---------|---------|--------|
| PCK@0.1 | {overall_pck * 100:.2f}% | > 75% |

**Status:** {"PASS" if overall_pck > 0.75 else "FAIL"} (target PCK@0.1 > 75%)

---

## 2. PCK per Keypoint

| Keypoint | PCK@0.1 |
|----------|---------|
"""

    for k, pck in sorted_kps:
        name = KEYPOINT_NAMES[k]
        report += f"| {name} | {pck * 100:.1f}% |\n"

    report += f"""
---

## 3. Najlepsze Keypoints

| Ranga | Keypoint | PCK@0.1 |
|-------|----------|---------|
"""

    for i, (k, pck) in enumerate(sorted_kps[:5], 1):
        name = KEYPOINT_NAMES[k]
        report += f"| {i} | {name} | {pck * 100:.1f}% |\n"

    report += f"""
---

## 4. Najgorsze Keypoints

| Ranga | Keypoint | PCK@0.1 |
|-------|----------|---------|
"""

    for i, (k, pck) in enumerate(reversed(sorted_kps[-5:]), 1):
        name = KEYPOINT_NAMES[k]
        report += f"| {i} | {name} | {pck * 100:.1f}% |\n"

    report += f"""
---

## 5. Rekomendacje

- [Do uzupełnienia na podstawie analizy]

---

*Raport wygenerowany automatycznie przez evaluate_keypoints.py*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Raport zapisany: {output_path}")


def main() -> None:
    """Główna funkcja ewaluacji."""
    args = parse_args()

    print("=" * 60)
    print("Ewaluacja modelu detekcji keypoints")
    print("=" * 60)

    device = torch.device(args.device)

    # Sprawdź pliki
    if not args.weights.exists():
        print(f"❌ Nie znaleziono wag: {args.weights}")
        return

    # Import modelu
    from scripts.training.train_keypoints import (
        KeypointsDataset,
        create_model,
    )

    # Dataset
    json_path = args.data_dir / f"{args.split}.json"
    images_dir = args.data_dir / "images"

    if not json_path.exists():
        print(f"❌ Nie znaleziono: {json_path}")
        return

    dataset = KeypointsDataset(json_path, images_dir, 256, is_train=False)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4)

    print(f"Zbiór {args.split}: {len(dataset)} obrazów")

    # Model
    model = create_model(args.model, NUM_KEYPOINTS, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    # Ewaluacja
    print("\nEwaluacja...")

    all_pred_kps = []
    all_gt_kps = []
    all_visibility = []

    with torch.no_grad():
        for images, keypoints, visibility in tqdm(loader):
            images = images.to(device)

            heatmaps = model(images)
            pred_kps = decode_heatmaps(heatmaps, img_size=256)

            all_pred_kps.append(pred_kps.cpu())
            all_gt_kps.append(keypoints)
            all_visibility.append(visibility)

    all_pred_kps = torch.cat(all_pred_kps, dim=0)
    all_gt_kps = torch.cat(all_gt_kps, dim=0)
    all_visibility = torch.cat(all_visibility, dim=0)

    # Oblicz metryki
    overall_pck, per_keypoint_pck = compute_pck(
        all_pred_kps, all_gt_kps, all_visibility
    )

    # Wyświetl wyniki
    print("\n" + "=" * 60)
    print("Wyniki")
    print("=" * 60)
    print(f"PCK@0.1: {overall_pck * 100:.2f}%")
    print(f"Target: > 75%")
    print(f"Status: {'PASS' if overall_pck > 0.75 else 'FAIL'}")

    # Generuj raport
    report_path = args.output_dir / "keypoints-evaluation.md"
    generate_report(overall_pck, per_keypoint_pck, args, report_path)

    # Zapisz metryki JSON
    metrics = {
        "pck_0.1": overall_pck,
        "per_keypoint_pck": {KEYPOINT_NAMES[k]: v for k, v in per_keypoint_pck.items()},
    }
    metrics_path = args.output_dir / "keypoints_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metryki zapisane: {metrics_path}")

    # Wizualizacje
    if args.visualize > 0:
        print(f"\nGenerowanie {args.visualize} wizualizacji...")
        viz_dir = args.output_dir / "keypoints_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Losowe indeksy
        indices = np.random.choice(len(dataset), min(args.visualize, len(dataset)), replace=False)

        for idx in tqdm(indices, desc="Visualizing"):
            img_tensor, gt_kps, vis = dataset[idx]

            # Denormalizuj obraz
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
            img_tensor = img_tensor.clamp(0, 1)

            # Konwertuj do PIL
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            image = Image.fromarray(img_np)

            # Predykcja
            with torch.no_grad():
                hm = model(img_tensor.unsqueeze(0).to(device))
                pred_kps = decode_heatmaps(hm, 256)[0].cpu().numpy()

            visualize_keypoints(
                image,
                pred_kps,
                vis.numpy(),
                viz_dir / f"pred_{idx:04d}.png",
                gt_kps.numpy(),
            )

        print(f"Wizualizacje zapisane: {viz_dir}")


if __name__ == "__main__":
    main()
