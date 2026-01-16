#!/usr/bin/env python3
"""
Skrypt do ewaluacji modelu klasyfikacji ras psów.

Generuje:
- Metryki Top-1 i Top-5 accuracy
- Macierz pomyłek
- Raport per-class accuracy
- Wizualizacje predykcji

Użycie:
    python scripts/training/evaluate_breed.py --weights models/breed.pt
    python scripts/training/evaluate_breed.py --weights runs/breed/best.pt --visualize 20
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Ewaluacja klasyfikatora ras psów"
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
        default="efficientnet_b4",
        help="Architektura modelu (default: efficientnet_b4)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/breed_training"),
        help="Katalog z danymi",
    )
    parser.add_argument(
        "--breeds-json",
        type=Path,
        default=Path("data/breed_training/breeds.json"),
        help="Ścieżka do mapowania ras",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Zbiór do ewaluacji (default: test)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Rozmiar obrazu (default: 224)",
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
        help="Liczba obrazów do wizualizacji (default: 10)",
    )
    return parser.parse_args()


def get_transforms(img_size: int) -> transforms.Compose:
    """Zwraca transformacje dla ewaluacji."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_model(model_name: str, num_classes: int, weights_path: Path, device: torch.device) -> nn.Module:
    """Tworzy i ładuje model."""
    try:
        import timm
    except ImportError:
        raise ImportError("Zainstaluj timm: pip install timm")

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """
    Ewaluuje model.

    Returns:
        Słownik z metrykami
    """
    all_preds = []
    all_labels = []
    all_top5_preds = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        # Top-1
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

        # Top-5
        _, top5 = outputs.topk(5, dim=1)
        all_top5_preds.extend(top5.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_top5_preds = np.array(all_top5_preds)

    # Metryki globalne
    top1_acc = (all_preds == all_labels).mean()
    top5_acc = np.mean([
        all_labels[i] in all_top5_preds[i]
        for i in range(len(all_labels))
    ])

    # Macierz pomyłek
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true, pred] += 1

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (all_preds[mask] == c).mean()
        else:
            per_class_acc[c] = 0.0

    return {
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": per_class_acc,
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "top5_predictions": all_top5_preds.tolist(),
    }


def generate_report(
    metrics: dict,
    breeds: dict[str, str],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """Generuje raport ewaluacji w Markdown."""
    from datetime import datetime

    # Sortuj per-class accuracy
    sorted_acc = sorted(
        metrics["per_class_accuracy"].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Najlepsze i najgorsze rasy
    best_breeds = sorted_acc[:10]
    worst_breeds = sorted_acc[-10:]

    report = f"""# Raport Ewaluacji - Model Klasyfikacji Ras Psów

**Data:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Model:** {args.model}
**Wagi:** {args.weights}
**Zbiór:** {args.split}

---

## 1. Podsumowanie

| Metryka | Wartość | Target |
|---------|---------|--------|
| Top-1 Accuracy | {metrics['top1_accuracy'] * 100:.2f}% | > 60% |
| Top-5 Accuracy | {metrics['top5_accuracy'] * 100:.2f}% | > 80% |

**Status:** {"PASS" if metrics['top5_accuracy'] > 0.80 else "FAIL"} (target Top-5 > 80%)

---

## 2. Top 10 Najlepszych Ras

| Ranga | Rasa | Accuracy |
|-------|------|----------|
"""

    for i, (class_id, acc) in enumerate(best_breeds, 1):
        breed_name = breeds.get(str(class_id), f"Class {class_id}")
        report += f"| {i} | {breed_name} | {acc * 100:.1f}% |\n"

    report += f"""
---

## 3. Top 10 Najgorszych Ras

| Ranga | Rasa | Accuracy |
|-------|------|----------|
"""

    for i, (class_id, acc) in enumerate(reversed(worst_breeds), 1):
        breed_name = breeds.get(str(class_id), f"Class {class_id}")
        report += f"| {i} | {breed_name} | {acc * 100:.1f}% |\n"

    report += f"""
---

## 4. Statystyki

- Liczba klas: {len(breeds)}
- Liczba próbek w zbiorze: {len(metrics['labels'])}
- Średnia accuracy per-class: {np.mean(list(metrics['per_class_accuracy'].values())) * 100:.2f}%

---

## 5. Wnioski

### 5.1 Mocne strony
- [Do uzupełnienia]

### 5.2 Słabe strony
- [Do uzupełnienia]

### 5.3 Rekomendacje
- [Do uzupełnienia]

---

*Raport wygenerowany automatycznie przez evaluate_breed.py*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Raport zapisany: {output_path}")


def visualize_predictions(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    breeds: dict[str, str],
    device: torch.device,
    output_dir: Path,
    num_images: int,
    transform: transforms.Compose,
) -> None:
    """Wizualizuje przykładowe predykcje."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("Brak matplotlib - pomijam wizualizacje")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Losowe indeksy
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Visualizing"):
        img_path, true_label = dataset.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # Predykcja
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            top5_probs, top5_idx = probs.topk(5)

        # Rysuj
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Obraz
        ax1.imshow(img)
        true_name = breeds.get(str(true_label), f"Class {true_label}")
        pred_name = breeds.get(str(top5_idx[0].item()), f"Class {top5_idx[0].item()}")
        color = "green" if top5_idx[0].item() == true_label else "red"
        ax1.set_title(f"True: {true_name}\nPred: {pred_name}", color=color)
        ax1.axis("off")

        # Top-5 predictions
        top5_names = [breeds.get(str(i.item()), f"Class {i.item()}") for i in top5_idx]
        colors = ["green" if i.item() == true_label else "steelblue" for i in top5_idx]
        ax2.barh(range(5), top5_probs.cpu().numpy(), color=colors)
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(top5_names)
        ax2.set_xlabel("Probability")
        ax2.set_title("Top-5 Predictions")
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(output_dir / f"pred_{idx:04d}.png", dpi=100, bbox_inches="tight")
        plt.close()


def main() -> None:
    """Główna funkcja ewaluacji."""
    args = parse_args()

    print("=" * 60)
    print("Ewaluacja klasyfikatora ras psów")
    print("=" * 60)

    device = torch.device(args.device)

    # Sprawdź pliki
    if not args.weights.exists():
        print(f"Nie znaleziono wag: {args.weights}")
        return

    if not args.breeds_json.exists():
        print(f"Nie znaleziono breeds.json: {args.breeds_json}")
        return

    # Wczytaj mapowanie ras
    with open(args.breeds_json) as f:
        breeds = json.load(f)
    num_classes = len(breeds)
    print(f"Liczba klas: {num_classes}")

    # Przygotuj dataset
    transform = get_transforms(args.img_size)
    dataset = datasets.ImageFolder(
        args.data_dir / args.split,
        transform=transform,
    )
    print(f"Zbiór {args.split}: {len(dataset)} obrazów")

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Załaduj model
    print(f"\nŁadowanie modelu {args.model}...")
    model = create_model(args.model, num_classes, args.weights, device)

    # Ewaluacja
    print("\nEwaluacja...")
    metrics = evaluate(model, loader, device, num_classes)

    # Wyświetl wyniki
    print("\n" + "=" * 60)
    print("Wyniki")
    print("=" * 60)
    print(f"Top-1 Accuracy: {metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy'] * 100:.2f}%")

    target = 0.80
    status = "PASS" if metrics["top5_accuracy"] > target else "FAIL"
    print(f"\nStatus: {status} (target Top-5 > {target * 100}%)")

    # Generuj raport
    report_path = args.output_dir / "breed-evaluation.md"
    generate_report(metrics, breeds, args, report_path)

    # Zapisz metryki JSON
    metrics_path = args.output_dir / "breed_metrics.json"
    with open(metrics_path, "w") as f:
        # Nie zapisuj całej macierzy pomyłek do JSON (za duża)
        save_metrics = {
            "top1_accuracy": metrics["top1_accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
            "per_class_accuracy": metrics["per_class_accuracy"],
        }
        json.dump(save_metrics, f, indent=2)
    print(f"Metryki zapisane: {metrics_path}")

    # Wizualizacje
    if args.visualize > 0:
        print(f"\nGenerowanie {args.visualize} wizualizacji...")
        viz_dir = args.output_dir / "breed_visualizations"
        visualize_predictions(
            model, dataset, breeds, device, viz_dir, args.visualize, transform
        )
        print(f"Wizualizacje zapisane: {viz_dir}")


if __name__ == "__main__":
    main()
