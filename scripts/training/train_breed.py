#!/usr/bin/env python3
"""
Skrypt do trenowania modelu klasyfikacji ras psów (EfficientNet-B4).

Użycie:
    python scripts/training/train_breed.py
    python scripts/training/train_breed.py --epochs 30 --batch 16
    python scripts/training/train_breed.py --model vit_base_patch16_224
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Trening klasyfikatora ras psów"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b4",
        help="Architektura z timm (default: efficientnet_b4)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Użyj pretrained weights (default: True)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Ścieżka do checkpointu do wznowienia",
    )

    # Dane
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

    # Trening
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Liczba epok (default: 50)",
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
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Liczba workerów (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda jeśli dostępne)",
    )

    # Augmentacja
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Rozmiar obrazu (default: 224)",
    )

    # Wyjście
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/breed"),
        help="Katalog wyjściowy (default: runs/breed)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nazwa eksperymentu",
    )

    return parser.parse_args()


def get_transforms(img_size: int, is_train: bool) -> transforms.Compose:
    """
    Zwraca transformacje dla obrazów.

    Args:
        img_size: Rozmiar obrazu
        is_train: Czy to zbiór treningowy

    Returns:
        Kompozycja transformacji
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),  # 256 dla 224
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])


def create_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    Tworzy model z timm.

    Args:
        model_name: Nazwa modelu
        num_classes: Liczba klas
        pretrained: Czy użyć pretrained weights

    Returns:
        Model PyTorch
    """
    try:
        import timm
    except ImportError:
        raise ImportError("Zainstaluj timm: pip install timm")

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Trenuje model przez jedną epokę.

    Returns:
        Tuple (średnia strata, dokładność)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.2f}%",
        })

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Waliduje model.

    Returns:
        Tuple (średnia strata, top-1 accuracy, top-5 accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        # Top-1
        _, predicted = outputs.max(1)
        correct_top1 += predicted.eq(labels).sum().item()

        # Top-5
        _, top5_pred = outputs.topk(5, dim=1)
        correct_top5 += sum(
            labels[i] in top5_pred[i] for i in range(labels.size(0))
        )

        total += labels.size(0)

    return total_loss / total, correct_top1 / total, correct_top5 / total


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_acc: float,
    output_path: Path,
) -> None:
    """Zapisuje checkpoint modelu."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
    }
    torch.save(checkpoint, output_path)


def main() -> None:
    """Główna funkcja treningu."""
    args = parse_args()

    print("=" * 60)
    print("Trening klasyfikatora ras psów")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print()

    device = torch.device(args.device)

    # Sprawdź dane
    if not args.data_dir.exists():
        print(f"Nie znaleziono danych: {args.data_dir}")
        print("Uruchom najpierw: python scripts/training/prepare_breed_data.py")
        return

    # Wczytaj mapowanie ras
    if not args.breeds_json.exists():
        print(f"Nie znaleziono breeds.json: {args.breeds_json}")
        return

    with open(args.breeds_json) as f:
        breeds = json.load(f)
    num_classes = len(breeds)
    print(f"Liczba klas: {num_classes}")

    # Przygotuj datasety
    train_transform = get_transforms(args.img_size, is_train=True)
    val_transform = get_transforms(args.img_size, is_train=False)

    train_dataset = datasets.ImageFolder(
        args.data_dir / "train",
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        args.data_dir / "val",
        transform=val_transform,
    )

    print(f"Train: {len(train_dataset)} obrazów")
    print(f"Val: {len(val_dataset)} obrazów")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Utwórz model
    print(f"\nŁadowanie modelu {args.model}...")
    model = create_model(args.model, num_classes, args.pretrained)
    model = model.to(device)

    # Optymalizator i scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Wznów trening jeśli checkpoint
    start_epoch = 0
    best_acc = 0.0

    if args.resume and args.resume.exists():
        print(f"Wznawianie z: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]

    # Katalog wyjściowy
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"{args.model}_{timestamp}"

    output_dir = args.output_dir / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zapisz konfigurację
    config = {
        "model": args.model,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "img_size": args.img_size,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Historia treningu
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": [],
    }

    # Trening
    print("\nRozpoczynam trening...")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Trening
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Walidacja
        val_loss, val_top1, val_top5 = validate(
            model, val_loader, criterion, device
        )

        # Scheduler
        scheduler.step()

        # Zapisz historię
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(val_top1)
        history["val_top5"].append(val_top5)

        # Wyświetl metryki
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc * 100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Top-1: {val_top1 * 100:.2f}%, Top-5: {val_top5 * 100:.2f}%")

        # Zapisz najlepszy model
        if val_top5 > best_acc:
            best_acc = val_top5
            print(f"  Nowy najlepszy model! Top-5: {best_acc * 100:.2f}%")
            torch.save(model.state_dict(), output_dir / "best.pt")

        # Checkpoint co 10 epok
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_acc,
                output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            )

    # Zapisz ostatni model
    torch.save(model.state_dict(), output_dir / "last.pt")

    # Zapisz historię
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Podsumowanie
    print("\n" + "=" * 60)
    print("Trening zakończony!")
    print("=" * 60)
    print(f"Najlepsza Top-5 accuracy: {best_acc * 100:.2f}%")
    print(f"Wyniki zapisane w: {output_dir}")
    print(f"\nNajlepsze wagi: {output_dir / 'best.pt'}")
    print(f"Skopiuj do: cp {output_dir / 'best.pt'} models/breed.pt")


if __name__ == "__main__":
    main()
