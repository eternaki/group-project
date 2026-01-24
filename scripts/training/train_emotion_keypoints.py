#!/usr/bin/env python3
"""
Skrypt do treningu modelu klasyfikacji emocji na podstawie keypoints.

Nowa architektura wykorzystuje 46 keypoints (138 wartości) jako wejście
zamiast surowych pikseli obrazu. Jest to zgodne z podejściem DogFACS.

Użycie:
    python scripts/training/train_emotion_keypoints.py --data_path data/emotions_keypoints.csv

Format danych (CSV):
    - 138 kolumn z keypoints: kp_0_x, kp_0_y, kp_0_v, kp_1_x, ..., kp_45_v
    - 1 kolumna z etykietą: emotion (0-5)

Klasy emocji:
    0: happy
    1: sad
    2: angry
    3: fearful
    4: relaxed
    5: neutral
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Dodaj root projektu do PYTHONPATH
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from packages.models.emotion import (
    KeypointsEmotionMLP,
    EMOTION_CLASSES,
    NUM_EMOTIONS,
    INPUT_FEATURES,
)


class KeypointsEmotionDataset(Dataset):
    """Dataset do treningu modelu emocji na podstawie keypoints."""

    def __init__(self, keypoints: np.ndarray, labels: np.ndarray) -> None:
        """
        Inicjalizuje dataset.

        Args:
            keypoints: Array o kształcie (N, 138) z keypoints
            labels: Array o kształcie (N,) z etykietami emocji
        """
        self.keypoints = torch.from_numpy(keypoints).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keypoints[idx], self.labels[idx]


def load_data_from_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Ładuje dane z pliku CSV.

    Format CSV:
        kp_0_x, kp_0_y, kp_0_v, ..., kp_45_v, emotion

    Args:
        csv_path: Ścieżka do pliku CSV

    Returns:
        Tuple (keypoints, labels)
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Ostatnia kolumna to etykieta
    keypoints = df.iloc[:, :-1].values.astype(np.float32)
    labels = df.iloc[:, -1].values.astype(np.int64)

    return keypoints, labels


def create_synthetic_data(num_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Tworzy syntetyczne dane do testowania pipeline treningu.

    Args:
        num_samples: Liczba próbek

    Returns:
        Tuple (keypoints, labels)
    """
    print(f"Tworzenie {num_samples} syntetycznych próbek...")

    keypoints = np.random.randn(num_samples, INPUT_FEATURES).astype(np.float32)
    labels = np.random.randint(0, NUM_EMOTIONS, num_samples).astype(np.int64)

    return keypoints, labels


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Trenuje model przez jedną epokę."""
    model.train()
    total_loss = 0.0

    for keypoints, labels in tqdm(dataloader, desc="Training", leave=False):
        keypoints = keypoints.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(keypoints)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Ewaluuje model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for keypoints, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            keypoints = keypoints.to(device)
            labels = labels.to(device)

            outputs = model(keypoints)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


def main() -> None:
    """Główna funkcja treningu."""
    parser = argparse.ArgumentParser(
        description="Trening modelu emocji na podstawie keypoints"
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        help="Ścieżka do pliku CSV z danymi (opcjonalne - użyje syntetycznych danych)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=project_root / "models" / "emotion_keypoints.pt",
        help="Ścieżka do zapisania wag modelu",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Liczba epok treningu",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Rozmiar batcha",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Wymiary warstw ukrytych MLP",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate",
    )
    parser.add_argument(
        "--synthetic_samples",
        type=int,
        default=5000,
        help="Liczba syntetycznych próbek (jeśli brak data_path)",
    )

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # Załaduj dane
    if args.data_path and args.data_path.exists():
        print(f"Ładowanie danych z: {args.data_path}")
        keypoints, labels = load_data_from_csv(args.data_path)
    else:
        print("Brak pliku danych - używam syntetycznych danych do testu pipeline")
        keypoints, labels = create_synthetic_data(args.synthetic_samples)

    print(f"Liczba próbek: {len(labels)}")
    print(f"Rozkład klas: {np.bincount(labels)}")

    # Dataset i DataLoader
    dataset = KeypointsEmotionDataset(keypoints, labels)

    # Split na train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = KeypointsEmotionMLP(
        input_dim=INPUT_FEATURES,
        hidden_dims=args.hidden_dims,
        num_classes=NUM_EMOTIONS,
        dropout=args.dropout,
    ).to(device)

    print(f"\nArchitektura modelu:")
    print(model)
    print(f"\nParametry: {sum(p.numel() for p in model.parameters()):,}")

    # Trening
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\nRozpoczynam trening ({args.epochs} epok)...")
    print("=" * 60)

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler
        scheduler.step(val_loss)

        # Historia
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Zapisz najlepszy model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_path)

        # Log
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Best: {best_val_acc:.2f}%"
        )

    print("=" * 60)
    print(f"\nTrening zakończony!")
    print(f"Najlepsza dokładność walidacyjna: {best_val_acc:.2f}%")
    print(f"Model zapisany do: {args.output_path}")

    # Zapisz metryki
    metrics_path = args.output_path.with_suffix(".json")
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "best_val_accuracy": best_val_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "emotion_classes": EMOTION_CLASSES,
        "history": history,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metryki zapisane do: {metrics_path}")


if __name__ == "__main__":
    main()
