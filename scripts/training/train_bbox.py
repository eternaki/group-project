#!/usr/bin/env python3
"""
Skrypt do trenowania modelu YOLOv8 do detekcji psów.

Użycie:
    python scripts/training/train_bbox.py
    python scripts/training/train_bbox.py --epochs 50 --batch 8
    python scripts/training/train_bbox.py --resume runs/bbox/yolov8m_dogs/weights/last.pt
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Trening YOLOv8 do detekcji psów"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Model bazowy (default: yolov8m.pt)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Ścieżka do checkpointu do wznowienia treningu",
    )

    # Dane
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/bbox_training/dataset.yaml"),
        help="Ścieżka do dataset.yaml",
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
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Rozmiar obrazu (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device (default: 0, użyj 'cpu' dla CPU)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Liczba workerów (default: 8)",
    )

    # Optymalizacja
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Początkowy learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Końcowy learning rate = lr0 * lrf (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="SGD momentum (default: 0.937)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay (default: 0.0005)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )

    # Wyjście
    parser.add_argument(
        "--project",
        type=str,
        default="runs/bbox",
        help="Katalog projektu (default: runs/bbox)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nazwa eksperymentu (default: yolov8m_dogs_YYYYMMDD)",
    )

    # Augmentacja
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Włącz augmentację (default: True)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_false",
        dest="augment",
        help="Wyłącz augmentację",
    )

    return parser.parse_args()


def check_environment() -> dict:
    """Sprawdza środowisko treningowe."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
    }

    if info["cuda_available"]:
        for i in range(info["gpu_count"]):
            info["gpu_names"].append(torch.cuda.get_device_name(i))

    return info


def main() -> None:
    """Główna funkcja treningu."""
    args = parse_args()

    # Sprawdź środowisko
    print("=" * 60)
    print("Trening YOLOv8 - Detekcja Psów")
    print("=" * 60)

    env_info = check_environment()
    print(f"PyTorch: {env_info['pytorch_version']}")
    print(f"CUDA: {env_info['cuda_available']}")
    if env_info["cuda_available"]:
        print(f"CUDA version: {env_info['cuda_version']}")
        for i, name in enumerate(env_info["gpu_names"]):
            print(f"GPU {i}: {name}")
    print()

    # Sprawdź czy dataset istnieje
    if not args.data.exists():
        print(f"Nie znaleziono datasetu: {args.data}")
        print("Uruchom najpierw: python scripts/training/prepare_bbox_data.py")
        return

    # Import ultralytics (po sprawdzeniu środowiska)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Nie znaleziono ultralytics. Zainstaluj: pip install ultralytics")
        return

    # Nazwa eksperymentu
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"yolov8m_dogs_{timestamp}"

    print(f"Eksperyment: {args.name}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print()

    # Załaduj model
    if args.resume:
        print(f"Wznawianie treningu z: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Ładowanie modelu bazowego: {args.model}")
        model = YOLO(args.model)

    # Konfiguracja treningu
    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "save": True,
        "save_period": 10,  # Zapisuj co 10 epok
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "SGD",
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,  # Box loss gain
        "cls": 0.5,  # Class loss gain
        "dfl": 1.5,  # DFL loss gain
        "plots": True,
        "verbose": True,
    }

    # Augmentacja
    if args.augment:
        train_args.update({
            "hsv_h": 0.015,  # Hue augmentation
            "hsv_s": 0.7,    # Saturation augmentation
            "hsv_v": 0.4,    # Value augmentation
            "degrees": 10.0,  # Rotation
            "translate": 0.1, # Translation
            "scale": 0.5,    # Scale
            "shear": 0.0,    # Shear
            "perspective": 0.0,
            "flipud": 0.0,   # Flip up-down
            "fliplr": 0.5,   # Flip left-right
            "mosaic": 1.0,   # Mosaic augmentation
            "mixup": 0.0,    # Mixup augmentation
            "copy_paste": 0.0,
        })
    else:
        train_args.update({
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
        })

    print("Rozpoczynam trening...")
    print("-" * 60)

    # Trening
    results = model.train(**train_args)

    # Podsumowanie
    print()
    print("=" * 60)
    print("Trening zakończony!")
    print("=" * 60)

    output_dir = Path(args.project) / args.name
    print(f"Wyniki zapisane w: {output_dir}")
    print(f"Najlepsze wagi: {output_dir}/weights/best.pt")
    print(f"Ostatnie wagi: {output_dir}/weights/last.pt")

    # Wyświetl końcowe metryki
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print()
        print("Końcowe metryki:")
        print(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")

    print()
    print("Następne kroki:")
    print(f"1. Ewaluacja: python scripts/training/evaluate_bbox.py --weights {output_dir}/weights/best.pt")
    print(f"2. Kopiuj wagi: cp {output_dir}/weights/best.pt models/bbox.pt")


if __name__ == "__main__":
    main()
