#!/usr/bin/env python3
"""
Skrypt do ewaluacji modelu YOLOv8 do detekcji psów.

Generuje:
- Metryki (mAP, Precision, Recall)
- Macierz pomyłek
- Wizualizacje predykcji
- Pomiar czasu inference

Użycie:
    python scripts/training/evaluate_bbox.py --weights models/bbox.pt
    python scripts/training/evaluate_bbox.py --weights runs/bbox/yolov8m_dogs/weights/best.pt
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Ewaluacja modelu YOLOv8 do detekcji psów"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Ścieżka do wag modelu (.pt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/bbox_training/dataset.yaml"),
        help="Ścieżka do dataset.yaml",
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
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device (default: 0, 'cpu' dla CPU)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Próg confidence (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Próg IoU dla NMS (default: 0.45)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/reports"),
        help="Katalog na raport (default: docs/reports)",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=10,
        help="Liczba obrazów do wizualizacji (default: 10)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Wykonaj benchmark czasu inference",
    )
    return parser.parse_args()


def measure_inference_time(
    model,
    imgsz: int = 640,
    device: str = "0",
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """
    Mierzy czas inference modelu.

    Args:
        model: Model YOLO
        imgsz: Rozmiar obrazu
        device: Device
        warmup: Liczba iteracji rozgrzewkowych
        iterations: Liczba iteracji pomiarowych

    Returns:
        Słownik z czasami
    """
    import torch

    # Przygotuj dummy image
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    print(f"Warmup ({warmup} iteracji)...")
    for _ in range(warmup):
        model(dummy, verbose=False)

    # Pomiar
    times = []
    print(f"Pomiar ({iterations} iteracji)...")

    for _ in range(iterations):
        start = time.perf_counter()
        model(dummy, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    # Synchronizacja CUDA
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    return {
        "device": device,
        "iterations": iterations,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
        "fps": 1000 / np.mean(times),
    }


def visualize_predictions(
    model,
    data_path: Path,
    split: str,
    output_dir: Path,
    num_images: int = 10,
    conf: float = 0.25,
) -> list[Path]:
    """
    Wizualizuje predykcje modelu.

    Args:
        model: Model YOLO
        data_path: Ścieżka do dataset.yaml
        split: Zbiór (train/val/test)
        output_dir: Katalog wyjściowy
        num_images: Liczba obrazów
        conf: Próg confidence

    Returns:
        Lista ścieżek do zapisanych wizualizacji
    """
    import yaml

    # Wczytaj konfigurację datasetu
    with open(data_path) as f:
        data_config = yaml.safe_load(f)

    dataset_path = Path(data_config["path"])
    images_dir = dataset_path / data_config[split]

    if not images_dir.exists():
        print(f"Nie znaleziono katalogu: {images_dir}")
        return []

    # Pobierz listę obrazów
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    np.random.shuffle(image_files)
    image_files = image_files[:num_images]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for i, img_path in enumerate(image_files):
        # Predykcja
        results = model(str(img_path), conf=conf, verbose=False)[0]

        # Rysuj wyniki
        annotated = results.plot()

        # Zapisz
        output_path = output_dir / f"prediction_{i:03d}.jpg"
        cv2.imwrite(str(output_path), annotated)
        saved_paths.append(output_path)

    return saved_paths


def generate_report(
    metrics: dict,
    inference_times: Optional[dict],
    visualization_paths: list[Path],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """
    Generuje raport ewaluacji w formacie Markdown.

    Args:
        metrics: Metryki ewaluacji
        inference_times: Czasy inference (opcjonalnie)
        visualization_paths: Ścieżki do wizualizacji
        args: Argumenty skryptu
        output_path: Ścieżka do raportu
    """
    from datetime import datetime

    report = f"""# Raport Ewaluacji - Model Detekcji Psów (YOLOv8)

**Data:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Model:** {args.weights}
**Dataset:** {args.data}
**Zbiór:** {args.split}

---

## 1. Podsumowanie

| Metryka | Wartość | Target |
|---------|---------|--------|
| mAP@0.5 | {metrics.get('mAP50', 'N/A'):.4f} | > 0.85 |
| mAP@0.5:0.95 | {metrics.get('mAP50-95', 'N/A'):.4f} | > 0.70 |
| Precision | {metrics.get('precision', 'N/A'):.4f} | > 0.85 |
| Recall | {metrics.get('recall', 'N/A'):.4f} | > 0.80 |

**Status:** {"PASS" if metrics.get('mAP50', 0) > 0.85 else "FAIL"} (target mAP@0.5 > 85%)

---

## 2. Szczegółowe Metryki

### 2.1 Box Metrics

| Metryka | Wartość |
|---------|---------|
| mAP@0.5 | {metrics.get('mAP50', 'N/A'):.4f} |
| mAP@0.5:0.95 | {metrics.get('mAP50-95', 'N/A'):.4f} |
| Precision | {metrics.get('precision', 'N/A'):.4f} |
| Recall | {metrics.get('recall', 'N/A'):.4f} |

"""

    if inference_times:
        report += f"""### 2.2 Czas Inference

| Parametr | Wartość |
|----------|---------|
| Device | {inference_times['device']} |
| Średni czas | {inference_times['mean_ms']:.2f} ms |
| Std | {inference_times['std_ms']:.2f} ms |
| Min | {inference_times['min_ms']:.2f} ms |
| Max | {inference_times['max_ms']:.2f} ms |
| FPS | {inference_times['fps']:.1f} |

**Status:** {"PASS" if inference_times['mean_ms'] < 50 else "FAIL"} (target < 50ms na GPU)

"""

    report += """---

## 3. Wizualizacje

### 3.1 Przykładowe Predykcje

"""

    for i, path in enumerate(visualization_paths):
        rel_path = path.relative_to(output_path.parent) if path.is_relative_to(output_path.parent) else path
        report += f"![Prediction {i+1}]({rel_path})\n\n"

    report += """---

## 4. Wnioski

### 4.1 Mocne strony
- [Do uzupełnienia na podstawie analizy]

### 4.2 Słabe strony / Przypadki błędów
- [Do uzupełnienia na podstawie analizy]

### 4.3 Rekomendacje
- [Do uzupełnienia na podstawie analizy]

---

*Raport wygenerowany automatycznie przez evaluate_bbox.py*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Raport zapisany: {output_path}")


def main() -> None:
    """Główna funkcja ewaluacji."""
    args = parse_args()

    print("=" * 60)
    print("Ewaluacja Modelu YOLOv8 - Detekcja Psów")
    print("=" * 60)

    # Sprawdź pliki
    if not args.weights.exists():
        print(f"Nie znaleziono wag: {args.weights}")
        return

    if not args.data.exists():
        print(f"Nie znaleziono datasetu: {args.data}")
        return

    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Nie znaleziono ultralytics. Zainstaluj: pip install ultralytics")
        return

    # Załaduj model
    print(f"Ładowanie modelu: {args.weights}")
    model = YOLO(str(args.weights))

    # Ewaluacja
    print(f"\nEwaluacja na zbiorze: {args.split}")
    print("-" * 60)

    val_results = model.val(
        data=str(args.data),
        split=args.split,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        save_json=True,
        plots=True,
        verbose=True,
    )

    # Wyodrębnij metryki
    metrics = {
        "mAP50": float(val_results.box.map50),
        "mAP50-95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
    }

    print("\n" + "=" * 60)
    print("Wyniki Ewaluacji")
    print("=" * 60)
    print(f"mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    # Benchmark (opcjonalnie)
    inference_times = None
    if args.benchmark:
        print("\n" + "-" * 60)
        print("Benchmark czasu inference")
        print("-" * 60)
        inference_times = measure_inference_time(
            model,
            device=args.device,
        )
        print(f"Średni czas: {inference_times['mean_ms']:.2f} ms")
        print(f"FPS: {inference_times['fps']:.1f}")

    # Wizualizacje
    visualization_paths = []
    if args.visualize > 0:
        print("\n" + "-" * 60)
        print(f"Generowanie {args.visualize} wizualizacji...")
        viz_dir = args.output_dir / "bbox_visualizations"
        visualization_paths = visualize_predictions(
            model,
            args.data,
            args.split,
            viz_dir,
            num_images=args.visualize,
            conf=args.conf,
        )
        print(f"Wizualizacje zapisane w: {viz_dir}")

    # Generuj raport
    print("\n" + "-" * 60)
    print("Generowanie raportu...")
    report_path = args.output_dir / "bbox-evaluation.md"
    generate_report(
        metrics,
        inference_times,
        visualization_paths,
        args,
        report_path,
    )

    # Zapisz metryki jako JSON
    metrics_path = args.output_dir / "bbox_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "inference_times": inference_times,
            "config": {
                "weights": str(args.weights),
                "data": str(args.data),
                "split": args.split,
                "conf": args.conf,
                "iou": args.iou,
            },
        }, f, indent=2)
    print(f"Metryki zapisane: {metrics_path}")

    # Podsumowanie
    print("\n" + "=" * 60)
    print("Podsumowanie")
    print("=" * 60)

    target_map = 0.85
    status = "PASS" if metrics["mAP50"] > target_map else "FAIL"
    print(f"mAP@0.5: {metrics['mAP50']:.4f} (target: >{target_map}) - {status}")

    if status == "PASS":
        print("\nModel spełnia wymagania! Można przejść do integracji.")
        print(f"Skopiuj wagi: cp {args.weights} models/bbox.pt")
    else:
        print("\nModel nie spełnia wymagań. Rozważ:")
        print("- Więcej epok treningu")
        print("- Więcej danych treningowych")
        print("- Dostrojenie hiperparametrów")


if __name__ == "__main__":
    main()
