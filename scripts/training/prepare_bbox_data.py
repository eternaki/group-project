#!/usr/bin/env python3
"""
Skrypt do przygotowania danych treningowych dla modelu detekcji psów (YOLOv8).

Obsługuje:
- Stanford Dogs Dataset (z Kaggle)
- Open Images V7 (podzbiór psów)

Konwertuje do formatu YOLO i tworzy podział train/val/test.

Użycie:
    python scripts/training/prepare_bbox_data.py --source stanford
    python scripts/training/prepare_bbox_data.py --source openimages --limit 10000
    python scripts/training/prepare_bbox_data.py --source all
"""

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


# Ścieżki domyślne
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
BBOX_TRAINING_DIR = DATA_DIR / "bbox_training"


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Przygotowanie danych dla modelu detekcji psów"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["stanford", "openimages", "all"],
        default="stanford",
        help="Źródło danych (default: stanford)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit obrazów do przetworzenia (dla Open Images)",
    )
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.8,0.1,0.1",
        help="Proporcje train/val/test (default: 0.8,0.1,0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed dla podziału danych (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BBOX_TRAINING_DIR,
        help=f"Katalog wyjściowy (default: {BBOX_TRAINING_DIR})",
    )
    return parser.parse_args()


def setup_directories(output_dir: Path) -> None:
    """Tworzy strukturę katalogów dla datasetu YOLO."""
    dirs = [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "images" / "test",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
        output_dir / "labels" / "test",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Utworzono strukturę katalogów w {output_dir}")


def convert_bbox_to_yolo(
    bbox: tuple[int, int, int, int],
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """
    Konwertuje bounding box z formatu (x, y, w, h) do formatu YOLO.

    Args:
        bbox: Bounding box jako (x, y, width, height) w pikselach
        img_width: Szerokość obrazu
        img_height: Wysokość obrazu

    Returns:
        Tuple (x_center, y_center, width, height) znormalizowane do [0, 1]
    """
    x, y, w, h = bbox

    # Oblicz środek
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height

    # Normalizuj szerokość i wysokość
    width = w / img_width
    height = h / img_height

    # Ogranicz do [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def parse_stanford_annotation(xml_path: Path) -> Optional[dict]:
    """
    Parsuje plik XML z anotacją Stanford Dogs.

    Args:
        xml_path: Ścieżka do pliku XML

    Returns:
        Słownik z informacjami o obrazie i bounding boxach
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Rozmiar obrazu
        size = root.find("size")
        if size is None:
            return None

        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # Nazwa pliku
        filename = root.find("filename").text
        if not filename.endswith((".jpg", ".jpeg", ".png")):
            filename += ".jpg"

        # Bounding boxy
        bboxes = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # Konwertuj do (x, y, w, h)
                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                bboxes.append(bbox)

        if not bboxes:
            return None

        return {
            "filename": filename,
            "width": width,
            "height": height,
            "bboxes": bboxes,
        }
    except Exception as e:
        print(f"Błąd parsowania {xml_path}: {e}")
        return None


def process_stanford_dogs(
    raw_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Przetwarza Stanford Dogs Dataset.

    Args:
        raw_dir: Katalog z surowymi danymi
        output_dir: Katalog wyjściowy
        limit: Limit obrazów (None = wszystkie)

    Returns:
        Lista słowników z informacjami o przetworzonych obrazach
    """
    stanford_dir = raw_dir / "stanford_dogs"
    images_dir = stanford_dir / "Images"
    annotations_dir = stanford_dir / "Annotation"

    if not images_dir.exists():
        print(f"Nie znaleziono Stanford Dogs w {stanford_dir}")
        print("Pobierz dataset: kaggle datasets download jessicali9530/stanford-dogs-dataset")
        return []

    processed = []
    image_files = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.JPEG"))

    if limit:
        image_files = image_files[:limit]

    print(f"Przetwarzanie {len(image_files)} obrazów ze Stanford Dogs...")

    for img_path in tqdm(image_files, desc="Stanford Dogs"):
        # Znajdź odpowiadający plik anotacji
        breed_folder = img_path.parent.name
        img_name = img_path.stem

        annotation_path = annotations_dir / breed_folder / img_name

        if not annotation_path.exists():
            # Spróbuj bez rozszerzenia w nazwie
            continue

        annotation = parse_stanford_annotation(annotation_path)
        if annotation is None:
            continue

        # Dodaj ścieżkę źródłową
        annotation["source_path"] = img_path
        annotation["source"] = "stanford"
        processed.append(annotation)

    print(f"Przetworzono {len(processed)} obrazów ze Stanford Dogs")
    return processed


def process_open_images(
    raw_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Przetwarza Open Images V7 (podzbiór psów).

    Args:
        raw_dir: Katalog z surowymi danymi
        output_dir: Katalog wyjściowy
        limit: Limit obrazów

    Returns:
        Lista słowników z informacjami o przetworzonych obrazach
    """
    oi_dir = raw_dir / "open_images"
    images_dir = oi_dir / "images"
    annotations_file = oi_dir / "dog_annotations.csv"

    if not annotations_file.exists():
        print(f"Nie znaleziono Open Images w {oi_dir}")
        print("Pobierz podzbiór psów z Open Images V7")
        return []

    # Tutaj byłaby implementacja parsowania Open Images
    # Format CSV: ImageID,XMin,XMax,YMin,YMax,IsGroupOf
    print("Open Images - implementacja w toku...")
    return []


def split_data(
    data: list[dict],
    split_ratio: tuple[float, float, float],
    seed: int,
) -> dict[str, list[dict]]:
    """
    Dzieli dane na train/val/test.

    Args:
        data: Lista danych do podziału
        split_ratio: Proporcje (train, val, test)
        seed: Seed dla losowości

    Returns:
        Słownik z kluczami 'train', 'val', 'test'
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(data))

    train_end = int(len(data) * split_ratio[0])
    val_end = train_end + int(len(data) * split_ratio[1])

    return {
        "train": [data[i] for i in indices[:train_end]],
        "val": [data[i] for i in indices[train_end:val_end]],
        "test": [data[i] for i in indices[val_end:]],
    }


def save_yolo_dataset(
    splits: dict[str, list[dict]],
    output_dir: Path,
) -> dict[str, int]:
    """
    Zapisuje dataset w formacie YOLO.

    Args:
        splits: Słownik z podziałem danych
        output_dir: Katalog wyjściowy

    Returns:
        Słownik z liczbą obrazów w każdym zbiorze
    """
    counts = {}

    for split_name, data in splits.items():
        images_dir = output_dir / "images" / split_name
        labels_dir = output_dir / "labels" / split_name

        count = 0
        for item in tqdm(data, desc=f"Zapisywanie {split_name}"):
            source_path = item["source_path"]
            if not source_path.exists():
                continue

            # Unikalna nazwa pliku
            new_name = f"{item['source']}_{count:06d}"

            # Kopiuj obraz
            img_ext = source_path.suffix.lower()
            if img_ext not in [".jpg", ".jpeg", ".png"]:
                img_ext = ".jpg"

            dst_img = images_dir / f"{new_name}{img_ext}"
            shutil.copy2(source_path, dst_img)

            # Zapisz etykiety w formacie YOLO
            label_lines = []
            for bbox in item["bboxes"]:
                yolo_bbox = convert_bbox_to_yolo(
                    bbox, item["width"], item["height"]
                )
                # class_id x_center y_center width height
                label_lines.append(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                                   f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")

            dst_label = labels_dir / f"{new_name}.txt"
            dst_label.write_text("\n".join(label_lines))

            count += 1

        counts[split_name] = count

    return counts


def create_dataset_yaml(output_dir: Path, counts: dict[str, int]) -> None:
    """
    Tworzy plik dataset.yaml dla YOLOv8.

    Args:
        output_dir: Katalog datasetu
        counts: Liczba obrazów w każdym zbiorze
    """
    yaml_content = f"""# Dog Detection Dataset
# Wygenerowano automatycznie przez prepare_bbox_data.py

path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Klasy
names:
  0: dog

# Statystyki
# Train: {counts.get('train', 0)} obrazów
# Val: {counts.get('val', 0)} obrazów
# Test: {counts.get('test', 0)} obrazów
# Total: {sum(counts.values())} obrazów
"""

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Utworzono {yaml_path}")


def create_dataset_stats(
    splits: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """
    Tworzy plik ze statystykami datasetu.

    Args:
        splits: Dane podzielone na zbiory
        output_dir: Katalog wyjściowy
    """
    stats = {
        "total_images": sum(len(s) for s in splits.values()),
        "splits": {k: len(v) for k, v in splits.items()},
        "sources": {},
        "bbox_stats": {
            "total_bboxes": 0,
            "avg_bboxes_per_image": 0.0,
            "images_with_multiple_dogs": 0,
        },
    }

    # Zlicz źródła i bboxes
    all_data = [item for split in splits.values() for item in split]
    for item in all_data:
        source = item.get("source", "unknown")
        stats["sources"][source] = stats["sources"].get(source, 0) + 1

        num_bboxes = len(item.get("bboxes", []))
        stats["bbox_stats"]["total_bboxes"] += num_bboxes
        if num_bboxes > 1:
            stats["bbox_stats"]["images_with_multiple_dogs"] += 1

    if stats["total_images"] > 0:
        stats["bbox_stats"]["avg_bboxes_per_image"] = (
            stats["bbox_stats"]["total_bboxes"] / stats["total_images"]
        )

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Utworzono {stats_path}")


def main() -> None:
    """Główna funkcja skryptu."""
    args = parse_args()

    # Parsuj proporcje podziału
    split_ratio = tuple(map(float, args.split_ratio.split(",")))
    if abs(sum(split_ratio) - 1.0) > 0.001:
        raise ValueError(f"Proporcje muszą sumować się do 1.0, otrzymano: {split_ratio}")

    print("=" * 60)
    print("Przygotowanie danych dla modelu detekcji psów (YOLOv8)")
    print("=" * 60)
    print(f"Źródło: {args.source}")
    print(f"Podział: train={split_ratio[0]}, val={split_ratio[1]}, test={split_ratio[2]}")
    print(f"Katalog wyjściowy: {args.output_dir}")
    print()

    # Utwórz strukturę katalogów
    setup_directories(args.output_dir)

    # Zbierz dane ze źródeł
    all_data = []

    if args.source in ["stanford", "all"]:
        stanford_data = process_stanford_dogs(RAW_DIR, args.output_dir, args.limit)
        all_data.extend(stanford_data)

    if args.source in ["openimages", "all"]:
        oi_data = process_open_images(RAW_DIR, args.output_dir, args.limit)
        all_data.extend(oi_data)

    if not all_data:
        print("Nie znaleziono żadnych danych do przetworzenia!")
        print("\nAby pobrać Stanford Dogs:")
        print("  kaggle datasets download jessicali9530/stanford-dogs-dataset")
        print("  unzip stanford-dogs-dataset.zip -d data/raw/stanford_dogs/")
        return

    print(f"\nZebrano łącznie {len(all_data)} obrazów")

    # Podziel dane
    splits = split_data(all_data, split_ratio, args.seed)

    # Zapisz w formacie YOLO
    counts = save_yolo_dataset(splits, args.output_dir)

    # Utwórz pliki konfiguracyjne
    create_dataset_yaml(args.output_dir, counts)
    create_dataset_stats(splits, args.output_dir)

    print("\n" + "=" * 60)
    print("Podsumowanie")
    print("=" * 60)
    print(f"Train: {counts.get('train', 0)} obrazów")
    print(f"Val: {counts.get('val', 0)} obrazów")
    print(f"Test: {counts.get('test', 0)} obrazów")
    print(f"Total: {sum(counts.values())} obrazów")
    print(f"\nDataset gotowy w: {args.output_dir}")
    print(f"Użyj dataset.yaml do treningu YOLOv8")


if __name__ == "__main__":
    main()
