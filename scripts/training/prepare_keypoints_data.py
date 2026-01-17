#!/usr/bin/env python3
"""
Skrypt do przygotowania danych treningowych dla detekcji keypoints.

Obsługuje format DogFLW (Dog Facial Landmarks) i mapuje na naszą 20-punktową schemę.

Użycie:
    python scripts/training/prepare_keypoints_data.py
    python scripts/training/prepare_keypoints_data.py --raw-dir data/raw/dogflw
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


# Ścieżki domyślne
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
KEYPOINTS_TRAINING_DIR = DATA_DIR / "keypoints_training"

# Mapowanie DogFLW (8 punktów) na naszą schemę (20 punktów)
# DogFLW ma mniej punktów, więc niektóre trzeba interpolować
DOGFLW_KEYPOINTS = [
    "left_eye",
    "right_eye",
    "nose",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "chin",
]

# Mapowanie DogFLW → nasza schema (indeksy)
# Punkty których nie ma w DogFLW będą interpolowane lub oznaczone jako niewidoczne
DOGFLW_TO_SCHEMA_MAP = {
    0: 0,   # left_eye → LEFT_EYE
    1: 1,   # right_eye → RIGHT_EYE
    2: 2,   # nose → NOSE
    3: 3,   # left_ear → LEFT_EAR_BASE (przybliżenie)
    4: 4,   # right_ear → RIGHT_EAR_BASE (przybliżenie)
    5: 7,   # mouth_left → LEFT_MOUTH_CORNER
    6: 8,   # mouth_right → RIGHT_MOUTH_CORNER
    7: 11,  # chin → CHIN
}


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Przygotowanie danych dla detekcji keypoints"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR / "dogflw",
        help="Katalog z pobranym DogFLW Dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=KEYPOINTS_TRAINING_DIR,
        help=f"Katalog wyjściowy (default: {KEYPOINTS_TRAINING_DIR})",
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
        "--img-size",
        type=int,
        default=256,
        help="Rozmiar obrazu (default: 256)",
    )
    return parser.parse_args()


def load_dogflw_annotations(raw_dir: Path) -> list[dict]:
    """
    Wczytuje anotacje z DogFLW dataset.

    DogFLW format to zazwyczaj:
    - images/ folder z obrazami
    - annotations.json lub .csv z keypoints

    Args:
        raw_dir: Katalog z danymi DogFLW

    Returns:
        Lista anotacji
    """
    annotations = []

    # Sprawdź różne możliwe formaty
    json_files = list(raw_dir.glob("*.json"))
    csv_files = list(raw_dir.glob("*.csv"))

    if json_files:
        # Format JSON
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    annotations.extend(data)
                elif isinstance(data, dict) and "annotations" in data:
                    annotations.extend(data["annotations"])

    elif csv_files:
        # Format CSV
        import csv
        for csv_file in csv_files:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    annotations.append(row)

    else:
        # Próba wczytania z folderów struktury images + labels
        images_dir = raw_dir / "images"
        labels_dir = raw_dir / "labels"

        if images_dir.exists() and labels_dir.exists():
            for img_path in images_dir.glob("*.[jp][pn][g]*"):
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    with open(label_path) as f:
                        keypoints_str = f.read().strip().split()
                        keypoints = [float(x) for x in keypoints_str]
                        annotations.append({
                            "image": img_path.name,
                            "keypoints": keypoints,
                        })

    return annotations


def map_dogflw_to_schema(
    dogflw_keypoints: list[float],
    img_width: int,
    img_height: int,
) -> list[tuple[float, float, float]]:
    """
    Mapuje keypoints z DogFLW na naszą 20-punktową schemę.

    Args:
        dogflw_keypoints: Keypoints z DogFLW [x1, y1, x2, y2, ...]
        img_width: Szerokość obrazu
        img_height: Wysokość obrazu

    Returns:
        Lista 20 keypoints jako (x, y, visibility)
    """
    # Inicjalizuj wszystkie 20 punktów jako niewidoczne
    schema_keypoints = [(0.0, 0.0, 0.0) for _ in range(20)]

    # Parsuj DogFLW keypoints (x, y pary)
    num_dogflw_points = len(dogflw_keypoints) // 2
    dogflw_points = []
    for i in range(num_dogflw_points):
        x = dogflw_keypoints[i * 2]
        y = dogflw_keypoints[i * 2 + 1]
        # Normalizuj do [0, 1]
        if x > 1:
            x = x / img_width
        if y > 1:
            y = y / img_height
        dogflw_points.append((x, y))

    # Mapuj znane punkty
    for dogflw_idx, schema_idx in DOGFLW_TO_SCHEMA_MAP.items():
        if dogflw_idx < len(dogflw_points):
            x, y = dogflw_points[dogflw_idx]
            if 0 <= x <= 1 and 0 <= y <= 1:
                schema_keypoints[schema_idx] = (x, y, 1.0)

    # Interpoluj brakujące punkty
    schema_keypoints = interpolate_missing_keypoints(schema_keypoints, dogflw_points)

    return schema_keypoints


def interpolate_missing_keypoints(
    schema_keypoints: list[tuple[float, float, float]],
    dogflw_points: list[tuple[float, float]],
) -> list[tuple[float, float, float]]:
    """
    Interpoluje brakujące keypoints na podstawie dostępnych.

    Args:
        schema_keypoints: Aktualna lista keypoints
        dogflw_points: Oryginalne punkty DogFLW

    Returns:
        Zaktualizowana lista keypoints
    """
    result = list(schema_keypoints)

    # Pobierz dostępne punkty
    left_eye = schema_keypoints[0] if schema_keypoints[0][2] > 0 else None
    right_eye = schema_keypoints[1] if schema_keypoints[1][2] > 0 else None
    nose = schema_keypoints[2] if schema_keypoints[2][2] > 0 else None
    left_ear = schema_keypoints[3] if schema_keypoints[3][2] > 0 else None
    right_ear = schema_keypoints[4] if schema_keypoints[4][2] > 0 else None
    left_mouth = schema_keypoints[7] if schema_keypoints[7][2] > 0 else None
    right_mouth = schema_keypoints[8] if schema_keypoints[8][2] > 0 else None
    chin = schema_keypoints[11] if schema_keypoints[11][2] > 0 else None

    # Interpoluj ear tips (5, 6) - powyżej ear base
    if left_ear:
        result[5] = (left_ear[0], max(0, left_ear[1] - 0.1), 0.5)
    if right_ear:
        result[6] = (right_ear[0], max(0, right_ear[1] - 0.1), 0.5)

    # Interpoluj upper_lip (9) - między nosem a ustami
    if nose and left_mouth and right_mouth:
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        result[9] = (
            (nose[0] + mouth_center_x) / 2,
            (nose[1] + mouth_center_y) / 2,
            0.5,
        )

    # Interpoluj lower_lip (10) - między upper_lip a chin
    if result[9][2] > 0 and chin:
        result[10] = (
            (result[9][0] + chin[0]) / 2,
            (result[9][1] + chin[1]) / 2,
            0.5,
        )

    # Interpoluj cheeks (12, 13) - na zewnątrz od oczu
    if left_eye and left_mouth:
        result[12] = (
            left_eye[0] - 0.05,
            (left_eye[1] + left_mouth[1]) / 2,
            0.5,
        )
    if right_eye and right_mouth:
        result[13] = (
            right_eye[0] + 0.05,
            (right_eye[1] + right_mouth[1]) / 2,
            0.5,
        )

    # Interpoluj forehead (14) - powyżej oczu
    if left_eye and right_eye:
        result[14] = (
            (left_eye[0] + right_eye[0]) / 2,
            min(left_eye[1], right_eye[1]) - 0.1,
            0.5,
        )

    # Interpoluj eyebrows (15, 16) - nad oczami
    if left_eye:
        result[15] = (left_eye[0], left_eye[1] - 0.05, 0.5)
    if right_eye:
        result[16] = (right_eye[0], right_eye[1] - 0.05, 0.5)

    # Interpoluj muzzle points (17, 18, 19)
    if nose:
        result[17] = (nose[0], nose[1] - 0.03, 0.5)  # muzzle_top
        result[18] = (nose[0] - 0.05, nose[1], 0.5)  # muzzle_left
        result[19] = (nose[0] + 0.05, nose[1], 0.5)  # muzzle_right

    return result


def create_coco_annotations(
    annotations: list[dict],
    images_dir: Path,
    output_dir: Path,
    split_ratio: tuple[float, float, float],
    seed: int,
) -> dict[str, list]:
    """
    Tworzy anotacje w formacie COCO dla keypoints.

    Args:
        annotations: Lista anotacji
        images_dir: Katalog z obrazami
        output_dir: Katalog wyjściowy
        split_ratio: Proporcje train/val/test
        seed: Seed dla losowości

    Returns:
        Słownik z podziałem na zbiory
    """
    np.random.seed(seed)

    # Przemieszaj
    indices = np.random.permutation(len(annotations))

    # Oblicz granice
    train_end = int(len(annotations) * split_ratio[0])
    val_end = train_end + int(len(annotations) * split_ratio[1])

    splits = {
        "train": [annotations[i] for i in indices[:train_end]],
        "val": [annotations[i] for i in indices[train_end:val_end]],
        "test": [annotations[i] for i in indices[val_end:]],
    }

    return splits


def save_coco_format(
    split_data: list[dict],
    output_path: Path,
    images_dir: Path,
) -> None:
    """
    Zapisuje dane w formacie COCO keypoints.

    Args:
        split_data: Lista anotacji dla danego zbioru
        output_path: Ścieżka do pliku JSON
        images_dir: Katalog z obrazami
    """
    from packages.data.schemas import KEYPOINT_NAMES, SKELETON_CONNECTIONS

    coco = {
        "info": {
            "description": "Dog Facial Keypoints Dataset",
            "version": "1.0",
        },
        "categories": [{
            "id": 1,
            "name": "dog",
            "supercategory": "animal",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": [[c[0], c[1]] for c in SKELETON_CONNECTIONS],
        }],
        "images": [],
        "annotations": [],
    }

    for idx, ann in enumerate(split_data):
        # Dodaj obraz
        img_name = ann.get("image", ann.get("filename", ""))
        img_path = images_dir / img_name

        if img_path.exists():
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
        else:
            width, height = 256, 256

        coco["images"].append({
            "id": idx,
            "file_name": img_name,
            "width": width,
            "height": height,
        })

        # Dodaj anotację
        keypoints = ann.get("keypoints", [])
        if isinstance(keypoints, list) and len(keypoints) > 0:
            mapped_kps = map_dogflw_to_schema(keypoints, width, height)

            # Flatten do formatu COCO [x1, y1, v1, x2, y2, v2, ...]
            flat_keypoints = []
            for x, y, v in mapped_kps:
                flat_keypoints.extend([x * width, y * height, int(v * 2)])

            coco["annotations"].append({
                "id": idx,
                "image_id": idx,
                "category_id": 1,
                "keypoints": flat_keypoints,
                "num_keypoints": sum(1 for _, _, v in mapped_kps if v > 0),
            })

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)


def main() -> None:
    """Główna funkcja skryptu."""
    args = parse_args()

    # Parsuj proporcje
    split_ratio = tuple(map(float, args.split_ratio.split(",")))
    if abs(sum(split_ratio) - 1.0) > 0.001:
        raise ValueError(f"Proporcje muszą sumować się do 1.0: {split_ratio}")

    print("=" * 60)
    print("Przygotowanie danych dla detekcji keypoints")
    print("=" * 60)
    print(f"Źródło: {args.raw_dir}")
    print(f"Wyjście: {args.output_dir}")
    print()

    # Sprawdź dane
    if not args.raw_dir.exists():
        print(f"❌ Nie znaleziono katalogu: {args.raw_dir}")
        print("\nAby pobrać DogFLW Dataset:")
        print("  kaggle datasets download -d sophiepolice/dog-facial-landmark-localization")
        print("  unzip dog-facial-landmark-localization.zip -d data/raw/dogflw/")
        print("\nLub użyj innego datasetu z keypoints.")
        return

    # Wczytaj anotacje
    print("Wczytywanie anotacji...")
    annotations = load_dogflw_annotations(args.raw_dir)

    if not annotations:
        print("❌ Nie znaleziono anotacji w katalogu")
        print("Sprawdź strukturę danych w:", args.raw_dir)
        return

    print(f"✓ Znaleziono {len(annotations)} anotacji")

    # Znajdź katalog z obrazami
    images_dir = args.raw_dir / "images"
    if not images_dir.exists():
        images_dir = args.raw_dir

    # Podziel dane
    print("\nPodział danych...")
    splits = create_coco_annotations(
        annotations, images_dir, args.output_dir, split_ratio, args.seed
    )

    # Zapisz
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        output_path = args.output_dir / f"{split_name}.json"
        save_coco_format(split_data, output_path, images_dir)
        print(f"✓ {split_name}: {len(split_data)} obrazów → {output_path}")

    # Skopiuj obrazy (opcjonalnie)
    images_output = args.output_dir / "images"
    if not images_output.exists():
        print("\nKopiowanie obrazów...")
        images_output.mkdir(exist_ok=True)
        for img_file in tqdm(list(images_dir.glob("*.[jp][pn][g]*"))):
            shutil.copy2(img_file, images_output / img_file.name)

    # Podsumowanie
    print("\n" + "=" * 60)
    print("Podsumowanie")
    print("=" * 60)
    print(f"Train: {len(splits['train'])} obrazów")
    print(f"Val: {len(splits['val'])} obrazów")
    print(f"Test: {len(splits['test'])} obrazów")
    print(f"\nDane gotowe w: {args.output_dir}")


if __name__ == "__main__":
    main()
