#!/usr/bin/env python3
"""
Skrypt do przygotowania danych treningowych dla klasyfikacji ras psów.

Obsługuje Stanford Dogs Dataset i przygotowuje dane w formacie ImageFolder.

Użycie:
    python scripts/training/prepare_breed_data.py
    python scripts/training/prepare_breed_data.py --min-samples 50 --max-breeds 120
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm


# Ścieżki domyślne
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
BREED_TRAINING_DIR = DATA_DIR / "breed_training"


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Przygotowanie danych dla klasyfikacji ras psów"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR / "stanford_dogs",
        help="Katalog z pobranym Stanford Dogs Dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BREED_TRAINING_DIR,
        help=f"Katalog wyjściowy (default: {BREED_TRAINING_DIR})",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimalna liczba obrazów na rasę (default: 100)",
    )
    parser.add_argument(
        "--max-breeds",
        type=int,
        default=120,
        help="Maksymalna liczba ras (default: 120 - wszystkie)",
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
    return parser.parse_args()


def parse_breed_name(folder_name: str) -> str:
    """
    Parsuje nazwę rasy z nazwy folderu Stanford Dogs.

    Args:
        folder_name: Nazwa folderu np. 'n02085620-Chihuahua'

    Returns:
        Czytelna nazwa rasy np. 'Chihuahua'
    """
    # Format: n02085620-Chihuahua lub n02085620_Chihuahua
    if "-" in folder_name:
        breed = folder_name.split("-", 1)[1]
    elif "_" in folder_name:
        parts = folder_name.split("_")
        if parts[0].startswith("n"):
            breed = "_".join(parts[1:])
        else:
            breed = folder_name
    else:
        breed = folder_name

    # Zamień podkreślenia na spacje
    breed = breed.replace("_", " ")

    return breed


def collect_breeds(images_dir: Path) -> dict[str, list[Path]]:
    """
    Zbiera wszystkie obrazy pogrupowane wg ras.

    Args:
        images_dir: Katalog z obrazami Stanford Dogs

    Returns:
        Słownik {nazwa_rasy: [ścieżki_do_obrazów]}
    """
    breeds = {}

    for breed_folder in sorted(images_dir.iterdir()):
        if not breed_folder.is_dir():
            continue

        breed_name = parse_breed_name(breed_folder.name)

        # Zbierz wszystkie obrazy
        images = list(breed_folder.glob("*.jpg")) + list(breed_folder.glob("*.JPEG"))

        if images:
            breeds[breed_name] = images

    return breeds


def filter_breeds(
    breeds: dict[str, list[Path]],
    min_samples: int,
    max_breeds: int,
) -> dict[str, list[Path]]:
    """
    Filtruje rasy wg minimalnej liczby próbek.

    Args:
        breeds: Słownik ras
        min_samples: Minimalna liczba obrazów
        max_breeds: Maksymalna liczba ras

    Returns:
        Przefiltrowany słownik ras
    """
    # Filtruj wg min_samples
    filtered = {
        breed: images
        for breed, images in breeds.items()
        if len(images) >= min_samples
    }

    # Sortuj wg liczby obrazów (malejąco) i ogranicz do max_breeds
    sorted_breeds = sorted(
        filtered.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )[:max_breeds]

    return dict(sorted_breeds)


def create_breed_mapping(breeds: dict[str, list[Path]]) -> dict[int, str]:
    """
    Tworzy mapowanie ID -> nazwa rasy.

    Args:
        breeds: Słownik ras

    Returns:
        Słownik {id: nazwa_rasy}
    """
    sorted_breeds = sorted(breeds.keys())
    return {i: breed for i, breed in enumerate(sorted_breeds)}


def split_data(
    breeds: dict[str, list[Path]],
    split_ratio: tuple[float, float, float],
    seed: int,
) -> dict[str, dict[str, list[Path]]]:
    """
    Dzieli dane na train/val/test z zachowaniem proporcji klas.

    Args:
        breeds: Słownik ras
        split_ratio: Proporcje (train, val, test)
        seed: Seed dla losowości

    Returns:
        Słownik {split: {breed: [images]}}
    """
    np.random.seed(seed)

    splits = {
        "train": {},
        "val": {},
        "test": {},
    }

    for breed, images in breeds.items():
        # Losowo przetasuj
        indices = np.random.permutation(len(images))

        # Oblicz granice podziału
        train_end = int(len(images) * split_ratio[0])
        val_end = train_end + int(len(images) * split_ratio[1])

        # Podziel
        splits["train"][breed] = [images[i] for i in indices[:train_end]]
        splits["val"][breed] = [images[i] for i in indices[train_end:val_end]]
        splits["test"][breed] = [images[i] for i in indices[val_end:]]

    return splits


def save_dataset(
    splits: dict[str, dict[str, list[Path]]],
    breed_mapping: dict[int, str],
    output_dir: Path,
) -> dict[str, int]:
    """
    Zapisuje dataset w formacie ImageFolder.

    Args:
        splits: Dane podzielone na zbiory
        breed_mapping: Mapowanie ID -> nazwa
        output_dir: Katalog wyjściowy

    Returns:
        Słownik z liczbą obrazów w każdym zbiorze
    """
    # Odwróć mapowanie do name -> id
    name_to_id = {v: k for k, v in breed_mapping.items()}

    counts = {}

    for split_name, breeds in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for breed_name, images in tqdm(breeds.items(), desc=f"Saving {split_name}"):
            # Utwórz folder dla rasy (użyj ID i nazwy)
            breed_id = name_to_id[breed_name]
            safe_name = breed_name.replace(" ", "_").replace("/", "-")
            breed_folder = split_dir / f"{breed_id:03d}_{safe_name}"
            breed_folder.mkdir(exist_ok=True)

            # Kopiuj obrazy
            for i, img_path in enumerate(images):
                dst = breed_folder / f"{i:04d}{img_path.suffix.lower()}"
                shutil.copy2(img_path, dst)
                count += 1

        counts[split_name] = count

    return counts


def save_breeds_json(
    breed_mapping: dict[int, str],
    output_dir: Path,
) -> Path:
    """
    Zapisuje mapowanie ras do JSON.

    Args:
        breed_mapping: Mapowanie ID -> nazwa
        output_dir: Katalog wyjściowy

    Returns:
        Ścieżka do pliku JSON
    """
    json_path = output_dir / "breeds.json"

    # Konwertuj klucze na stringi dla JSON
    mapping = {str(k): v for k, v in breed_mapping.items()}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    return json_path


def save_dataset_stats(
    splits: dict[str, dict[str, list[Path]]],
    breed_mapping: dict[int, str],
    output_dir: Path,
) -> None:
    """
    Zapisuje statystyki datasetu.

    Args:
        splits: Dane podzielone na zbiory
        breed_mapping: Mapowanie ID -> nazwa
        output_dir: Katalog wyjściowy
    """
    stats = {
        "num_breeds": len(breed_mapping),
        "breeds": list(breed_mapping.values()),
        "splits": {},
        "per_breed": {},
    }

    for split_name, breeds in splits.items():
        total = sum(len(images) for images in breeds.values())
        stats["splits"][split_name] = {
            "total_images": total,
            "per_breed": {breed: len(images) for breed, images in breeds.items()},
        }

    # Łączna liczba na rasę
    for breed in breed_mapping.values():
        stats["per_breed"][breed] = sum(
            len(splits[split].get(breed, []))
            for split in splits
        )

    # Zapisz
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Zapisano statystyki: {stats_path}")


def main() -> None:
    """Główna funkcja skryptu."""
    args = parse_args()

    # Parsuj proporcje podziału
    split_ratio = tuple(map(float, args.split_ratio.split(",")))
    if abs(sum(split_ratio) - 1.0) > 0.001:
        raise ValueError(f"Proporcje muszą sumować się do 1.0: {split_ratio}")

    print("=" * 60)
    print("Przygotowanie danych dla klasyfikacji ras psów")
    print("=" * 60)
    print(f"Źródło: {args.raw_dir}")
    print(f"Wyjście: {args.output_dir}")
    print(f"Min. próbek/rasę: {args.min_samples}")
    print(f"Max. ras: {args.max_breeds}")
    print()

    # Sprawdź czy dane istnieją
    images_dir = args.raw_dir / "Images"
    if not images_dir.exists():
        print(f"Nie znaleziono katalogu z obrazami: {images_dir}")
        print("\nAby pobrać Stanford Dogs Dataset:")
        print("  kaggle datasets download jessicali9530/stanford-dogs-dataset")
        print("  unzip stanford-dogs-dataset.zip -d data/raw/stanford_dogs/")
        return

    # Zbierz rasy
    print("Zbieranie ras...")
    all_breeds = collect_breeds(images_dir)
    print(f"Znaleziono {len(all_breeds)} ras")

    # Filtruj
    print(f"Filtrowanie (min {args.min_samples} obrazów/rasę)...")
    filtered_breeds = filter_breeds(all_breeds, args.min_samples, args.max_breeds)
    print(f"Wybrano {len(filtered_breeds)} ras")

    if not filtered_breeds:
        print("Brak ras spełniających kryteria!")
        return

    # Statystyki ras
    total_images = sum(len(imgs) for imgs in filtered_breeds.values())
    print(f"Łącznie {total_images} obrazów")

    # Utwórz mapowanie
    breed_mapping = create_breed_mapping(filtered_breeds)

    # Podziel dane
    print("\nPodział danych...")
    splits = split_data(filtered_breeds, split_ratio, args.seed)

    # Zapisz
    print("\nZapisywanie datasetu...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = save_dataset(splits, breed_mapping, args.output_dir)

    # Zapisz mapowanie ras
    json_path = save_breeds_json(breed_mapping, args.output_dir)
    print(f"Zapisano mapowanie ras: {json_path}")

    # Skopiuj też do packages/models/
    models_breeds_path = Path("packages/models/breeds.json")
    models_breeds_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(json_path, models_breeds_path)
    print(f"Skopiowano do: {models_breeds_path}")

    # Zapisz statystyki
    save_dataset_stats(splits, breed_mapping, args.output_dir)

    # Podsumowanie
    print("\n" + "=" * 60)
    print("Podsumowanie")
    print("=" * 60)
    print(f"Liczba ras: {len(breed_mapping)}")
    print(f"Train: {counts['train']} obrazów")
    print(f"Val: {counts['val']} obrazów")
    print(f"Test: {counts['test']} obrazów")
    print(f"Total: {sum(counts.values())} obrazów")
    print(f"\nDataset gotowy w: {args.output_dir}")
    print("\nPrzykładowe rasy:")
    for i in range(min(5, len(breed_mapping))):
        print(f"  {i}: {breed_mapping[i]}")
    if len(breed_mapping) > 5:
        print(f"  ... i {len(breed_mapping) - 5} więcej")


if __name__ == "__main__":
    main()
