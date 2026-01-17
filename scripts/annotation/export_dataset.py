#!/usr/bin/env python3
"""
Skrypt do eksportu finalnego datasetu Dog FACS.

Funkcje:
- Organizacja struktury katalogów
- Generowanie finalnego annotations.json
- Podział na train/val/test
- Generowanie statystyk i README
- Pakowanie do archiwum

Użycie:
    python scripts/annotation/export_dataset.py --input data/annotations/merged.json --output-dir data/final
    python scripts/annotation/export_dataset.py --split 0.8 0.1 0.1  # train/val/test
    python scripts/annotation/export_dataset.py --package  # utwórz archiwum
"""

import argparse
import hashlib
import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetStatistics:
    """Statystyki datasetu."""

    total_images: int = 0
    total_annotations: int = 0
    by_emotion: dict = field(default_factory=dict)
    by_breed: dict = field(default_factory=dict)
    by_split: dict = field(default_factory=dict)
    avg_annotations_per_image: float = 0.0
    total_size_mb: float = 0.0

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "total_images": self.total_images,
            "total_annotations": self.total_annotations,
            "avg_annotations_per_image": round(self.avg_annotations_per_image, 2),
            "by_emotion": self.by_emotion,
            "by_breed": dict(list(self.by_breed.items())[:20]),
            "by_split": self.by_split,
            "total_size_mb": round(self.total_size_mb, 2),
        }


class DatasetExporter:
    """
    Klasa do eksportu finalnego datasetu.

    Użycie:
        exporter = DatasetExporter(config)
        exporter.export(input_path, output_dir)
    """

    def __init__(
        self,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        random_seed: int = 42,
    ) -> None:
        """
        Inicjalizuje eksporter.

        Args:
            split_ratios: Proporcje train/val/test
            random_seed: Seed dla losowości
        """
        self.split_ratios = split_ratios
        self.random_seed = random_seed
        self.stats = DatasetStatistics()

        random.seed(random_seed)

    def load_coco(self, path: Path) -> dict:
        """Wczytuje plik COCO."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def calculate_statistics(
        self,
        data: dict,
        frames_dir: Optional[Path] = None,
    ) -> DatasetStatistics:
        """
        Oblicza statystyki datasetu.

        Args:
            data: Dane COCO
            frames_dir: Katalog z klatkami (dla rozmiaru)

        Returns:
            Statystyki
        """
        images = data.get("images", [])
        annotations = data.get("annotations", [])

        self.stats.total_images = len(images)
        self.stats.total_annotations = len(annotations)

        if self.stats.total_images > 0:
            self.stats.avg_annotations_per_image = (
                self.stats.total_annotations / self.stats.total_images
            )

        # Per emocja
        for ann in annotations:
            emotion = ann.get("emotion", {})
            if isinstance(emotion, dict):
                emotion_name = emotion.get("name", "unknown")
            else:
                emotion_name = str(emotion) if emotion else "unknown"

            if emotion_name not in self.stats.by_emotion:
                self.stats.by_emotion[emotion_name] = 0
            self.stats.by_emotion[emotion_name] += 1

        # Per rasa
        for ann in annotations:
            breed = ann.get("breed", {})
            if isinstance(breed, dict):
                breed_name = breed.get("name", "unknown")
            else:
                breed_name = str(breed) if breed else "unknown"

            if breed_name not in self.stats.by_breed:
                self.stats.by_breed[breed_name] = 0
            self.stats.by_breed[breed_name] += 1

        # Sortuj rasy
        self.stats.by_breed = dict(
            sorted(self.stats.by_breed.items(), key=lambda x: x[1], reverse=True)
        )

        # Rozmiar plików
        if frames_dir and frames_dir.exists():
            total_size = sum(
                f.stat().st_size for f in frames_dir.rglob("*") if f.is_file()
            )
            self.stats.total_size_mb = total_size / (1024 * 1024)

        return self.stats

    def split_dataset(
        self,
        data: dict,
    ) -> dict[str, dict]:
        """
        Dzieli dataset na train/val/test.

        Args:
            data: Dane COCO

        Returns:
            Słownik {split_name -> coco_data}
        """
        images = data.get("images", [])
        annotations = data.get("annotations", [])

        # Mapowanie anotacji per obraz
        ann_by_image = defaultdict(list)
        for ann in annotations:
            ann_by_image[ann.get("image_id")].append(ann)

        # Losowa kolejność obrazów
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)

        # Oblicz granice
        n = len(shuffled_images)
        train_end = int(n * self.split_ratios[0])
        val_end = train_end + int(n * self.split_ratios[1])

        splits = {
            "train": shuffled_images[:train_end],
            "val": shuffled_images[train_end:val_end],
            "test": shuffled_images[val_end:],
        }

        # Utwórz dane COCO dla każdego splitu
        result = {}

        for split_name, split_images in splits.items():
            split_image_ids = {img["id"] for img in split_images}

            split_annotations = [
                ann for ann in annotations
                if ann.get("image_id") in split_image_ids
            ]

            result[split_name] = {
                "info": data.get("info", {}),
                "licenses": data.get("licenses", []),
                "categories": data.get("categories", []),
                "images": split_images,
                "annotations": split_annotations,
            }

            self.stats.by_split[split_name] = {
                "images": len(split_images),
                "annotations": len(split_annotations),
            }

            logger.info(f"Split {split_name}: {len(split_images)} obrazów, {len(split_annotations)} anotacji")

        return result

    def copy_images(
        self,
        data: dict,
        source_dir: Path,
        target_dir: Path,
        split_name: str = "",
    ) -> int:
        """
        Kopiuje obrazy do katalogu docelowego.

        Args:
            data: Dane COCO
            source_dir: Katalog źródłowy
            target_dir: Katalog docelowy
            split_name: Nazwa splitu (opcjonalnie)

        Returns:
            Liczba skopiowanych plików
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        copied = 0

        for img in data.get("images", []):
            file_name = img.get("file_name", "")
            source_path = source_dir / file_name

            if source_path.exists():
                # Zachowaj strukturę katalogów lub spłaszcz
                if split_name:
                    target_path = target_dir / split_name / Path(file_name).name
                else:
                    target_path = target_dir / Path(file_name).name

                target_path.parent.mkdir(parents=True, exist_ok=True)

                if not target_path.exists():
                    shutil.copy2(source_path, target_path)
                    copied += 1

        logger.info(f"Skopiowano {copied} obrazów do {target_dir}")
        return copied

    def generate_readme(self, output_path: Path) -> None:
        """
        Generuje README dla datasetu.

        Args:
            output_path: Ścieżka wyjściowa
        """
        s = self.stats

        content = f"""# Dog FACS Dataset

## Opis

Dataset zawierający anotacje emocji psów w formacie COCO. Stworzony jako część projektu grupowego na Politechnice Gdańskiej (WETI).

## Statystyki

- **Łącznie obrazów:** {s.total_images}
- **Łącznie anotacji:** {s.total_annotations}
- **Średnio anotacji/obraz:** {s.avg_annotations_per_image:.2f}
- **Rozmiar:** ~{s.total_size_mb:.0f} MB

### Podział danych

| Split | Obrazy | Anotacje |
|-------|--------|----------|
"""
        for split_name, split_data in s.by_split.items():
            content += f"| {split_name} | {split_data['images']} | {split_data['annotations']} |\n"

        content += f"""
### Rozkład emocji

| Emocja | Liczba |
|--------|--------|
"""
        for emotion, count in sorted(s.by_emotion.items()):
            content += f"| {emotion} | {count} |\n"

        content += f"""
### Top 10 ras

| Rasa | Liczba |
|------|--------|
"""
        for breed, count in list(s.by_breed.items())[:10]:
            content += f"| {breed} | {count} |\n"

        content += f"""
## Struktura

```
dog-facs-dataset/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── README.md
└── statistics.json
```

## Format anotacji

Dataset używa formatu COCO z dodatkowymi polami:

```json
{{
  "id": 1,
  "image_id": 100,
  "category_id": 1,
  "bbox": [x, y, width, height],
  "area": 12345,
  "iscrowd": 0,
  "breed": {{
    "id": 5,
    "name": "Labrador Retriever",
    "confidence": 0.95
  }},
  "emotion": {{
    "id": 1,
    "name": "happy",
    "confidence": 0.87
  }},
  "keypoints": [x1, y1, v1, x2, y2, v2, ...],
  "num_keypoints": 46
}}
```

## Kategorie emocji

1. **happy** - Szczęśliwy (machanie ogonem, "uśmiech")
2. **sad** - Smutny (opuszczone uszy, przygnębiony wyraz)
3. **angry** - Zły/Agresywny (warczenie, pokazywanie zębów)
4. **relaxed** - Zrelaksowany (spokojne ciało, odpoczynek)
5. **fearful** - Przestraszony (ogon między nogami, chowanie się)
6. **neutral** - Neutralny (brak wyraźnych cech emocjonalnych)

## Licencja

Dataset stworzony do celów edukacyjnych i badawczych.

## Autorzy

Projekt grupowy - Politechnika Gdańska, WETI, 2025/2026

---

*Wygenerowano: {datetime.now().strftime('%Y-%m-%d')}*
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Zapisano README do {output_path}")

    def export(
        self,
        input_path: Path,
        output_dir: Path,
        frames_dir: Optional[Path] = None,
        copy_images: bool = False,
    ) -> Path:
        """
        Eksportuje finalny dataset.

        Args:
            input_path: Ścieżka do scalonych anotacji
            output_dir: Katalog wyjściowy
            frames_dir: Katalog z klatkami
            copy_images: Czy kopiować obrazy

        Returns:
            Ścieżka do katalogu wyjściowego
        """
        # Wczytaj dane
        data = self.load_coco(input_path)

        # Oblicz statystyki
        self.calculate_statistics(data, frames_dir)

        # Utwórz strukturę katalogów
        output_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir = output_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)

        # Podziel na splity
        splits = self.split_dataset(data)

        # Zapisz anotacje per split
        for split_name, split_data in splits.items():
            split_path = annotations_dir / f"{split_name}.json"

            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Zapisano {split_path}")

            # Kopiuj obrazy
            if copy_images and frames_dir:
                images_dir = output_dir / "images"
                self.copy_images(split_data, frames_dir, images_dir, split_name)

        # Zapisz statystyki
        stats_path = output_dir / "statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated": datetime.now().isoformat(),
                "statistics": self.stats.to_dict(),
            }, f, indent=2, ensure_ascii=False)

        # Generuj README
        readme_path = output_dir / "README.md"
        self.generate_readme(readme_path)

        logger.info(f"Eksport zakończony: {output_dir}")

        return output_dir

    def create_package(
        self,
        dataset_dir: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Tworzy archiwum datasetu.

        Args:
            dataset_dir: Katalog datasetu
            output_path: Ścieżka wyjściowa

        Returns:
            Ścieżka do archiwum
        """
        if output_path is None:
            output_path = dataset_dir.parent / f"dog-facs-dataset-{datetime.now().strftime('%Y%m%d')}"

        # Utwórz archiwum zip
        archive_path = shutil.make_archive(
            str(output_path),
            "zip",
            dataset_dir.parent,
            dataset_dir.name,
        )

        # Oblicz checksum
        with open(archive_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Zapisz checksum
        checksum_path = Path(archive_path).with_suffix(".sha256")
        with open(checksum_path, "w") as f:
            f.write(f"{checksum}  {Path(archive_path).name}\n")

        logger.info(f"Utworzono archiwum: {archive_path}")
        logger.info(f"SHA256: {checksum}")

        return Path(archive_path)


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Eksport finalnego datasetu Dog FACS"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/annotations/merged.json"),
        help="Ścieżka do scalonych anotacji",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/final/dog-facs-dataset"),
        help="Katalog wyjściowy",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/frames"),
        help="Katalog z klatkami",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Proporcje podziału (domyślnie: 0.8 0.1 0.1)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Kopiuj obrazy do katalogu wyjściowego",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="Utwórz archiwum ZIP",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (domyślnie: 42)",
    )

    args = parser.parse_args()

    # Eksporter
    exporter = DatasetExporter(
        split_ratios=tuple(args.split),
        random_seed=args.seed,
    )

    # Eksportuj
    output_dir = exporter.export(
        args.input,
        args.output_dir,
        args.frames_dir,
        args.copy_images,
    )

    # Pakowanie
    if args.package:
        exporter.create_package(output_dir)

    print(f"\n=== EKSPORT ZAKOŃCZONY ===")
    print(f"Katalog: {output_dir}")
    print(f"Obrazy:  {exporter.stats.total_images}")
    print(f"Anotacje: {exporter.stats.total_annotations}")


if __name__ == "__main__":
    main()
