#!/usr/bin/env python3
"""
Skrypt do stratyfikowanego wyboru próbek do weryfikacji.

Funkcje:
- Stratyfikowany sampling po emocji
- Stratyfikowany sampling po rasie
- Proporcjonalny rozkład próbek
- Eksport listy ID do weryfikacji

Użycie:
    python scripts/verification/sample_selector.py --annotations data/annotations/annotations.json --sample-size 6250
    python scripts/verification/sample_selector.py --percent 25
"""

import argparse
import json
import logging
import random
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
class SamplingConfig:
    """Konfiguracja samplowania."""

    sample_size: Optional[int] = None
    sample_percent: float = 25.0
    stratify_by_emotion: bool = True
    stratify_by_breed: bool = True
    min_per_stratum: int = 10
    random_seed: int = 42


@dataclass
class SampleDistribution:
    """Rozkład próbki."""

    total_images: int = 0
    sample_size: int = 0
    by_emotion: dict = field(default_factory=dict)
    by_breed: dict = field(default_factory=dict)
    image_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "total_images": self.total_images,
            "sample_size": self.sample_size,
            "sample_percent": round(self.sample_size / max(self.total_images, 1) * 100, 2),
            "by_emotion": self.by_emotion,
            "by_breed": dict(list(self.by_breed.items())[:20]),  # Top 20
        }


class SampleSelector:
    """
    Klasa do stratyfikowanego wyboru próbek.

    Użycie:
        selector = SampleSelector(config)
        distribution = selector.select_samples(annotations_path)
        selector.export_sample_ids(distribution, output_path)
    """

    def __init__(self, config: SamplingConfig) -> None:
        """
        Inicjalizuje selector.

        Args:
            config: Konfiguracja samplowania
        """
        self.config = config
        random.seed(config.random_seed)

    def load_annotations(self, annotations_path: Path) -> tuple[list[dict], list[dict]]:
        """
        Wczytuje anotacje COCO.

        Args:
            annotations_path: Ścieżka do pliku

        Returns:
            Tuple (images, annotations)
        """
        with open(annotations_path, encoding="utf-8") as f:
            data = json.load(f)

        return data.get("images", []), data.get("annotations", [])

    def _get_dominant_attribute(
        self,
        annotations: list[dict],
        attribute: str,
    ) -> str:
        """
        Pobiera dominujący atrybut z anotacji obrazu.

        Args:
            annotations: Lista anotacji dla obrazu
            attribute: Nazwa atrybutu (emotion, breed)

        Returns:
            Wartość dominującego atrybutu
        """
        if not annotations:
            return "unknown"

        # Zlicz wystąpienia
        counts = defaultdict(int)

        for ann in annotations:
            value = ann.get(attribute, {})
            if isinstance(value, dict):
                name = value.get("name", "unknown")
            else:
                name = str(value) if value else "unknown"
            counts[name] += 1

        # Zwróć najczęstszy
        if counts:
            return max(counts.keys(), key=lambda x: counts[x])
        return "unknown"

    def _stratify_images(
        self,
        images: list[dict],
        annotations: list[dict],
    ) -> dict[str, list[int]]:
        """
        Stratyfikuje obrazy według emocji.

        Args:
            images: Lista obrazów
            annotations: Lista anotacji

        Returns:
            Słownik {stratum -> [image_ids]}
        """
        # Grupuj anotacje po image_id
        ann_by_image = defaultdict(list)
        for ann in annotations:
            ann_by_image[ann.get("image_id")].append(ann)

        # Grupuj obrazy po emocji
        strata = defaultdict(list)

        for img in images:
            image_id = img["id"]
            img_annotations = ann_by_image.get(image_id, [])

            # Dominująca emocja
            emotion = self._get_dominant_attribute(img_annotations, "emotion")
            strata[emotion].append(image_id)

        return dict(strata)

    def _calculate_sample_sizes(
        self,
        strata: dict[str, list[int]],
        total_sample_size: int,
    ) -> dict[str, int]:
        """
        Oblicza rozmiary próbek per stratum.

        Args:
            strata: Słownik {stratum -> [image_ids]}
            total_sample_size: Całkowity rozmiar próbki

        Returns:
            Słownik {stratum -> sample_size}
        """
        total_images = sum(len(ids) for ids in strata.values())

        if total_images == 0:
            return {}

        sample_sizes = {}
        remaining = total_sample_size

        # Proporcjonalny podział
        for stratum, ids in strata.items():
            proportion = len(ids) / total_images
            size = int(total_sample_size * proportion)

            # Minimum per stratum
            size = max(size, min(self.config.min_per_stratum, len(ids)))
            size = min(size, len(ids))  # Nie więcej niż dostępne

            sample_sizes[stratum] = size
            remaining -= size

        # Rozdziel pozostałe
        if remaining > 0:
            sorted_strata = sorted(strata.keys(), key=lambda x: len(strata[x]), reverse=True)

            for stratum in sorted_strata:
                if remaining <= 0:
                    break

                available = len(strata[stratum]) - sample_sizes[stratum]
                add = min(remaining, available)
                sample_sizes[stratum] += add
                remaining -= add

        return sample_sizes

    def select_samples(
        self,
        annotations_path: Path,
    ) -> SampleDistribution:
        """
        Wybiera stratyfikowaną próbkę.

        Args:
            annotations_path: Ścieżka do pliku anotacji

        Returns:
            Rozkład próbki
        """
        logger.info(f"Wczytywanie anotacji z {annotations_path}")
        images, annotations = self.load_annotations(annotations_path)

        distribution = SampleDistribution(total_images=len(images))

        # Oblicz rozmiar próbki
        if self.config.sample_size:
            total_sample_size = min(self.config.sample_size, len(images))
        else:
            total_sample_size = int(len(images) * self.config.sample_percent / 100)

        distribution.sample_size = total_sample_size

        logger.info(f"Łącznie obrazów: {len(images)}, próbka: {total_sample_size}")

        # Stratyfikacja po emocji
        if self.config.stratify_by_emotion:
            strata = self._stratify_images(images, annotations)
            sample_sizes = self._calculate_sample_sizes(strata, total_sample_size)

            logger.info(f"Strata emocji: {list(strata.keys())}")

            # Wybierz próbki z każdego stratum
            selected_ids = []

            for stratum, ids in strata.items():
                size = sample_sizes.get(stratum, 0)
                sampled = random.sample(ids, min(size, len(ids)))
                selected_ids.extend(sampled)

                distribution.by_emotion[stratum] = {
                    "total": len(ids),
                    "sampled": len(sampled),
                    "percent": round(len(sampled) / max(len(ids), 1) * 100, 1),
                }

            distribution.image_ids = selected_ids

        else:
            # Losowa próbka bez stratyfikacji
            all_ids = [img["id"] for img in images]
            distribution.image_ids = random.sample(all_ids, total_sample_size)

        # Statystyki ras (dla informacji)
        if self.config.stratify_by_breed:
            ann_by_image = defaultdict(list)
            for ann in annotations:
                ann_by_image[ann.get("image_id")].append(ann)

            breed_counts = defaultdict(int)

            for image_id in distribution.image_ids:
                img_anns = ann_by_image.get(image_id, [])
                breed = self._get_dominant_attribute(img_anns, "breed")
                breed_counts[breed] += 1

            distribution.by_breed = dict(
                sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)
            )

        logger.info(f"Wybrano {len(distribution.image_ids)} obrazów")

        return distribution

    def export_sample_ids(
        self,
        distribution: SampleDistribution,
        output_path: Path,
    ) -> None:
        """
        Eksportuje listę ID próbki.

        Args:
            distribution: Rozkład próbki
            output_path: Ścieżka wyjściowa
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "generated": datetime.now().isoformat(),
            "config": {
                "sample_percent": self.config.sample_percent,
                "stratify_by_emotion": self.config.stratify_by_emotion,
                "stratify_by_breed": self.config.stratify_by_breed,
                "random_seed": self.config.random_seed,
            },
            "distribution": distribution.to_dict(),
            "image_ids": distribution.image_ids,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Zapisano sample IDs do {output_path}")

    def generate_report(self, distribution: SampleDistribution) -> str:
        """
        Generuje raport z samplowania.

        Args:
            distribution: Rozkład próbki

        Returns:
            Raport jako string
        """
        lines = [
            "=" * 60,
            "RAPORT STRATYFIKOWANEGO SAMPLOWANIA",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "OGÓLNE STATYSTYKI",
            "-" * 40,
            f"Łącznie obrazów:       {distribution.total_images}",
            f"Rozmiar próbki:        {distribution.sample_size}",
            f"Procent:               {distribution.sample_size / max(distribution.total_images, 1) * 100:.1f}%",
            "",
        ]

        # Emocje
        if distribution.by_emotion:
            lines.extend([
                "ROZKŁAD PO EMOCJI",
                "-" * 40,
            ])
            for emotion, data in sorted(distribution.by_emotion.items()):
                lines.append(
                    f"  {emotion:12s}: {data['sampled']:5d} / {data['total']:5d} ({data['percent']:5.1f}%)"
                )
            lines.append("")

        # Rasy
        if distribution.by_breed:
            lines.extend([
                "TOP 10 RAS W PRÓBCE",
                "-" * 40,
            ])
            for breed, count in list(distribution.by_breed.items())[:10]:
                percent = count / max(distribution.sample_size, 1) * 100
                lines.append(f"  {breed[:20]:20s}: {count:5d} ({percent:5.1f}%)")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Stratyfikowany wybór próbek do weryfikacji"
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/annotations.json"),
        help="Ścieżka do pliku anotacji COCO",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/verification/sample_ids.json"),
        help="Ścieżka wyjściowa",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Rozmiar próbki (liczba obrazów)",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=25.0,
        help="Procent próbki (domyślnie: 25)",
    )
    parser.add_argument(
        "--no-stratify-emotion",
        action="store_true",
        help="Wyłącz stratyfikację po emocji",
    )
    parser.add_argument(
        "--no-stratify-breed",
        action="store_true",
        help="Wyłącz statystyki ras",
    )
    parser.add_argument(
        "--min-per-stratum",
        type=int,
        default=10,
        help="Minimum próbek per stratum (domyślnie: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (domyślnie: 42)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Pokaż raport",
    )

    args = parser.parse_args()

    # Konfiguracja
    config = SamplingConfig(
        sample_size=args.sample_size,
        sample_percent=args.percent,
        stratify_by_emotion=not args.no_stratify_emotion,
        stratify_by_breed=not args.no_stratify_breed,
        min_per_stratum=args.min_per_stratum,
        random_seed=args.seed,
    )

    # Selector
    selector = SampleSelector(config)

    # Wybierz próbki
    distribution = selector.select_samples(args.annotations)

    # Eksportuj
    selector.export_sample_ids(distribution, args.output)

    # Raport
    if args.report:
        print(selector.generate_report(distribution))
    else:
        print(f"\n=== WYNIK ===")
        print(f"Wybrano {distribution.sample_size} obrazów z {distribution.total_images}")
        print(f"Zapisano do: {args.output}")


if __name__ == "__main__":
    main()
