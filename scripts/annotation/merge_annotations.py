#!/usr/bin/env python3
"""
Skrypt do scalania anotacji automatycznych z manualnymi korektami.

Funkcje:
- Aplikowanie korekt manualnych do anotacji automatycznych
- Zachowanie anotacji bez korekt
- Usuwanie duplikatów i odrzuconych anotacji
- Raport ze scalania

Użycie:
    python scripts/annotation/merge_annotations.py --auto data/annotations/annotations.json --corrections data/verification/corrections.json
    python scripts/annotation/merge_annotations.py --output data/annotations/merged.json
"""

import argparse
import json
import logging
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
class MergeStatistics:
    """Statystyki scalania."""

    total_auto_annotations: int = 0
    total_corrections: int = 0
    accepted: int = 0
    corrected: int = 0
    rejected: int = 0
    unchanged: int = 0
    duplicates_removed: int = 0
    orphans_removed: int = 0
    final_annotations: int = 0

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "total_auto_annotations": self.total_auto_annotations,
            "total_corrections": self.total_corrections,
            "accepted": self.accepted,
            "corrected": self.corrected,
            "rejected": self.rejected,
            "unchanged": self.unchanged,
            "duplicates_removed": self.duplicates_removed,
            "orphans_removed": self.orphans_removed,
            "final_annotations": self.final_annotations,
        }


class AnnotationMerger:
    """
    Klasa do scalania anotacji.

    Użycie:
        merger = AnnotationMerger()
        merged_data = merger.merge(auto_path, corrections_paths)
        merger.save(merged_data, output_path)
    """

    def __init__(self) -> None:
        """Inicjalizuje merger."""
        self.stats = MergeStatistics()

    def load_auto_annotations(self, path: Path) -> dict:
        """
        Wczytuje anotacje automatyczne.

        Args:
            path: Ścieżka do pliku COCO

        Returns:
            Dane COCO
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.stats.total_auto_annotations = len(data.get("annotations", []))
        logger.info(f"Wczytano {self.stats.total_auto_annotations} anotacji automatycznych")

        return data

    def load_corrections(self, paths: list[Path]) -> dict[int, dict]:
        """
        Wczytuje korekty z wielu plików.

        Args:
            paths: Lista ścieżek do plików korekt

        Returns:
            Słownik {annotation_id -> correction}
        """
        all_corrections = {}

        for path in paths:
            if not path.exists():
                logger.warning(f"Plik korekt nie istnieje: {path}")
                continue

            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            for corr in data.get("corrections", []):
                ann_id = corr.get("annotation_id")
                if ann_id is not None:
                    # Jeśli ten sam annotation_id jest w wielu plikach,
                    # użyj nowszej korekty
                    existing = all_corrections.get(ann_id)
                    if existing:
                        existing_time = existing.get("timestamp", "")
                        new_time = corr.get("timestamp", "")
                        if new_time > existing_time:
                            all_corrections[ann_id] = corr
                    else:
                        all_corrections[ann_id] = corr

        self.stats.total_corrections = len(all_corrections)
        logger.info(f"Wczytano {self.stats.total_corrections} korekt z {len(paths)} plików")

        return all_corrections

    def apply_correction(
        self,
        annotation: dict,
        correction: dict,
    ) -> Optional[dict]:
        """
        Aplikuje korektę do anotacji.

        Args:
            annotation: Oryginalna anotacja
            correction: Korekta

        Returns:
            Zmodyfikowana anotacja lub None (jeśli odrzucona)
        """
        corr_type = correction.get("type", "")
        corr_data = correction.get("corrected_data", {})

        if corr_type == "reject":
            self.stats.rejected += 1
            return None

        if corr_type == "accept":
            self.stats.accepted += 1
            # Oznacz jako zweryfikowane
            annotation["verified"] = True
            annotation["verified_by"] = correction.get("verified_by", "unknown")
            return annotation

        if corr_type == "correct":
            self.stats.corrected += 1

            # Aplikuj poprawki
            if "emotion" in corr_data:
                if isinstance(annotation.get("emotion"), dict):
                    annotation["emotion"]["name"] = corr_data["emotion"]
                    annotation["emotion"]["corrected"] = True
                else:
                    annotation["emotion"] = {
                        "name": corr_data["emotion"],
                        "corrected": True,
                    }

            if "breed" in corr_data:
                if isinstance(annotation.get("breed"), dict):
                    annotation["breed"]["name"] = corr_data["breed"]
                    annotation["breed"]["corrected"] = True
                else:
                    annotation["breed"] = {
                        "name": corr_data["breed"],
                        "corrected": True,
                    }

            if "bbox" in corr_data:
                annotation["bbox"] = corr_data["bbox"]

            annotation["verified"] = True
            annotation["verified_by"] = correction.get("verified_by", "unknown")
            annotation["correction_note"] = correction.get("note", "")

            return annotation

        # Nieznany typ - zachowaj bez zmian
        self.stats.unchanged += 1
        return annotation

    def remove_duplicates(self, annotations: list[dict]) -> list[dict]:
        """
        Usuwa duplikaty anotacji.

        Args:
            annotations: Lista anotacji

        Returns:
            Lista bez duplikatów
        """
        seen_ids = set()
        unique = []

        for ann in annotations:
            ann_id = ann.get("id")
            if ann_id not in seen_ids:
                seen_ids.add(ann_id)
                unique.append(ann)
            else:
                self.stats.duplicates_removed += 1

        return unique

    def remove_orphans(
        self,
        annotations: list[dict],
        valid_image_ids: set[int],
    ) -> list[dict]:
        """
        Usuwa anotacje bez odpowiadających obrazów.

        Args:
            annotations: Lista anotacji
            valid_image_ids: Zbiór ID prawidłowych obrazów

        Returns:
            Lista bez sierot
        """
        valid = []

        for ann in annotations:
            image_id = ann.get("image_id")
            if image_id in valid_image_ids:
                valid.append(ann)
            else:
                self.stats.orphans_removed += 1

        return valid

    def merge(
        self,
        auto_path: Path,
        corrections_paths: list[Path],
    ) -> dict:
        """
        Scala anotacje automatyczne z korektami.

        Args:
            auto_path: Ścieżka do anotacji automatycznych
            corrections_paths: Lista ścieżek do plików korekt

        Returns:
            Scalone dane COCO
        """
        # Wczytaj dane
        coco_data = self.load_auto_annotations(auto_path)
        corrections = self.load_corrections(corrections_paths)

        # Zbiór prawidłowych image_id
        valid_image_ids = {img["id"] for img in coco_data.get("images", [])}

        # Przetwórz anotacje
        merged_annotations = []

        for ann in coco_data.get("annotations", []):
            ann_id = ann.get("id")

            # Sprawdź czy jest korekta
            if ann_id in corrections:
                result = self.apply_correction(ann.copy(), corrections[ann_id])
                if result is not None:
                    merged_annotations.append(result)
            else:
                # Bez korekty - zachowaj
                self.stats.unchanged += 1
                merged_annotations.append(ann)

        # Usuń duplikaty
        merged_annotations = self.remove_duplicates(merged_annotations)

        # Usuń sieroty
        merged_annotations = self.remove_orphans(merged_annotations, valid_image_ids)

        self.stats.final_annotations = len(merged_annotations)

        # Zaktualizuj dane COCO
        coco_data["annotations"] = merged_annotations
        coco_data["info"] = coco_data.get("info", {})
        coco_data["info"]["merged_date"] = datetime.now().isoformat()
        coco_data["info"]["merge_stats"] = self.stats.to_dict()

        logger.info(f"Scalono do {self.stats.final_annotations} anotacji")

        return coco_data

    def save(self, data: dict, output_path: Path) -> None:
        """
        Zapisuje scalone dane.

        Args:
            data: Dane COCO
            output_path: Ścieżka wyjściowa
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Zapisano do {output_path}")

    def generate_report(self) -> str:
        """
        Generuje raport ze scalania.

        Returns:
            Raport jako string
        """
        s = self.stats

        lines = [
            "=" * 60,
            "RAPORT SCALANIA ANOTACJI",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "ŹRÓDŁA",
            "-" * 40,
            f"Anotacje automatyczne:   {s.total_auto_annotations}",
            f"Korekty manualne:        {s.total_corrections}",
            "",
            "REZULTAT SCALANIA",
            "-" * 40,
            f"Zaakceptowane:           {s.accepted}",
            f"Poprawione:              {s.corrected}",
            f"Odrzucone:               {s.rejected}",
            f"Bez zmian:               {s.unchanged}",
            "",
            "CZYSZCZENIE",
            "-" * 40,
            f"Duplikaty usunięte:      {s.duplicates_removed}",
            f"Sieroty usunięte:        {s.orphans_removed}",
            "",
            "WYNIK KOŃCOWY",
            "-" * 40,
            f"Finalne anotacje:        {s.final_annotations}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Scalanie anotacji automatycznych z korektami manualnymi"
    )

    parser.add_argument(
        "--auto",
        type=Path,
        default=Path("data/annotations/annotations.json"),
        help="Ścieżka do anotacji automatycznych",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        nargs="+",
        default=[Path("data/verification/corrections.json")],
        help="Ścieżki do plików korekt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/merged.json"),
        help="Ścieżka wyjściowa",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Pokaż raport",
    )

    args = parser.parse_args()

    # Merge
    merger = AnnotationMerger()
    merged_data = merger.merge(args.auto, args.corrections)
    merger.save(merged_data, args.output)

    # Raport
    if args.report:
        print(merger.generate_report())
    else:
        print(f"\n=== SCALANIE ZAKOŃCZONE ===")
        print(f"Wejście:  {merger.stats.total_auto_annotations} anotacji")
        print(f"Korekty:  {merger.stats.total_corrections}")
        print(f"Wynik:    {merger.stats.final_annotations} anotacji")
        print(f"Zapisano: {args.output}")


if __name__ == "__main__":
    main()
