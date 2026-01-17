#!/usr/bin/env python3
"""
Skrypt do walidacji formatu COCO.

Funkcje:
- Walidacja struktury JSON
- Sprawdzenie wymaganych pól
- Walidacja mapowania image-annotation
- Wykrywanie sierot i duplikatów
- Walidacja z pycocotools (opcjonalnie)

Użycie:
    python scripts/annotation/validate_coco.py --input data/annotations/merged.json
    python scripts/annotation/validate_coco.py --input data/annotations/merged.json --strict
    python scripts/annotation/validate_coco.py --input data/annotations/merged.json --fix --output data/annotations/fixed.json
"""

import argparse
import json
import logging
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
class ValidationResult:
    """Wynik walidacji."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Dodaje błąd."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Dodaje ostrzeżenie."""
        self.warnings.append(message)

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors[:20],  # Pierwsze 20
            "warnings": self.warnings[:20],
            "info": self.info,
        }


class COCOValidator:
    """
    Klasa do walidacji formatu COCO.

    Użycie:
        validator = COCOValidator()
        result = validator.validate(coco_path)
        validator.print_report(result)
    """

    # Wymagane pola według specyfikacji COCO
    REQUIRED_TOP_LEVEL = ["images", "annotations", "categories"]
    REQUIRED_IMAGE_FIELDS = ["id", "file_name", "width", "height"]
    REQUIRED_ANNOTATION_FIELDS = ["id", "image_id", "category_id", "bbox"]
    REQUIRED_CATEGORY_FIELDS = ["id", "name"]

    def __init__(self, strict: bool = False) -> None:
        """
        Inicjalizuje walidator.

        Args:
            strict: Tryb ścisły (więcej walidacji)
        """
        self.strict = strict

    def load_coco(self, path: Path) -> Optional[dict]:
        """
        Wczytuje plik COCO.

        Args:
            path: Ścieżka do pliku

        Returns:
            Dane COCO lub None
        """
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Błąd parsowania JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Błąd wczytywania pliku: {e}")
            return None

    def validate_structure(self, data: dict, result: ValidationResult) -> None:
        """
        Waliduje strukturę COCO.

        Args:
            data: Dane COCO
            result: Wynik walidacji
        """
        # Sprawdź pola top-level
        for field_name in self.REQUIRED_TOP_LEVEL:
            if field_name not in data:
                result.add_error(f"Brak wymaganego pola: {field_name}")
            elif not isinstance(data[field_name], list):
                result.add_error(f"Pole {field_name} powinno być listą")

        # Info (opcjonalne ale zalecane)
        if "info" not in data:
            result.add_warning("Brak pola 'info' (zalecane)")

    def validate_images(self, images: list[dict], result: ValidationResult) -> set[int]:
        """
        Waliduje obrazy.

        Args:
            images: Lista obrazów
            result: Wynik walidacji

        Returns:
            Zbiór ID obrazów
        """
        image_ids = set()
        file_names = set()

        for i, img in enumerate(images):
            # Wymagane pola
            for field_name in self.REQUIRED_IMAGE_FIELDS:
                if field_name not in img:
                    result.add_error(f"Obraz [{i}]: brak pola '{field_name}'")

            # Unikalne ID
            img_id = img.get("id")
            if img_id is not None:
                if img_id in image_ids:
                    result.add_error(f"Duplikat image_id: {img_id}")
                image_ids.add(img_id)

            # Unikalne file_name
            file_name = img.get("file_name")
            if file_name:
                if file_name in file_names:
                    result.add_warning(f"Duplikat file_name: {file_name}")
                file_names.add(file_name)

            # Walidacja wartości
            if self.strict:
                width = img.get("width", 0)
                height = img.get("height", 0)
                if width <= 0 or height <= 0:
                    result.add_error(f"Obraz {img_id}: nieprawidłowe wymiary ({width}x{height})")

        result.info["total_images"] = len(images)
        result.info["unique_image_ids"] = len(image_ids)

        return image_ids

    def validate_categories(
        self,
        categories: list[dict],
        result: ValidationResult,
    ) -> set[int]:
        """
        Waliduje kategorie.

        Args:
            categories: Lista kategorii
            result: Wynik walidacji

        Returns:
            Zbiór ID kategorii
        """
        category_ids = set()

        for i, cat in enumerate(categories):
            # Wymagane pola
            for field_name in self.REQUIRED_CATEGORY_FIELDS:
                if field_name not in cat:
                    result.add_error(f"Kategoria [{i}]: brak pola '{field_name}'")

            # Unikalne ID
            cat_id = cat.get("id")
            if cat_id is not None:
                if cat_id in category_ids:
                    result.add_error(f"Duplikat category_id: {cat_id}")
                category_ids.add(cat_id)

        result.info["total_categories"] = len(categories)

        return category_ids

    def validate_annotations(
        self,
        annotations: list[dict],
        valid_image_ids: set[int],
        valid_category_ids: set[int],
        result: ValidationResult,
    ) -> None:
        """
        Waliduje anotacje.

        Args:
            annotations: Lista anotacji
            valid_image_ids: Prawidłowe ID obrazów
            valid_category_ids: Prawidłowe ID kategorii
            result: Wynik walidacji
        """
        annotation_ids = set()
        orphan_count = 0
        invalid_category_count = 0

        for i, ann in enumerate(annotations):
            # Wymagane pola
            for field_name in self.REQUIRED_ANNOTATION_FIELDS:
                if field_name not in ann:
                    result.add_error(f"Anotacja [{i}]: brak pola '{field_name}'")

            # Unikalne ID
            ann_id = ann.get("id")
            if ann_id is not None:
                if ann_id in annotation_ids:
                    result.add_error(f"Duplikat annotation_id: {ann_id}")
                annotation_ids.add(ann_id)

            # Sprawdź image_id
            image_id = ann.get("image_id")
            if image_id not in valid_image_ids:
                orphan_count += 1
                if self.strict:
                    result.add_error(f"Anotacja {ann_id}: nieistniejący image_id {image_id}")

            # Sprawdź category_id
            category_id = ann.get("category_id")
            if category_id not in valid_category_ids:
                invalid_category_count += 1
                if self.strict:
                    result.add_error(f"Anotacja {ann_id}: nieistniejący category_id {category_id}")

            # Walidacja bbox
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                result.add_error(f"Anotacja {ann_id}: bbox powinien mieć 4 elementy")
            elif self.strict:
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    result.add_error(f"Anotacja {ann_id}: nieprawidłowe wymiary bbox")

            # Walidacja area (jeśli obecne)
            if "area" in ann and self.strict:
                area = ann.get("area", 0)
                expected_area = bbox[2] * bbox[3] if len(bbox) == 4 else 0
                if abs(area - expected_area) > 1:
                    result.add_warning(f"Anotacja {ann_id}: area nie zgadza się z bbox")

        result.info["total_annotations"] = len(annotations)
        result.info["unique_annotation_ids"] = len(annotation_ids)
        result.info["orphan_annotations"] = orphan_count
        result.info["invalid_category_refs"] = invalid_category_count

        if orphan_count > 0:
            result.add_warning(f"Znaleziono {orphan_count} anotacji bez odpowiadających obrazów")

        if invalid_category_count > 0:
            result.add_warning(f"Znaleziono {invalid_category_count} anotacji z nieprawidłowymi kategoriami")

    def validate_with_pycocotools(
        self,
        path: Path,
        result: ValidationResult,
    ) -> None:
        """
        Waliduje z użyciem pycocotools.

        Args:
            path: Ścieżka do pliku
            result: Wynik walidacji
        """
        try:
            from pycocotools.coco import COCO

            coco = COCO(str(path))

            # Podstawowe statystyki
            result.info["pycocotools_images"] = len(coco.getImgIds())
            result.info["pycocotools_annotations"] = len(coco.getAnnIds())
            result.info["pycocotools_categories"] = len(coco.getCatIds())

            logger.info("Walidacja pycocotools: PASS")

        except ImportError:
            result.add_warning("pycocotools nie jest zainstalowane - pominięto walidację")
        except Exception as e:
            result.add_error(f"pycocotools walidacja nie powiodła się: {e}")

    def validate(self, path: Path) -> ValidationResult:
        """
        Przeprowadza pełną walidację.

        Args:
            path: Ścieżka do pliku COCO

        Returns:
            Wynik walidacji
        """
        result = ValidationResult()

        # Wczytaj plik
        data = self.load_coco(path)
        if data is None:
            result.add_error("Nie można wczytać pliku COCO")
            return result

        # Walidacja struktury
        self.validate_structure(data, result)
        if not result.is_valid:
            return result

        # Walidacja obrazów
        valid_image_ids = self.validate_images(data.get("images", []), result)

        # Walidacja kategorii
        valid_category_ids = self.validate_categories(data.get("categories", []), result)

        # Walidacja anotacji
        self.validate_annotations(
            data.get("annotations", []),
            valid_image_ids,
            valid_category_ids,
            result,
        )

        # Walidacja pycocotools (opcjonalnie)
        if self.strict:
            self.validate_with_pycocotools(path, result)

        return result

    def fix_issues(self, path: Path, output_path: Path) -> ValidationResult:
        """
        Naprawia typowe problemy.

        Args:
            path: Ścieżka wejściowa
            output_path: Ścieżka wyjściowa

        Returns:
            Wynik walidacji po naprawie
        """
        data = self.load_coco(path)
        if data is None:
            result = ValidationResult()
            result.add_error("Nie można wczytać pliku")
            return result

        # Zbierz prawidłowe ID
        valid_image_ids = {img["id"] for img in data.get("images", [])}
        valid_category_ids = {cat["id"] for cat in data.get("categories", [])}

        # Napraw anotacje
        fixed_annotations = []
        removed_count = 0

        for ann in data.get("annotations", []):
            # Usuń sieroty
            if ann.get("image_id") not in valid_image_ids:
                removed_count += 1
                continue

            # Napraw brakujące pole area
            if "area" not in ann:
                bbox = ann.get("bbox", [0, 0, 0, 0])
                if len(bbox) >= 4:
                    ann["area"] = bbox[2] * bbox[3]

            # Napraw brakujące iscrowd
            if "iscrowd" not in ann:
                ann["iscrowd"] = 0

            fixed_annotations.append(ann)

        data["annotations"] = fixed_annotations

        # Zapisz
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Naprawiono plik, usunięto {removed_count} nieprawidłowych anotacji")

        # Waliduj naprawiony plik
        return self.validate(output_path)

    def generate_report(self, result: ValidationResult) -> str:
        """
        Generuje raport walidacji.

        Args:
            result: Wynik walidacji

        Returns:
            Raport jako string
        """
        lines = [
            "=" * 60,
            "RAPORT WALIDACJI COCO",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            f"STATUS: {'✓ VALID' if result.is_valid else '✗ INVALID'}",
            "",
            "STATYSTYKI",
            "-" * 40,
        ]

        for key, value in result.info.items():
            lines.append(f"  {key}: {value}")

        lines.append("")

        if result.errors:
            lines.extend([
                f"BŁĘDY ({len(result.errors)})",
                "-" * 40,
            ])
            for error in result.errors[:10]:
                lines.append(f"  ✗ {error}")
            if len(result.errors) > 10:
                lines.append(f"  ... i {len(result.errors) - 10} więcej")
            lines.append("")

        if result.warnings:
            lines.extend([
                f"OSTRZEŻENIA ({len(result.warnings)})",
                "-" * 40,
            ])
            for warning in result.warnings[:10]:
                lines.append(f"  ! {warning}")
            if len(result.warnings) > 10:
                lines.append(f"  ... i {len(result.warnings) - 10} więcej")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Walidacja formatu COCO"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Ścieżka do pliku COCO",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Tryb ścisły (więcej walidacji)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Napraw typowe problemy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ścieżka wyjściowa dla naprawionego pliku (z --fix)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Wynik w formacie JSON",
    )

    args = parser.parse_args()

    validator = COCOValidator(strict=args.strict)

    if args.fix:
        output_path = args.output or args.input.with_suffix(".fixed.json")
        result = validator.fix_issues(args.input, output_path)
        print(f"Naprawiono i zapisano do: {output_path}")
    else:
        result = validator.validate(args.input)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(validator.generate_report(result))

    # Exit code
    exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
