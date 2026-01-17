#!/usr/bin/env python3
"""
Skrypt do obliczania metryk zgodności między annotatorami.

Funkcje:
- Cohen's Kappa dla zgodności par annotatorów
- Fleiss' Kappa dla wielu annotatorów
- Procent zgodności (percent agreement)
- Confusion matrix dla emocji
- Raport zgodności

Użycie:
    python scripts/verification/agreement_calculator.py --corrections data/verification/corrections_*.json
    python scripts/verification/agreement_calculator.py --auto-vs-manual --auto data/annotations/annotations.json --manual data/verification/corrections.json
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AgreementMetrics:
    """Metryki zgodności."""

    percent_agreement: float = 0.0
    cohens_kappa: float = 0.0
    fleiss_kappa: float = 0.0
    confusion_matrix: dict = field(default_factory=dict)
    per_class_agreement: dict = field(default_factory=dict)
    total_comparisons: int = 0
    agreed_count: int = 0

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "percent_agreement": round(self.percent_agreement, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
            "fleiss_kappa": round(self.fleiss_kappa, 4),
            "total_comparisons": self.total_comparisons,
            "agreed_count": self.agreed_count,
            "per_class_agreement": {
                k: round(v, 4) for k, v in self.per_class_agreement.items()
            },
        }


class AgreementCalculator:
    """
    Klasa do obliczania metryk zgodności.

    Użycie:
        calculator = AgreementCalculator()
        metrics = calculator.calculate_agreement(labels1, labels2)
        calculator.generate_report(metrics)
    """

    def __init__(self, classes: Optional[list[str]] = None) -> None:
        """
        Inicjalizuje kalkulator.

        Args:
            classes: Lista klas (opcjonalnie)
        """
        self.classes = classes or [
            "happy", "sad", "angry", "relaxed", "fearful", "neutral"
        ]

    def _percent_agreement(
        self,
        labels1: list[str],
        labels2: list[str],
    ) -> float:
        """
        Oblicza procent zgodności.

        Args:
            labels1: Etykiety annotatora 1
            labels2: Etykiety annotatora 2

        Returns:
            Procent zgodności
        """
        if len(labels1) != len(labels2) or len(labels1) == 0:
            return 0.0

        agreed = sum(1 for a, b in zip(labels1, labels2) if a == b)
        return agreed / len(labels1)

    def _cohens_kappa(
        self,
        labels1: list[str],
        labels2: list[str],
    ) -> float:
        """
        Oblicza Cohen's Kappa.

        Args:
            labels1: Etykiety annotatora 1
            labels2: Etykiety annotatora 2

        Returns:
            Cohen's Kappa
        """
        if len(labels1) != len(labels2) or len(labels1) == 0:
            return 0.0

        n = len(labels1)
        all_labels = list(set(labels1) | set(labels2))

        # Confusion matrix
        matrix = defaultdict(lambda: defaultdict(int))
        for a, b in zip(labels1, labels2):
            matrix[a][b] += 1

        # P_o (observed agreement)
        p_o = sum(matrix[c][c] for c in all_labels) / n

        # P_e (expected agreement by chance)
        p_e = 0.0
        for c in all_labels:
            row_sum = sum(matrix[c].values())
            col_sum = sum(matrix[r][c] for r in all_labels)
            p_e += (row_sum / n) * (col_sum / n)

        # Kappa
        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0

        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

    def _fleiss_kappa(
        self,
        ratings_matrix: list[list[int]],
    ) -> float:
        """
        Oblicza Fleiss' Kappa dla wielu annotatorów.

        Args:
            ratings_matrix: Macierz [n_subjects x n_categories]
                            Każdy element = liczba annotatorów, którzy wybrali daną kategorię

        Returns:
            Fleiss' Kappa
        """
        if not ratings_matrix or not ratings_matrix[0]:
            return 0.0

        N = len(ratings_matrix)  # liczba podmiotów
        k = len(ratings_matrix[0])  # liczba kategorii
        n = sum(ratings_matrix[0])  # liczba annotatorów per podmiot

        if n <= 1:
            return 0.0

        # P_i dla każdego podmiotu
        P_i = []
        for row in ratings_matrix:
            p = sum(r * (r - 1) for r in row) / (n * (n - 1))
            P_i.append(p)

        # Średnie P
        P_bar = sum(P_i) / N

        # p_j dla każdej kategorii
        p_j = []
        for j in range(k):
            total = sum(row[j] for row in ratings_matrix)
            p_j.append(total / (N * n))

        # P_e
        P_e = sum(p ** 2 for p in p_j)

        # Kappa
        if P_e == 1.0:
            return 1.0 if P_bar == 1.0 else 0.0

        kappa = (P_bar - P_e) / (1 - P_e)
        return kappa

    def _build_confusion_matrix(
        self,
        labels1: list[str],
        labels2: list[str],
    ) -> dict[str, dict[str, int]]:
        """
        Buduje confusion matrix.

        Args:
            labels1: Etykiety annotatora 1 (prawdziwe/oryginalne)
            labels2: Etykiety annotatora 2 (predykcje/korekty)

        Returns:
            Confusion matrix jako nested dict
        """
        matrix = {c: {c2: 0 for c2 in self.classes} for c in self.classes}

        for true_label, pred_label in zip(labels1, labels2):
            if true_label in matrix and pred_label in matrix[true_label]:
                matrix[true_label][pred_label] += 1

        return matrix

    def calculate_agreement(
        self,
        labels1: list[str],
        labels2: list[str],
    ) -> AgreementMetrics:
        """
        Oblicza wszystkie metryki zgodności dla dwóch annotatorów.

        Args:
            labels1: Etykiety annotatora 1
            labels2: Etykiety annotatora 2

        Returns:
            Metryki zgodności
        """
        metrics = AgreementMetrics()

        if len(labels1) != len(labels2):
            logger.warning("Różna liczba etykiet!")
            return metrics

        metrics.total_comparisons = len(labels1)
        metrics.agreed_count = sum(1 for a, b in zip(labels1, labels2) if a == b)

        # Podstawowe metryki
        metrics.percent_agreement = self._percent_agreement(labels1, labels2)
        metrics.cohens_kappa = self._cohens_kappa(labels1, labels2)

        # Confusion matrix
        metrics.confusion_matrix = self._build_confusion_matrix(labels1, labels2)

        # Per-class agreement
        for cls in self.classes:
            cls_indices = [i for i, l in enumerate(labels1) if l == cls]
            if cls_indices:
                agreed = sum(1 for i in cls_indices if labels1[i] == labels2[i])
                metrics.per_class_agreement[cls] = agreed / len(cls_indices)
            else:
                metrics.per_class_agreement[cls] = 0.0

        return metrics

    def calculate_multi_annotator_agreement(
        self,
        all_labels: list[list[str]],
    ) -> AgreementMetrics:
        """
        Oblicza zgodność dla wielu annotatorów.

        Args:
            all_labels: Lista list etykiet [annotator1, annotator2, ...]

        Returns:
            Metryki zgodności
        """
        metrics = AgreementMetrics()

        if not all_labels or len(all_labels) < 2:
            return metrics

        n_annotators = len(all_labels)
        n_subjects = len(all_labels[0])

        # Sprawdź spójność
        for labels in all_labels:
            if len(labels) != n_subjects:
                logger.warning("Niezgodna liczba etykiet między annotatorami!")
                return metrics

        metrics.total_comparisons = n_subjects

        # Buduj macierz dla Fleiss' Kappa
        ratings_matrix = []

        for i in range(n_subjects):
            # Zlicz głosy dla każdej kategorii
            votes = {c: 0 for c in self.classes}
            for annotator_labels in all_labels:
                label = annotator_labels[i]
                if label in votes:
                    votes[label] += 1

            ratings_matrix.append([votes[c] for c in self.classes])

        # Fleiss' Kappa
        metrics.fleiss_kappa = self._fleiss_kappa(ratings_matrix)

        # Procent pełnej zgodności (wszyscy annotatorzy zgodnie)
        full_agreement = 0
        for i in range(n_subjects):
            labels_at_i = [all_labels[a][i] for a in range(n_annotators)]
            if len(set(labels_at_i)) == 1:
                full_agreement += 1

        metrics.percent_agreement = full_agreement / n_subjects if n_subjects > 0 else 0.0
        metrics.agreed_count = full_agreement

        return metrics

    def load_corrections(self, corrections_path: Path) -> dict[int, dict]:
        """
        Wczytuje plik korekt.

        Args:
            corrections_path: Ścieżka do pliku

        Returns:
            Słownik {annotation_id -> correction}
        """
        with open(corrections_path, encoding="utf-8") as f:
            data = json.load(f)

        corrections = {}
        for corr in data.get("corrections", []):
            ann_id = corr.get("annotation_id")
            if ann_id is not None:
                corrections[ann_id] = corr

        return corrections

    def compare_auto_vs_manual(
        self,
        auto_annotations_path: Path,
        manual_corrections_path: Path,
    ) -> AgreementMetrics:
        """
        Porównuje anotacje automatyczne z manualnymi korektami.

        Args:
            auto_annotations_path: Ścieżka do anotacji automatycznych
            manual_corrections_path: Ścieżka do korekt manualnych

        Returns:
            Metryki zgodności
        """
        # Wczytaj anotacje automatyczne
        with open(auto_annotations_path, encoding="utf-8") as f:
            auto_data = json.load(f)

        auto_annotations = {
            ann["id"]: ann for ann in auto_data.get("annotations", [])
        }

        # Wczytaj korekty
        corrections = self.load_corrections(manual_corrections_path)

        # Zbierz pary etykiet
        auto_labels = []
        manual_labels = []

        for ann_id, corr in corrections.items():
            if ann_id not in auto_annotations:
                continue

            auto_ann = auto_annotations[ann_id]

            # Oryginalna emocja
            auto_emotion = auto_ann.get("emotion", {})
            if isinstance(auto_emotion, dict):
                auto_label = auto_emotion.get("name", "unknown")
            else:
                auto_label = str(auto_emotion) if auto_emotion else "unknown"

            # Manualna etykieta
            corr_type = corr.get("type", "")
            corr_data = corr.get("corrected_data", {})

            if corr_type == "accept":
                manual_label = auto_label  # Akceptacja = zgoda
            elif corr_type == "correct" and "emotion" in corr_data:
                manual_label = corr_data["emotion"]
            elif corr_type == "reject":
                manual_label = "rejected"  # Specjalna etykieta dla odrzuconych
            else:
                manual_label = auto_label

            auto_labels.append(auto_label)
            manual_labels.append(manual_label)

        logger.info(f"Porównano {len(auto_labels)} anotacji")

        return self.calculate_agreement(auto_labels, manual_labels)

    def generate_report(self, metrics: AgreementMetrics) -> str:
        """
        Generuje raport zgodności.

        Args:
            metrics: Metryki zgodności

        Returns:
            Raport jako string
        """
        lines = [
            "=" * 60,
            "RAPORT ZGODNOŚCI ANNOTATORÓW",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "OGÓLNE METRYKI",
            "-" * 40,
            f"Procent zgodności:     {metrics.percent_agreement * 100:.2f}%",
            f"Cohen's Kappa:         {metrics.cohens_kappa:.4f}",
        ]

        if metrics.fleiss_kappa != 0:
            lines.append(f"Fleiss' Kappa:         {metrics.fleiss_kappa:.4f}")

        lines.extend([
            f"",
            f"Łącznie porównań:      {metrics.total_comparisons}",
            f"Zgodnych:              {metrics.agreed_count}",
            "",
        ])

        # Interpretacja Kappa
        kappa = metrics.cohens_kappa if metrics.cohens_kappa != 0 else metrics.fleiss_kappa
        interpretation = self._interpret_kappa(kappa)
        lines.extend([
            "INTERPRETACJA KAPPA",
            "-" * 40,
            f"Wartość:  {kappa:.4f}",
            f"Ocena:    {interpretation}",
            "",
        ])

        # Per-class agreement
        if metrics.per_class_agreement:
            lines.extend([
                "ZGODNOŚĆ PER KLASA",
                "-" * 40,
            ])
            for cls, agreement in sorted(metrics.per_class_agreement.items()):
                bar = "█" * int(agreement * 20)
                lines.append(f"  {cls:12s}: {agreement * 100:5.1f}%  {bar}")
            lines.append("")

        # Confusion matrix
        if metrics.confusion_matrix:
            lines.extend([
                "CONFUSION MATRIX",
                "-" * 40,
            ])

            # Nagłówek
            classes = list(metrics.confusion_matrix.keys())
            header = "           " + "  ".join(f"{c[:6]:>6s}" for c in classes)
            lines.append(header)

            # Wiersze
            for cls in classes:
                row = metrics.confusion_matrix.get(cls, {})
                values = "  ".join(f"{row.get(c, 0):6d}" for c in classes)
                lines.append(f"{cls[:10]:10s} {values}")

            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpretuje wartość Kappa."""
        if kappa < 0:
            return "Poniżej losowej zgodności"
        elif kappa < 0.20:
            return "Słaba zgodność"
        elif kappa < 0.40:
            return "Umiarkowana zgodność"
        elif kappa < 0.60:
            return "Średnia zgodność"
        elif kappa < 0.80:
            return "Dobra zgodność"
        elif kappa < 1.00:
            return "Bardzo dobra zgodność"
        else:
            return "Pełna zgodność"

    def save_metrics(
        self,
        metrics: AgreementMetrics,
        output_path: Path,
    ) -> None:
        """Zapisuje metryki do pliku JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "generated": datetime.now().isoformat(),
            "metrics": metrics.to_dict(),
            "confusion_matrix": metrics.confusion_matrix,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Zapisano metryki do {output_path}")


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Obliczanie metryk zgodności między annotatorami"
    )

    parser.add_argument(
        "--corrections",
        type=Path,
        nargs="+",
        help="Pliki korekt od różnych annotatorów",
    )
    parser.add_argument(
        "--auto-vs-manual",
        action="store_true",
        help="Porównaj anotacje automatyczne z manualnymi",
    )
    parser.add_argument(
        "--auto",
        type=Path,
        default=Path("data/annotations/annotations.json"),
        help="Ścieżka do anotacji automatycznych (z --auto-vs-manual)",
    )
    parser.add_argument(
        "--manual",
        type=Path,
        default=Path("data/verification/corrections.json"),
        help="Ścieżka do korekt manualnych (z --auto-vs-manual)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/verification/agreement_metrics.json"),
        help="Ścieżka wyjściowa dla metryk",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Pokaż raport",
    )

    args = parser.parse_args()

    calculator = AgreementCalculator()

    # Auto vs Manual
    if args.auto_vs_manual:
        logger.info("Porównanie: automatyczne vs manualne")
        metrics = calculator.compare_auto_vs_manual(args.auto, args.manual)

    # Porównanie wielu annotatorów
    elif args.corrections and len(args.corrections) >= 2:
        logger.info(f"Porównanie {len(args.corrections)} annotatorów")

        # TODO: Implementacja porównania wielu plików korekt
        # Na razie porównujemy pierwsze dwa
        corr1 = calculator.load_corrections(args.corrections[0])
        corr2 = calculator.load_corrections(args.corrections[1])

        # Znajdź wspólne anotacje
        common_ids = set(corr1.keys()) & set(corr2.keys())

        labels1 = []
        labels2 = []

        for ann_id in common_ids:
            c1 = corr1[ann_id]
            c2 = corr2[ann_id]

            # Wyciągnij etykiety emocji z korekt
            l1 = c1.get("corrected_data", {}).get("emotion", c1.get("type", "unknown"))
            l2 = c2.get("corrected_data", {}).get("emotion", c2.get("type", "unknown"))

            labels1.append(l1)
            labels2.append(l2)

        metrics = calculator.calculate_agreement(labels1, labels2)

    else:
        parser.print_help()
        return

    # Zapisz metryki
    calculator.save_metrics(metrics, args.output)

    # Raport
    if args.report:
        print(calculator.generate_report(metrics))
    else:
        print(f"\n=== WYNIK ===")
        print(f"Procent zgodności: {metrics.percent_agreement * 100:.2f}%")
        print(f"Cohen's Kappa:     {metrics.cohens_kappa:.4f}")
        print(f"Zapisano do: {args.output}")


if __name__ == "__main__":
    main()
