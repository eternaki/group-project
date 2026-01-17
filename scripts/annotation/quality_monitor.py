#!/usr/bin/env python3
"""
Skrypt do monitorowania jakości anotacji.

Funkcje:
- Logowanie wyników confidence
- Flagowanie klatek o niskiej pewności
- Eksport przykładowych wizualizacji
- Śledzenie metryk jakości

Użycie:
    python scripts/annotation/quality_monitor.py --annotations data/annotations/annotations.json
    python scripts/annotation/quality_monitor.py --visualize --sample 10
    python scripts/annotation/quality_monitor.py --report
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """Progi jakości."""

    min_bbox_confidence: float = 0.3
    min_breed_confidence: float = 0.5
    min_emotion_confidence: float = 0.4
    min_keypoints_visible: int = 10
    low_confidence_flag: float = 0.5


@dataclass
class QualityMetrics:
    """Metryki jakości anotacji."""

    total_images: int = 0
    total_annotations: int = 0
    avg_bbox_confidence: float = 0.0
    avg_breed_confidence: float = 0.0
    avg_emotion_confidence: float = 0.0
    avg_keypoints_visible: float = 0.0
    low_confidence_count: int = 0
    flagged_images: list[str] = field(default_factory=list)
    by_emotion: dict = field(default_factory=dict)
    by_breed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "total_images": self.total_images,
            "total_annotations": self.total_annotations,
            "avg_bbox_confidence": round(self.avg_bbox_confidence, 4),
            "avg_breed_confidence": round(self.avg_breed_confidence, 4),
            "avg_emotion_confidence": round(self.avg_emotion_confidence, 4),
            "avg_keypoints_visible": round(self.avg_keypoints_visible, 1),
            "low_confidence_count": self.low_confidence_count,
            "low_confidence_percent": round(
                self.low_confidence_count / max(self.total_annotations, 1) * 100, 2
            ),
            "by_emotion": self.by_emotion,
            "by_breed": dict(list(self.by_breed.items())[:20]),  # Top 20
        }


class QualityMonitor:
    """
    Klasa do monitorowania jakości anotacji.

    Użycie:
        monitor = QualityMonitor(annotations_path)
        metrics = monitor.analyze()
        monitor.generate_report()
    """

    def __init__(
        self,
        annotations_path: Path,
        frames_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        thresholds: Optional[QualityThresholds] = None,
    ) -> None:
        """
        Inicjalizuje monitor.

        Args:
            annotations_path: Ścieżka do pliku COCO JSON
            frames_dir: Katalog z klatkami
            output_dir: Katalog wyjściowy
            thresholds: Progi jakości
        """
        self.annotations_path = Path(annotations_path)
        self.frames_dir = frames_dir or Path("data/frames")
        self.output_dir = output_dir or Path("data/quality")
        self.thresholds = thresholds or QualityThresholds()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.coco_data: dict = {}
        self.metrics = QualityMetrics()

        self._load_annotations()

    def _load_annotations(self) -> None:
        """Wczytuje anotacje COCO."""
        if not self.annotations_path.exists():
            logger.warning(f"Plik anotacji nie istnieje: {self.annotations_path}")
            return

        try:
            with open(self.annotations_path, encoding="utf-8") as f:
                self.coco_data = json.load(f)
            logger.info(f"Wczytano anotacje: {len(self.coco_data.get('images', []))} obrazów")
        except Exception as e:
            logger.error(f"Błąd wczytywania anotacji: {e}")

    def analyze(self) -> QualityMetrics:
        """
        Analizuje jakość anotacji.

        Returns:
            Metryki jakości
        """
        images = self.coco_data.get("images", [])
        annotations = self.coco_data.get("annotations", [])

        self.metrics.total_images = len(images)
        self.metrics.total_annotations = len(annotations)

        # Mapowanie obrazów
        image_map = {img["id"]: img for img in images}

        # Agregatory
        bbox_confidences = []
        breed_confidences = []
        emotion_confidences = []
        keypoints_visible_counts = []

        for ann in annotations:
            # BBox confidence
            bbox_conf = ann.get("score", ann.get("confidence", 1.0))
            bbox_confidences.append(bbox_conf)

            # Breed confidence
            breed_data = ann.get("breed", {})
            if isinstance(breed_data, dict):
                breed_conf = breed_data.get("confidence", 0)
                breed_name = breed_data.get("name", "unknown")
            else:
                breed_conf = 0
                breed_name = str(breed_data) if breed_data else "unknown"

            breed_confidences.append(breed_conf)

            # Statystyki per rasa
            if breed_name not in self.metrics.by_breed:
                self.metrics.by_breed[breed_name] = 0
            self.metrics.by_breed[breed_name] += 1

            # Emotion confidence
            emotion_data = ann.get("emotion", {})
            if isinstance(emotion_data, dict):
                emotion_conf = emotion_data.get("confidence", 0)
                emotion_name = emotion_data.get("name", "unknown")
            else:
                emotion_conf = 0
                emotion_name = str(emotion_data) if emotion_data else "unknown"

            emotion_confidences.append(emotion_conf)

            # Statystyki per emocja
            if emotion_name not in self.metrics.by_emotion:
                self.metrics.by_emotion[emotion_name] = {"count": 0, "avg_confidence": 0}
            self.metrics.by_emotion[emotion_name]["count"] += 1

            # Keypoints
            keypoints = ann.get("keypoints", [])
            if keypoints:
                # Keypoints w formacie [x, y, v, x, y, v, ...]
                visible_count = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
                keypoints_visible_counts.append(visible_count)

            # Flagowanie niskiej jakości
            if bbox_conf < self.thresholds.low_confidence_flag:
                self.metrics.low_confidence_count += 1

                image_id = ann.get("image_id")
                if image_id in image_map:
                    self.metrics.flagged_images.append(image_map[image_id].get("file_name", ""))

        # Oblicz średnie
        if bbox_confidences:
            self.metrics.avg_bbox_confidence = sum(bbox_confidences) / len(bbox_confidences)

        if breed_confidences:
            self.metrics.avg_breed_confidence = sum(breed_confidences) / len(breed_confidences)

        if emotion_confidences:
            self.metrics.avg_emotion_confidence = sum(emotion_confidences) / len(emotion_confidences)

        if keypoints_visible_counts:
            self.metrics.avg_keypoints_visible = sum(keypoints_visible_counts) / len(keypoints_visible_counts)

        # Średnie confidence per emocja
        for emotion_name in self.metrics.by_emotion:
            count = self.metrics.by_emotion[emotion_name]["count"]
            if count > 0:
                emotion_confidences_filtered = [
                    ann.get("emotion", {}).get("confidence", 0)
                    for ann in annotations
                    if ann.get("emotion", {}).get("name") == emotion_name
                ]
                if emotion_confidences_filtered:
                    self.metrics.by_emotion[emotion_name]["avg_confidence"] = round(
                        sum(emotion_confidences_filtered) / len(emotion_confidences_filtered), 4
                    )

        # Sortuj rasy po ilości
        self.metrics.by_breed = dict(
            sorted(self.metrics.by_breed.items(), key=lambda x: x[1], reverse=True)
        )

        return self.metrics

    def visualize_sample(
        self,
        sample_size: int = 10,
        output_subdir: str = "samples",
    ) -> list[Path]:
        """
        Tworzy wizualizacje próbki anotacji.

        Args:
            sample_size: Liczba próbek
            output_subdir: Podkatalog wyjściowy

        Returns:
            Lista ścieżek do wizualizacji
        """
        output_path = self.output_dir / output_subdir
        output_path.mkdir(parents=True, exist_ok=True)

        images = self.coco_data.get("images", [])
        annotations = self.coco_data.get("annotations", [])

        if not images:
            logger.warning("Brak obrazów do wizualizacji")
            return []

        # Losowa próbka
        sample_images = random.sample(images, min(sample_size, len(images)))

        # Mapowanie anotacji per obraz
        ann_by_image = {}
        for ann in annotations:
            image_id = ann.get("image_id")
            if image_id not in ann_by_image:
                ann_by_image[image_id] = []
            ann_by_image[image_id].append(ann)

        visualizations = []

        for img_info in sample_images:
            image_id = img_info["id"]
            file_name = img_info.get("file_name", "")

            # Wczytaj obraz
            image_path = self.frames_dir / file_name
            if not image_path.exists():
                logger.warning(f"Obraz nie istnieje: {image_path}")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            # Rysuj anotacje
            image_anns = ann_by_image.get(image_id, [])

            for ann in image_anns:
                # BBox
                bbox = ann.get("bbox", [])
                if len(bbox) >= 4:
                    x, y, w, h = [int(v) for v in bbox[:4]]
                    conf = ann.get("score", ann.get("confidence", 0))

                    # Kolor zależny od confidence
                    if conf >= 0.7:
                        color = (0, 255, 0)  # Zielony
                    elif conf >= 0.5:
                        color = (0, 255, 255)  # Żółty
                    else:
                        color = (0, 0, 255)  # Czerwony

                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                    # Etykiety
                    breed = ann.get("breed", {})
                    breed_name = breed.get("name", "") if isinstance(breed, dict) else str(breed)

                    emotion = ann.get("emotion", {})
                    emotion_name = emotion.get("name", "") if isinstance(emotion, dict) else str(emotion)

                    label = f"{breed_name[:15]} | {emotion_name} ({conf:.2f})"
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Keypoints
                keypoints = ann.get("keypoints", [])
                if keypoints:
                    for i in range(0, len(keypoints) - 2, 3):
                        kx, ky, kv = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                        if kv > 0:
                            cv2.circle(image, (int(kx), int(ky)), 3, (255, 0, 0), -1)

            # Zapisz
            output_file = output_path / f"sample_{image_id}.jpg"
            cv2.imwrite(str(output_file), image)
            visualizations.append(output_file)

            logger.debug(f"Zapisano wizualizację: {output_file}")

        logger.info(f"Utworzono {len(visualizations)} wizualizacji w {output_path}")
        return visualizations

    def export_flagged(self, output_file: Optional[Path] = None) -> Path:
        """
        Eksportuje listę flagowanych klatek.

        Args:
            output_file: Ścieżka wyjściowa

        Returns:
            Ścieżka do pliku
        """
        if output_file is None:
            output_file = self.output_dir / "flagged_frames.json"

        data = {
            "generated": datetime.now().isoformat(),
            "threshold": self.thresholds.low_confidence_flag,
            "total_flagged": len(self.metrics.flagged_images),
            "frames": list(set(self.metrics.flagged_images)),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Wyeksportowano {len(data['frames'])} flagowanych klatek do {output_file}")
        return output_file

    def generate_report(self) -> str:
        """
        Generuje raport jakości.

        Returns:
            Raport jako string
        """
        m = self.metrics

        lines = [
            "=" * 60,
            "RAPORT JAKOŚCI ANOTACJI",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "OGÓLNE STATYSTYKI",
            "-" * 40,
            f"Łącznie obrazów:       {m.total_images}",
            f"Łącznie anotacji:      {m.total_annotations}",
            f"Średnio per obraz:     {m.total_annotations / max(m.total_images, 1):.2f}",
            "",
            "JAKOŚĆ DETEKCJI",
            "-" * 40,
            f"Śr. confidence bbox:   {m.avg_bbox_confidence:.4f}",
            f"Śr. confidence rasa:   {m.avg_breed_confidence:.4f}",
            f"Śr. confidence emocja: {m.avg_emotion_confidence:.4f}",
            f"Śr. widoczne keypoints: {m.avg_keypoints_visible:.1f}",
            "",
            "FLAGI JAKOŚCI",
            "-" * 40,
            f"Niska pewność:         {m.low_confidence_count} ({m.low_confidence_count / max(m.total_annotations, 1) * 100:.1f}%)",
            f"Próg flagowania:       {self.thresholds.low_confidence_flag}",
            "",
        ]

        # Emocje
        if m.by_emotion:
            lines.extend([
                "ROZKŁAD EMOCJI",
                "-" * 40,
            ])
            for emotion, data in sorted(m.by_emotion.items()):
                count = data.get("count", 0)
                conf = data.get("avg_confidence", 0)
                percent = count / max(m.total_annotations, 1) * 100
                lines.append(f"  {emotion:12s}: {count:5d} ({percent:5.1f}%) conf: {conf:.3f}")
            lines.append("")

        # Top rasy
        if m.by_breed:
            lines.extend([
                "TOP 10 RAS",
                "-" * 40,
            ])
            for breed, count in list(m.by_breed.items())[:10]:
                percent = count / max(m.total_annotations, 1) * 100
                lines.append(f"  {breed[:20]:20s}: {count:5d} ({percent:5.1f}%)")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def save_metrics(self, output_file: Optional[Path] = None) -> Path:
        """
        Zapisuje metryki do JSON.

        Args:
            output_file: Ścieżka wyjściowa

        Returns:
            Ścieżka do pliku
        """
        if output_file is None:
            output_file = self.output_dir / "quality_metrics.json"

        data = {
            "generated": datetime.now().isoformat(),
            "source": str(self.annotations_path),
            "thresholds": {
                "min_bbox_confidence": self.thresholds.min_bbox_confidence,
                "min_breed_confidence": self.thresholds.min_breed_confidence,
                "min_emotion_confidence": self.thresholds.min_emotion_confidence,
                "low_confidence_flag": self.thresholds.low_confidence_flag,
            },
            "metrics": self.metrics.to_dict(),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Zapisano metryki do {output_file}")
        return output_file


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Monitorowanie jakości anotacji Dog FACS Dataset"
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/annotations.json"),
        help="Ścieżka do pliku anotacji COCO",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/frames"),
        help="Katalog z klatkami",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/quality"),
        help="Katalog wyjściowy",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generuj raport",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Twórz wizualizacje próbek",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Liczba próbek do wizualizacji (domyślnie: 10)",
    )
    parser.add_argument(
        "--export-flagged",
        action="store_true",
        help="Eksportuj listę flagowanych klatek",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Zapisz metryki do JSON",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.5,
        help="Próg niskiej pewności (domyślnie: 0.5)",
    )

    args = parser.parse_args()

    # Progi
    thresholds = QualityThresholds(
        low_confidence_flag=args.low_confidence_threshold,
    )

    # Monitor
    monitor = QualityMonitor(
        annotations_path=args.annotations,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        thresholds=thresholds,
    )

    # Analiza
    metrics = monitor.analyze()

    # Raport
    if args.report:
        print(monitor.generate_report())

    # Wizualizacje
    if args.visualize:
        monitor.visualize_sample(sample_size=args.sample)

    # Eksport flagowanych
    if args.export_flagged:
        monitor.export_flagged()

    # Zapisz metryki
    if args.save_metrics:
        monitor.save_metrics()

    # Domyślnie: pokaż raport
    if not any([args.report, args.visualize, args.export_flagged, args.save_metrics]):
        print(monitor.generate_report())


if __name__ == "__main__":
    main()
