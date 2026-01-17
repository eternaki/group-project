#!/usr/bin/env python3
"""
Skrypt do śledzenia postępu kolekcji wideo.

Funkcje:
- Generowanie raportów postępu
- Śledzenie wideo per emocja
- Monitorowanie różnorodności ras
- Eksport do CSV

Użycie:
    python scripts/download/collection_tracker.py --report
    python scripts/download/collection_tracker.py --export-csv progress.csv
"""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class CollectionTarget:
    """Cele kolekcji."""

    total_videos: int = 2500
    per_emotion_min: int = 400
    target_breeds: int = 20
    avg_duration_sec: int = 20


class CollectionTracker:
    """
    Klasa do śledzenia postępu kolekcji.

    Użycie:
        tracker = CollectionTracker()
        tracker.generate_report()
    """

    def __init__(
        self,
        metadata_file: Path = Path("data/collection/metadata.json"),
        progress_file: Path = Path("data/collection/progress.csv"),
    ) -> None:
        """
        Inicjalizuje tracker.

        Args:
            metadata_file: Ścieżka do pliku metadanych
            progress_file: Ścieżka do pliku postępu CSV
        """
        self.metadata_file = metadata_file
        self.progress_file = progress_file
        self.targets = CollectionTarget()
        self.metadata: list[dict] = []

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Wczytuje metadane."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.metadata = data.get("videos", [])
            except Exception as e:
                print(f"Błąd wczytywania metadanych: {e}")
                self.metadata = []

    def get_statistics(self) -> dict:
        """
        Oblicza statystyki kolekcji.

        Returns:
            Słownik ze statystykami
        """
        stats = {
            "total": len(self.metadata),
            "target": self.targets.total_videos,
            "progress_percent": 0,
            "by_emotion": {},
            "by_status": {},
            "total_duration_min": 0,
            "total_size_gb": 0,
            "avg_duration_sec": 0,
            "unique_channels": set(),
            "date_range": {"first": None, "last": None},
        }

        emotions = ["happy", "sad", "angry", "relaxed", "fearful", "neutral"]
        for emotion in emotions:
            stats["by_emotion"][emotion] = {
                "count": 0,
                "target": self.targets.per_emotion_min,
                "progress_percent": 0,
            }

        for video in self.metadata:
            emotion = video.get("emotion", "unknown")

            # Per emocja
            if emotion in stats["by_emotion"]:
                stats["by_emotion"][emotion]["count"] += 1

            # Per status
            status = video.get("status", "unknown")
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Sumy
            stats["total_duration_min"] += video.get("duration", 0) / 60
            stats["total_size_gb"] += video.get("file_size_mb", 0) / 1024

            # Kanały
            channel = video.get("channel", "")
            if channel:
                stats["unique_channels"].add(channel)

            # Daty
            download_date = video.get("download_date", "")
            if download_date:
                if stats["date_range"]["first"] is None or download_date < stats["date_range"]["first"]:
                    stats["date_range"]["first"] = download_date
                if stats["date_range"]["last"] is None or download_date > stats["date_range"]["last"]:
                    stats["date_range"]["last"] = download_date

        # Obliczenia końcowe
        if stats["total"] > 0:
            stats["progress_percent"] = round(
                stats["total"] / self.targets.total_videos * 100, 1
            )
            stats["avg_duration_sec"] = round(
                stats["total_duration_min"] * 60 / stats["total"], 1
            )

        for emotion in stats["by_emotion"]:
            count = stats["by_emotion"][emotion]["count"]
            target = stats["by_emotion"][emotion]["target"]
            if target > 0:
                stats["by_emotion"][emotion]["progress_percent"] = round(
                    count / target * 100, 1
                )

        stats["total_duration_min"] = round(stats["total_duration_min"], 1)
        stats["total_size_gb"] = round(stats["total_size_gb"], 2)
        stats["unique_channels"] = len(stats["unique_channels"])

        return stats

    def generate_report(self) -> str:
        """
        Generuje raport tekstowy.

        Returns:
            Raport jako string
        """
        stats = self.get_statistics()

        lines = [
            "=" * 60,
            "RAPORT POSTĘPU KOLEKCJI DOG FACS DATASET",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "OGÓLNE STATYSTYKI",
            "-" * 40,
            f"Łącznie wideo:     {stats['total']} / {stats['target']} ({stats['progress_percent']}%)",
            f"Czas trwania:      {stats['total_duration_min']} min",
            f"Rozmiar:           {stats['total_size_gb']} GB",
            f"Śr. długość:       {stats['avg_duration_sec']} sek",
            f"Unikalne kanały:   {stats['unique_channels']}",
            "",
            "POSTĘP PER EMOCJA",
            "-" * 40,
        ]

        # Progress bar
        def progress_bar(percent: float, width: int = 20) -> str:
            filled = int(width * percent / 100)
            empty = width - filled
            return f"[{'█' * filled}{'░' * empty}] {percent}%"

        for emotion, data in sorted(stats["by_emotion"].items()):
            bar = progress_bar(data["progress_percent"])
            lines.append(
                f"  {emotion:12s}: {data['count']:4d} / {data['target']:4d}  {bar}"
            )

        lines.extend([
            "",
            "STATUS WIDEO",
            "-" * 40,
        ])

        for status, count in sorted(stats["by_status"].items()):
            lines.append(f"  {status:15s}: {count}")

        if stats["date_range"]["first"]:
            lines.extend([
                "",
                "ZAKRES DAT",
                "-" * 40,
                f"  Pierwsze:  {stats['date_range']['first'][:10]}",
                f"  Ostatnie:  {stats['date_range']['last'][:10]}",
            ])

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def export_to_csv(self, output_path: Optional[Path] = None) -> Path:
        """
        Eksportuje dane do CSV.

        Args:
            output_path: Ścieżka wyjściowa

        Returns:
            Ścieżka do pliku CSV
        """
        if output_path is None:
            output_path = self.progress_file

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Nagłówki
            writer.writerow([
                "video_id",
                "title",
                "emotion",
                "duration",
                "status",
                "download_date",
                "channel",
                "file_size_mb",
            ])

            # Dane
            for video in self.metadata:
                writer.writerow([
                    video.get("video_id", ""),
                    video.get("title", ""),
                    video.get("emotion", ""),
                    video.get("duration", 0),
                    video.get("status", ""),
                    video.get("download_date", "")[:10] if video.get("download_date") else "",
                    video.get("channel", ""),
                    video.get("file_size_mb", 0),
                ])

        print(f"Wyeksportowano do: {output_path}")
        return output_path

    def export_daily_summary(self, output_path: Optional[Path] = None) -> Path:
        """
        Eksportuje dzienne podsumowanie do CSV.

        Args:
            output_path: Ścieżka wyjściowa

        Returns:
            Ścieżka do pliku CSV
        """
        if output_path is None:
            output_path = Path("data/collection/daily_summary.csv")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Grupuj per dzień
        daily_stats = {}

        for video in self.metadata:
            date = video.get("download_date", "")[:10]
            if not date:
                continue

            if date not in daily_stats:
                daily_stats[date] = {
                    "count": 0,
                    "duration_min": 0,
                    "size_gb": 0,
                    "emotions": {},
                }

            daily_stats[date]["count"] += 1
            daily_stats[date]["duration_min"] += video.get("duration", 0) / 60
            daily_stats[date]["size_gb"] += video.get("file_size_mb", 0) / 1024

            emotion = video.get("emotion", "unknown")
            daily_stats[date]["emotions"][emotion] = (
                daily_stats[date]["emotions"].get(emotion, 0) + 1
            )

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "date",
                "videos_count",
                "duration_min",
                "size_gb",
                "happy",
                "sad",
                "angry",
                "relaxed",
                "fearful",
                "neutral",
            ])

            for date in sorted(daily_stats.keys()):
                data = daily_stats[date]
                emotions = data["emotions"]

                writer.writerow([
                    date,
                    data["count"],
                    round(data["duration_min"], 1),
                    round(data["size_gb"], 3),
                    emotions.get("happy", 0),
                    emotions.get("sad", 0),
                    emotions.get("angry", 0),
                    emotions.get("relaxed", 0),
                    emotions.get("fearful", 0),
                    emotions.get("neutral", 0),
                ])

        print(f"Wyeksportowano podsumowanie dzienne do: {output_path}")
        return output_path

    def check_targets(self) -> dict:
        """
        Sprawdza osiągnięcie celów.

        Returns:
            Słownik z wynikami
        """
        stats = self.get_statistics()

        results = {
            "total_met": stats["total"] >= self.targets.total_videos,
            "emotions_met": {},
            "all_emotions_met": True,
        }

        for emotion, data in stats["by_emotion"].items():
            met = data["count"] >= self.targets.per_emotion_min
            results["emotions_met"][emotion] = met
            if not met:
                results["all_emotions_met"] = False

        return results


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Śledzenie postępu kolekcji Dog FACS Dataset"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generuj raport tekstowy",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        metavar="PATH",
        help="Eksportuj do CSV",
    )
    parser.add_argument(
        "--daily-summary",
        action="store_true",
        help="Eksportuj dzienne podsumowanie",
    )
    parser.add_argument(
        "--check-targets",
        action="store_true",
        help="Sprawdź osiągnięcie celów",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("data/collection/metadata.json"),
        help="Ścieżka do pliku metadanych",
    )

    args = parser.parse_args()

    tracker = CollectionTracker(metadata_file=args.metadata_file)

    if args.report:
        print(tracker.generate_report())
        return

    if args.export_csv:
        tracker.export_to_csv(args.export_csv)
        return

    if args.daily_summary:
        tracker.export_daily_summary()
        return

    if args.check_targets:
        results = tracker.check_targets()
        print("\n=== SPRAWDZENIE CELÓW ===")
        print(f"Cel łączny (2500): {'✓' if results['total_met'] else '✗'}")
        print("\nPer emocja (min 400):")
        for emotion, met in results["emotions_met"].items():
            print(f"  {emotion}: {'✓' if met else '✗'}")
        return

    # Domyślnie: raport
    print(tracker.generate_report())


if __name__ == "__main__":
    main()
