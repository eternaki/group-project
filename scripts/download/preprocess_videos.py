#!/usr/bin/env python3
"""
Skrypt do preprocessingu pobranych wideo.

Funkcje:
- Przycinanie wideo do określonej długości
- Standaryzacja rozdzielczości
- Usuwanie uszkodzonych plików
- Organizacja w katalogi per emocja

Użycie:
    python scripts/download/preprocess_videos.py --input-dir data/raw --output-dir data/processed
    python scripts/download/preprocess_videos.py --validate-only
"""

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Konfiguracja preprocessingu."""

    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/processed")
    target_duration: int = 20  # sekundy
    max_resolution: int = 720
    output_format: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"


class VideoPreprocessor:
    """
    Klasa do preprocessingu wideo.

    Użycie:
        preprocessor = VideoPreprocessor(config)
        preprocessor.process_all()
    """

    def __init__(self, config: PreprocessConfig) -> None:
        """
        Inicjalizuje preprocessor.

        Args:
            config: Konfiguracja preprocessingu
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def get_video_info(self, video_path: Path) -> Optional[dict]:
        """
        Pobiera informacje o wideo.

        Args:
            video_path: Ścieżka do pliku wideo

        Returns:
            Słownik z informacjami lub None
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                return None

            info = {
                "path": str(video_path),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": 0,
            }

            if info["fps"] > 0:
                info["duration"] = info["frame_count"] / info["fps"]

            cap.release()
            return info

        except Exception as e:
            logger.error(f"Błąd odczytu {video_path}: {e}")
            return None

    def validate_video(self, video_path: Path) -> tuple[bool, str]:
        """
        Waliduje plik wideo.

        Args:
            video_path: Ścieżka do pliku

        Returns:
            Tuple (czy_poprawny, komunikat)
        """
        if not video_path.exists():
            return False, "Plik nie istnieje"

        if video_path.stat().st_size == 0:
            return False, "Plik jest pusty"

        info = self.get_video_info(video_path)
        if info is None:
            return False, "Nie można odczytać wideo"

        if info["duration"] < 5:
            return False, f"Za krótkie ({info['duration']:.1f}s)"

        if info["width"] == 0 or info["height"] == 0:
            return False, "Nieprawidłowe wymiary"

        return True, "OK"

    def trim_video(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float = 0,
        duration: Optional[float] = None,
    ) -> bool:
        """
        Przycina wideo do określonej długości.

        Args:
            input_path: Ścieżka wejściowa
            output_path: Ścieżka wyjściowa
            start_time: Czas startu (sekundy)
            duration: Długość (sekundy)

        Returns:
            True jeśli sukces
        """
        if duration is None:
            duration = self.config.target_duration

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ss", str(start_time),
            "-t", str(duration),
            "-c:v", self.config.video_codec,
            "-c:a", self.config.audio_codec,
            "-vf", f"scale=-2:{self.config.max_resolution}",
            "-preset", "fast",
            "-crf", "23",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout podczas przetwarzania {input_path}")
            return False
        except FileNotFoundError:
            logger.error("FFmpeg nie jest zainstalowane")
            return False
        except Exception as e:
            logger.error(f"Błąd przetwarzania {input_path}: {e}")
            return False

    def process_video(self, video_path: Path, emotion: str) -> Optional[Path]:
        """
        Przetwarza pojedyncze wideo.

        Args:
            video_path: Ścieżka do wideo
            emotion: Kategoria emocji

        Returns:
            Ścieżka do przetworzonego pliku lub None
        """
        # Walidacja
        is_valid, message = self.validate_video(video_path)
        if not is_valid:
            logger.warning(f"Pomijam {video_path.name}: {message}")
            return None

        # Info o wideo
        info = self.get_video_info(video_path)
        if info is None:
            return None

        # Oblicz czas startu (środek wideo)
        duration = info["duration"]
        if duration > self.config.target_duration:
            start_time = (duration - self.config.target_duration) / 2
        else:
            start_time = 0

        # Ścieżka wyjściowa
        output_dir = self.config.output_dir / emotion
        output_path = output_dir / f"{video_path.stem}.{self.config.output_format}"

        # Przycięcie
        success = self.trim_video(video_path, output_path, start_time)

        if success and output_path.exists():
            logger.info(f"Przetworzono: {video_path.name} -> {output_path.name}")
            return output_path

        return None

    def process_all(self) -> dict:
        """
        Przetwarza wszystkie wideo z input_dir.

        Returns:
            Słownik ze statystykami
        """
        stats = {
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "by_emotion": {},
        }

        # Znajdź wszystkie wideo
        video_extensions = [".mp4", ".webm", ".mkv", ".avi", ".mov"]

        for emotion_dir in self.config.input_dir.iterdir():
            if not emotion_dir.is_dir():
                continue

            emotion = emotion_dir.name
            stats["by_emotion"][emotion] = {"processed": 0, "skipped": 0, "errors": 0}

            logger.info(f"\n=== Przetwarzanie: {emotion} ===")

            for video_file in emotion_dir.iterdir():
                if video_file.suffix.lower() not in video_extensions:
                    continue

                stats["total"] += 1

                result = self.process_video(video_file, emotion)

                if result:
                    stats["processed"] += 1
                    stats["by_emotion"][emotion]["processed"] += 1
                else:
                    stats["skipped"] += 1
                    stats["by_emotion"][emotion]["skipped"] += 1

        return stats

    def validate_all(self) -> dict:
        """
        Waliduje wszystkie wideo bez przetwarzania.

        Returns:
            Słownik z wynikami walidacji
        """
        results = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "issues": [],
        }

        video_extensions = [".mp4", ".webm", ".mkv", ".avi", ".mov"]

        for emotion_dir in self.config.input_dir.iterdir():
            if not emotion_dir.is_dir():
                continue

            for video_file in emotion_dir.iterdir():
                if video_file.suffix.lower() not in video_extensions:
                    continue

                results["total"] += 1

                is_valid, message = self.validate_video(video_file)

                if is_valid:
                    results["valid"] += 1
                else:
                    results["invalid"] += 1
                    results["issues"].append({
                        "file": str(video_file),
                        "issue": message,
                    })

        return results

    def cleanup_invalid(self, dry_run: bool = True) -> list[Path]:
        """
        Usuwa nieprawidłowe pliki wideo.

        Args:
            dry_run: Jeśli True, tylko pokazuje co byłoby usunięte

        Returns:
            Lista usuniętych/do usunięcia plików
        """
        to_remove = []
        video_extensions = [".mp4", ".webm", ".mkv", ".avi", ".mov"]

        for emotion_dir in self.config.input_dir.iterdir():
            if not emotion_dir.is_dir():
                continue

            for video_file in emotion_dir.iterdir():
                if video_file.suffix.lower() not in video_extensions:
                    continue

                is_valid, message = self.validate_video(video_file)

                if not is_valid:
                    to_remove.append(video_file)

                    if dry_run:
                        logger.info(f"[DRY RUN] Usunąć: {video_file} ({message})")
                    else:
                        logger.info(f"Usuwam: {video_file} ({message})")
                        video_file.unlink()

        return to_remove


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Preprocessing wideo dla Dog FACS Dataset"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Katalog z pobranymi wideo",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Katalog wyjściowy",
    )
    parser.add_argument(
        "--target-duration",
        type=int,
        default=20,
        help="Docelowy czas trwania (sekundy)",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=720,
        help="Maksymalna rozdzielczość",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Tylko walidacja, bez przetwarzania",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Usuń nieprawidłowe pliki",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pokaż co byłoby usunięte (z --cleanup)",
    )

    args = parser.parse_args()

    config = PreprocessConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_duration=args.target_duration,
        max_resolution=args.max_resolution,
    )

    preprocessor = VideoPreprocessor(config)

    # Tylko walidacja
    if args.validate_only:
        print("\n=== WALIDACJA WIDEO ===")
        results = preprocessor.validate_all()
        print(f"Łącznie: {results['total']}")
        print(f"Poprawne: {results['valid']}")
        print(f"Nieprawidłowe: {results['invalid']}")

        if results["issues"]:
            print("\nProblemy:")
            for issue in results["issues"][:10]:
                print(f"  - {Path(issue['file']).name}: {issue['issue']}")
            if len(results["issues"]) > 10:
                print(f"  ... i {len(results['issues']) - 10} więcej")
        return

    # Czyszczenie
    if args.cleanup:
        print("\n=== CZYSZCZENIE NIEPRAWIDŁOWYCH PLIKÓW ===")
        removed = preprocessor.cleanup_invalid(dry_run=args.dry_run)
        print(f"{'Do usunięcia' if args.dry_run else 'Usunięto'}: {len(removed)} plików")
        return

    # Przetwarzanie
    print("\n=== PREPROCESSING WIDEO ===")
    stats = preprocessor.process_all()

    print("\n=== PODSUMOWANIE ===")
    print(f"Łącznie: {stats['total']}")
    print(f"Przetworzono: {stats['processed']}")
    print(f"Pominięto: {stats['skipped']}")

    print("\nPer emocja:")
    for emotion, data in stats["by_emotion"].items():
        print(f"  {emotion}: {data['processed']} przetworzonych, {data['skipped']} pominiętych")


if __name__ == "__main__":
    main()
