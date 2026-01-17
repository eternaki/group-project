#!/usr/bin/env python3
"""
Skrypt do pobierania wideo z YouTube dla Dog FACS Dataset.

Używa yt-dlp do pobierania wideo na podstawie zapytań wyszukiwania.
Zapisuje metadane do pliku JSON i śledzi postęp.

Użycie:
    python scripts/download/download_videos.py --emotion happy --limit 10
    python scripts/download/download_videos.py --url "https://youtube.com/watch?v=..." --emotion happy
    python scripts/download/download_videos.py --search "happy dog playing" --emotion happy --limit 5
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
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
class VideoMetadata:
    """Metadane pobranego wideo."""

    video_id: str
    title: str
    url: str
    duration: int  # sekundy
    emotion: str
    download_date: str
    file_path: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    file_size_mb: float = 0.0
    channel: str = ""
    upload_date: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    status: str = "downloaded"  # downloaded, processed, annotated, error


@dataclass
class DownloadConfig:
    """Konfiguracja pobierania."""

    output_dir: Path = field(default_factory=lambda: Path("data/raw"))
    metadata_file: Path = field(default_factory=lambda: Path("data/collection/metadata.json"))
    max_resolution: int = 720
    min_duration: int = 10
    max_duration: int = 60
    target_duration: int = 20


class VideoDownloader:
    """
    Klasa do pobierania wideo z YouTube.

    Użycie:
        downloader = VideoDownloader(config)
        metadata = downloader.download_video(url, emotion="happy")
    """

    def __init__(self, config: DownloadConfig) -> None:
        """
        Inicjalizuje downloader.

        Args:
            config: Konfiguracja pobierania
        """
        self.config = config
        self.metadata_list: list[VideoMetadata] = []

        # Utwórz katalogi
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Wczytaj istniejące metadane
        self._load_existing_metadata()

    def _load_existing_metadata(self) -> None:
        """Wczytuje istniejące metadane z pliku."""
        if self.config.metadata_file.exists():
            try:
                with open(self.config.metadata_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.metadata_list = [
                        VideoMetadata(**item) for item in data.get("videos", [])
                    ]
                logger.info(f"Wczytano {len(self.metadata_list)} istniejących wpisów")
            except Exception as e:
                logger.warning(f"Nie można wczytać metadanych: {e}")
                self.metadata_list = []

    def _save_metadata(self) -> None:
        """Zapisuje metadane do pliku."""
        data = {
            "last_updated": datetime.now().isoformat(),
            "total_videos": len(self.metadata_list),
            "videos": [asdict(m) for m in self.metadata_list],
        }

        with open(self.config.metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Zapisano metadane: {self.config.metadata_file}")

    def _get_downloaded_ids(self) -> set[str]:
        """Zwraca zbiór ID już pobranych wideo."""
        return {m.video_id for m in self.metadata_list}

    def download_video(
        self,
        url: str,
        emotion: str,
        force: bool = False,
    ) -> Optional[VideoMetadata]:
        """
        Pobiera pojedyncze wideo.

        Args:
            url: URL wideo YouTube
            emotion: Kategoria emocji
            force: Pobierz nawet jeśli już istnieje

        Returns:
            VideoMetadata lub None w przypadku błędu
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp nie jest zainstalowane. Uruchom: pip install yt-dlp")
            return None

        # Pobierz informacje o wideo
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            logger.error(f"Nie można pobrać informacji o wideo: {e}")
            return None

        video_id = info.get("id", "")

        # Sprawdź czy już pobrane
        if video_id in self._get_downloaded_ids() and not force:
            logger.info(f"Wideo {video_id} już pobrane, pomijam")
            return None

        # Sprawdź czas trwania
        duration = info.get("duration", 0)
        if duration < self.config.min_duration:
            logger.warning(f"Wideo {video_id} za krótkie ({duration}s), pomijam")
            return None
        if duration > self.config.max_duration:
            logger.warning(f"Wideo {video_id} za długie ({duration}s), pomijam")
            return None

        # Katalog dla emocji
        emotion_dir = self.config.output_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)

        # Opcje pobierania
        output_template = str(emotion_dir / "%(id)s.%(ext)s")
        ydl_opts = {
            "format": f"best[height<={self.config.max_resolution}]",
            "outtmpl": output_template,
            "quiet": False,
            "no_warnings": True,
            "progress_hooks": [self._progress_hook],
        }

        # Pobierz wideo
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            logger.error(f"Błąd pobierania {video_id}: {e}")
            return None

        # Znajdź pobrany plik
        downloaded_files = list(emotion_dir.glob(f"{video_id}.*"))
        if not downloaded_files:
            logger.error(f"Nie znaleziono pobranego pliku dla {video_id}")
            return None

        file_path = downloaded_files[0]
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Utwórz metadane
        metadata = VideoMetadata(
            video_id=video_id,
            title=info.get("title", ""),
            url=url,
            duration=duration,
            emotion=emotion,
            download_date=datetime.now().isoformat(),
            file_path=str(file_path),
            width=info.get("width", 0),
            height=info.get("height", 0),
            fps=info.get("fps", 0.0),
            file_size_mb=round(file_size_mb, 2),
            channel=info.get("channel", ""),
            upload_date=info.get("upload_date", ""),
            description=info.get("description", "")[:500],
            tags=info.get("tags", [])[:10],
            status="downloaded",
        )

        self.metadata_list.append(metadata)
        self._save_metadata()

        logger.info(f"Pobrano: {video_id} ({duration}s) -> {emotion}")
        return metadata

    def search_and_download(
        self,
        query: str,
        emotion: str,
        limit: int = 10,
    ) -> list[VideoMetadata]:
        """
        Wyszukuje i pobiera wideo.

        Args:
            query: Zapytanie wyszukiwania
            emotion: Kategoria emocji
            limit: Maksymalna liczba wideo do pobrania

        Returns:
            Lista pobranych metadanych
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp nie jest zainstalowane")
            return []

        logger.info(f"Wyszukiwanie: '{query}' (limit: {limit})")

        # Wyszukaj wideo
        search_opts = {
            "quiet": True,
            "extract_flat": True,
            "default_search": f"ytsearch{limit * 2}",
        }

        try:
            with yt_dlp.YoutubeDL(search_opts) as ydl:
                results = ydl.extract_info(query, download=False)
        except Exception as e:
            logger.error(f"Błąd wyszukiwania: {e}")
            return []

        entries = results.get("entries", [])
        logger.info(f"Znaleziono {len(entries)} wyników")

        # Pobierz wideo
        downloaded = []
        for entry in entries:
            if len(downloaded) >= limit:
                break

            url = entry.get("url") or f"https://youtube.com/watch?v={entry.get('id')}"
            metadata = self.download_video(url, emotion)

            if metadata:
                downloaded.append(metadata)

        logger.info(f"Pobrano {len(downloaded)} wideo dla '{query}'")
        return downloaded

    def _progress_hook(self, d: dict) -> None:
        """Hook do wyświetlania postępu."""
        if d["status"] == "downloading":
            percent = d.get("_percent_str", "?%")
            speed = d.get("_speed_str", "?")
            logger.debug(f"Pobieranie: {percent} @ {speed}")
        elif d["status"] == "finished":
            logger.debug("Pobieranie zakończone")

    def get_statistics(self) -> dict:
        """
        Zwraca statystyki kolekcji.

        Returns:
            Słownik ze statystykami
        """
        stats = {
            "total": len(self.metadata_list),
            "by_emotion": {},
            "by_status": {},
            "total_duration_min": 0,
            "total_size_gb": 0,
        }

        for m in self.metadata_list:
            # Per emocja
            stats["by_emotion"][m.emotion] = stats["by_emotion"].get(m.emotion, 0) + 1

            # Per status
            stats["by_status"][m.status] = stats["by_status"].get(m.status, 0) + 1

            # Sumy
            stats["total_duration_min"] += m.duration / 60
            stats["total_size_gb"] += m.file_size_mb / 1024

        stats["total_duration_min"] = round(stats["total_duration_min"], 1)
        stats["total_size_gb"] = round(stats["total_size_gb"], 2)

        return stats


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Pobieranie wideo z YouTube dla Dog FACS Dataset"
    )

    parser.add_argument(
        "--url",
        type=str,
        help="URL pojedynczego wideo do pobrania",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Zapytanie wyszukiwania",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        required=True,
        choices=["happy", "sad", "angry", "relaxed", "fearful", "neutral"],
        help="Kategoria emocji",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit wideo do pobrania (domyślnie: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Katalog wyjściowy",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Pokaż statystyki i wyjdź",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pobierz nawet jeśli już istnieje",
    )

    args = parser.parse_args()

    # Konfiguracja
    config = DownloadConfig(output_dir=args.output_dir)
    downloader = VideoDownloader(config)

    # Tylko statystyki
    if args.stats:
        stats = downloader.get_statistics()
        print("\n=== STATYSTYKI KOLEKCJI ===")
        print(f"Łącznie wideo: {stats['total']}")
        print(f"Czas trwania: {stats['total_duration_min']} min")
        print(f"Rozmiar: {stats['total_size_gb']} GB")
        print("\nPer emocja:")
        for emotion, count in sorted(stats["by_emotion"].items()):
            print(f"  {emotion}: {count}")
        return

    # Pobierz pojedyncze wideo
    if args.url:
        metadata = downloader.download_video(args.url, args.emotion, force=args.force)
        if metadata:
            print(f"\nPobrano: {metadata.title}")
            print(f"  ID: {metadata.video_id}")
            print(f"  Czas: {metadata.duration}s")
            print(f"  Plik: {metadata.file_path}")
        return

    # Wyszukaj i pobierz
    if args.search:
        downloaded = downloader.search_and_download(
            args.search, args.emotion, args.limit
        )
        print(f"\nPobrano {len(downloaded)} wideo")
        return

    # Bez argumentów - pokaż pomoc
    parser.print_help()


if __name__ == "__main__":
    main()
