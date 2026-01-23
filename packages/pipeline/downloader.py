"""
Moduł do pobierania wideo z YouTube i innych źródeł.

Obsługuje pobieranie wideo po URL z walidacją czasu trwania
i rozdzielczości dla aplikacji demo.
"""

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests


@dataclass
class DownloadResult:
    """
    Wynik pobierania wideo.

    Attributes:
        success: Czy pobieranie się powiodło
        path: Ścieżka do pobranego pliku (jeśli sukces)
        title: Tytuł wideo
        duration: Czas trwania w sekundach
        error: Komunikat błędu (jeśli niepowodzenie)
    """

    success: bool
    path: Optional[Path]
    title: str
    duration: float
    error: Optional[str] = None


class YouTubeDownloader:
    """
    Klasa do pobierania wideo z YouTube i innych źródeł.

    Obsługuje:
    - YouTube (youtube.com, youtu.be)
    - Bezpośrednie linki do plików wideo (.mp4, .mov, .webm)

    Użycie:
        downloader = YouTubeDownloader(output_dir=Path("./temp"), max_duration=30)

        # Sprawdź info przed pobraniem
        info = downloader.get_video_info("https://youtube.com/watch?v=...")
        if info and info["duration"] <= 30:
            result = downloader.download("https://youtube.com/watch?v=...")
            if result.success:
                print(f"Pobrano: {result.path}")
    """

    # Wzorce URL dla YouTube
    YOUTUBE_PATTERNS = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]

    # Obsługiwane rozszerzenia dla bezpośrednich linków
    VIDEO_EXTENSIONS = [".mp4", ".mov", ".webm", ".avi", ".mkv"]

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        max_duration: int = 30,
        max_resolution: int = 720,
    ) -> None:
        """
        Inicjalizuje downloader.

        Args:
            output_dir: Katalog do zapisywania plików (domyślnie temp)
            max_duration: Maksymalny czas trwania w sekundach
            max_resolution: Maksymalna rozdzielczość (wysokość)
        """
        if output_dir is None:
            output_dir = Path(tempfile.gettempdir()) / "dog_facs_videos"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_duration = max_duration
        self.max_resolution = max_resolution

    def is_youtube_url(self, url: str) -> bool:
        """
        Sprawdza czy URL jest linkiem do YouTube.

        Args:
            url: URL do sprawdzenia

        Returns:
            True jeśli to YouTube URL
        """
        for pattern in self.YOUTUBE_PATTERNS:
            if re.match(pattern, url):
                return True
        return False

    def is_direct_video_url(self, url: str) -> bool:
        """
        Sprawdza czy URL jest bezpośrednim linkiem do pliku wideo.

        Args:
            url: URL do sprawdzenia

        Returns:
            True jeśli to bezpośredni link do wideo
        """
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        return any(path_lower.endswith(ext) for ext in self.VIDEO_EXTENSIONS)

    def get_video_info(self, url: str) -> Optional[dict]:
        """
        Pobiera informacje o wideo bez ściągania pliku.

        Args:
            url: URL wideo

        Returns:
            Słownik z informacjami lub None w przypadku błędu

        Info keys:
            - title: Tytuł wideo
            - duration: Czas trwania w sekundach
            - width: Szerokość
            - height: Wysokość
            - url: Oryginalny URL
        """
        if self.is_youtube_url(url):
            return self._get_youtube_info(url)
        elif self.is_direct_video_url(url):
            return self._get_direct_video_info(url)
        else:
            return None

    def _get_youtube_info(self, url: str) -> Optional[dict]:
        """Pobiera informacje o wideo z YouTube."""
        try:
            import yt_dlp

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                return {
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "width": info.get("width", 0),
                    "height": info.get("height", 0),
                    "url": url,
                    "source": "youtube",
                }

        except ImportError:
            return {"error": "yt-dlp nie jest zainstalowane"}
        except Exception as e:
            return {"error": str(e)}

    def _get_direct_video_info(self, url: str) -> Optional[dict]:
        """Pobiera informacje o bezpośrednim linku do wideo."""
        try:
            # HEAD request dla sprawdzenia dostępności
            response = requests.head(url, timeout=10, allow_redirects=True)

            if response.status_code != 200:
                return None

            # Wyciągnij nazwę pliku z URL
            parsed = urlparse(url)
            filename = Path(parsed.path).name

            return {
                "title": filename,
                "duration": 0,  # Nie możemy określić bez pobierania
                "width": 0,
                "height": 0,
                "url": url,
                "source": "direct",
            }

        except Exception:
            return None

    def download(self, url: str) -> DownloadResult:
        """
        Pobiera wideo z URL.

        Args:
            url: URL wideo (YouTube lub bezpośredni link)

        Returns:
            DownloadResult z wynikiem operacji
        """
        if self.is_youtube_url(url):
            return self._download_youtube(url)
        elif self.is_direct_video_url(url):
            return self._download_direct(url)
        else:
            return DownloadResult(
                success=False,
                path=None,
                title="",
                duration=0,
                error="Nieobsługiwany format URL. Użyj YouTube lub bezpośredniego linku do wideo.",
            )

    def _download_youtube(self, url: str) -> DownloadResult:
        """Pobiera wideo z YouTube."""
        try:
            import yt_dlp

            # Najpierw sprawdź czas trwania
            info = self._get_youtube_info(url)
            if info is None or "error" in info:
                return DownloadResult(
                    success=False,
                    path=None,
                    title="",
                    duration=0,
                    error=info.get("error", "Nie można pobrać informacji o wideo"),
                )

            duration = info.get("duration", 0)
            if duration > self.max_duration:
                return DownloadResult(
                    success=False,
                    path=None,
                    title=info.get("title", ""),
                    duration=duration,
                    error=f"Wideo za długie ({duration:.0f}s). Maksymalny czas: {self.max_duration}s",
                )

            # Pobierz wideo
            ydl_opts = {
                "format": f"best[height<={self.max_resolution}]",
                "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result_info = ydl.extract_info(url, download=True)
                video_id = result_info.get("id", "video")
                ext = result_info.get("ext", "mp4")
                output_path = self.output_dir / f"{video_id}.{ext}"

                return DownloadResult(
                    success=True,
                    path=output_path,
                    title=result_info.get("title", "Unknown"),
                    duration=result_info.get("duration", 0),
                )

        except ImportError:
            return DownloadResult(
                success=False,
                path=None,
                title="",
                duration=0,
                error="yt-dlp nie jest zainstalowane. Uruchom: pip install yt-dlp",
            )
        except Exception as e:
            return DownloadResult(
                success=False,
                path=None,
                title="",
                duration=0,
                error=f"Błąd pobierania: {str(e)}",
            )

    def _download_direct(self, url: str) -> DownloadResult:
        """Pobiera wideo z bezpośredniego linku."""
        try:
            # Wyciągnij nazwę pliku
            parsed = urlparse(url)
            filename = Path(parsed.path).name
            if not filename:
                filename = "video.mp4"

            output_path = self.output_dir / filename

            # Pobierz plik
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Sprawdź rozmiar (max 100MB)
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > 100 * 1024 * 1024:
                return DownloadResult(
                    success=False,
                    path=None,
                    title=filename,
                    duration=0,
                    error="Plik za duży (max 100MB)",
                )

            # Zapisz plik
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return DownloadResult(
                success=True,
                path=output_path,
                title=filename,
                duration=0,  # Nie znamy bez analizy
            )

        except requests.exceptions.Timeout:
            return DownloadResult(
                success=False,
                path=None,
                title="",
                duration=0,
                error="Timeout podczas pobierania",
            )
        except requests.exceptions.RequestException as e:
            return DownloadResult(
                success=False,
                path=None,
                title="",
                duration=0,
                error=f"Błąd pobierania: {str(e)}",
            )
        except Exception as e:
            return DownloadResult(
                success=False,
                path=None,
                title="",
                duration=0,
                error=f"Nieoczekiwany błąd: {str(e)}",
            )

    def cleanup(self, path: Path) -> None:
        """
        Usuwa pobrany plik.

        Args:
            path: Ścieżka do pliku
        """
        if path and path.exists():
            path.unlink(missing_ok=True)
