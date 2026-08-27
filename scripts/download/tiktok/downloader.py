"""
Pobieranie pojedynczych wideo (TikTok, YouTube) przez yt-dlp.
"""

from dataclasses import dataclass
from pathlib import Path

from scripts.download.tiktok.config import MAX_VIDEO_DURATION_SECONDS


@dataclass
class VideoMetadata:
    """Metadane wideo pobrane bez ściągania pliku (do filtrowania przed pobraniem)."""

    duration: float
    description: str
    title: str = ""


@dataclass
class VideoDownloadResult:
    """Wynik pobrania wideo."""

    success: bool
    path: Path | None
    duration: float
    error: str | None = None


def get_video_metadata(url: str) -> VideoMetadata | None:
    """
    Pobiera metadane wideo (czas trwania, opis) bez ściągania pliku.

    Args:
        url: URL wideo (dowolna platforma obsługiwana przez yt-dlp)

    Returns:
        VideoMetadata lub None, jeśli nie udało się pobrać informacji
    """
    import yt_dlp

    ydl_opts = {"quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return VideoMetadata(
                duration=info.get("duration", 0) or 0,
                description=info.get("description", "") or "",
                title=info.get("title", "") or "",
            )
    except Exception:
        return None


def download_video(url: str, output_dir: Path) -> VideoDownloadResult:
    """
    Pobiera pojedyncze wideo pod wskazany katalog.

    Odrzuca wideo dłuższe niż MAX_VIDEO_DURATION_SECONDS bez pobierania pliku.

    Args:
        url: URL wideo (dowolna platforma obsługiwana przez yt-dlp)
        output_dir: Katalog docelowy na pobrany plik

    Returns:
        VideoDownloadResult z wynikiem operacji
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sprawdzamy długość BEZ wymuszania formatu - "format" na etapie samego
    # sprawdzania potrafi wywalić się z błędem na YouTube (adaptacyjne strumienie
    # wideo/audio bez wspólnego "best"), przez co odrzucenie zbyt długiego wideo
    # nigdy nie było osiągane - leciał od razu wyjątek pobierania.
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as probe:
            info = probe.extract_info(url, download=False)
    except Exception as e:
        return VideoDownloadResult(success=False, path=None, duration=0, error=str(e))

    duration = info.get("duration", 0) or 0
    if duration > MAX_VIDEO_DURATION_SECONDS:
        return VideoDownloadResult(
            success=False,
            path=None,
            duration=duration,
            error=f"Wideo za długie ({duration:.0f}s)",
        )

    import imageio_ffmpeg

    ydl_opts = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        # bestvideo+bestaudio wymaga scalenia - potrzebne, gdy platforma nie
        # oferuje pojedynczego strumienia "best" (typowe na YouTube). ffmpeg nie
        # jest zainstalowany systemowo, więc wskazujemy binarkę z imageio-ffmpeg
        # (pip, bez uprawnień administratora) zamiast polegać na PATH.
        "format": "bestvideo+bestaudio/best",
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result_info = ydl.extract_info(url, download=True)
            video_id = result_info.get("id", "video")
            ext = result_info.get("ext", "mp4")

            return VideoDownloadResult(
                success=True,
                path=output_dir / f"{video_id}.{ext}",
                duration=duration,
            )

    except Exception as e:
        return VideoDownloadResult(
            success=False,
            path=None,
            duration=0,
            error=str(e),
        )
