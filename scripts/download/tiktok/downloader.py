"""
Pobieranie pojedynczych wideo TikTok przez yt-dlp.
"""

from dataclasses import dataclass
from pathlib import Path

from scripts.download.tiktok.config import MAX_VIDEO_DURATION_SECONDS


@dataclass
class TikTokVideoMetadata:
    """Metadane wideo pobrane bez ściągania pliku (do filtrowania przed pobraniem)."""

    duration: float
    description: str


@dataclass
class TikTokDownloadResult:
    """Wynik pobrania wideo TikTok."""

    success: bool
    path: Path | None
    duration: float
    error: str | None = None


def get_tiktok_video_metadata(url: str) -> TikTokVideoMetadata | None:
    """
    Pobiera metadane wideo (czas trwania, opis) bez ściągania pliku.

    Args:
        url: URL wideo TikTok

    Returns:
        TikTokVideoMetadata lub None, jeśli nie udało się pobrać informacji
    """
    import yt_dlp

    ydl_opts = {"quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return TikTokVideoMetadata(
                duration=info.get("duration", 0) or 0,
                description=info.get("description", "") or "",
            )
    except Exception:
        return None


def download_tiktok_video(url: str, output_dir: Path) -> TikTokDownloadResult:
    """
    Pobiera pojedyncze wideo TikTok pod wskazany katalog.

    Odrzuca wideo dłuższe niż MAX_VIDEO_DURATION_SECONDS bez pobierania pliku.

    Args:
        url: URL wideo TikTok
        output_dir: Katalog docelowy na pobrany plik

    Returns:
        TikTokDownloadResult z wynikiem operacji
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "format": "best",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration", 0) or 0

            if duration > MAX_VIDEO_DURATION_SECONDS:
                return TikTokDownloadResult(
                    success=False,
                    path=None,
                    duration=duration,
                    error=f"Wideo za długie ({duration:.0f}s)",
                )

            result_info = ydl.extract_info(url, download=True)
            video_id = result_info.get("id", "video")
            ext = result_info.get("ext", "mp4")

            return TikTokDownloadResult(
                success=True,
                path=output_dir / f"{video_id}.{ext}",
                duration=duration,
            )

    except Exception as e:
        return TikTokDownloadResult(
            success=False,
            path=None,
            duration=0,
            error=str(e),
        )
