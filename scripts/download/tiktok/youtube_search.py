"""
Wyszukiwanie wideo YouTube (Shorts i zwykłe) po zapytaniu tekstowym.

W przeciwieństwie do TikToka, yt-dlp obsługuje wyszukiwanie YouTube wprost
(prefiks ytsearchN:) bez potrzeby przeglądarki i bez captchy.
"""

from scripts.download.tiktok.common import VideoInfo


class YouTubeSearcher:
    """
    Wyszukuje wideo YouTube po zapytaniu tekstowym przez yt-dlp.

    Użycie:
        searcher = YouTubeSearcher()
        async for video in searcher.search("dog reaction funny face", count=30):
            print(video.url)
    """

    async def search(self, query: str, count: int):
        """
        Wyszukuje wideo na YouTube dla danego zapytania.

        Args:
            query: Zapytanie tekstowe
            count: Maksymalna liczba wyników

        Yields:
            VideoInfo dla każdego znalezionego wideo
        """
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{count}:{query}", download=False)

        for entry in (info or {}).get("entries", []) or []:
            if entry is None or not entry.get("id"):
                continue
            yield VideoInfo(
                video_id=entry["id"],
                url=f"https://www.youtube.com/watch?v={entry['id']}",
                source_label=query,
                platform="youtube",
            )
