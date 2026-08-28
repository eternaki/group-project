"""
Wyszukiwanie wideo YouTube (Shorts i zwykłe) po zapytaniu tekstowym.

Preferuje oficjalne YouTube Data API v3 (search.list) - yt-dlp scrapowany
ytsearch zaczął dostawać HTTP 403 (rate-limit na scraping wyszukiwania).
Pobieranie samego pliku wideo dalej idzie przez yt-dlp (downloader.py) -
to osobna operacja, której ten rate-limit nie dotyczył.

Bez YOUTUBE_API_KEY (zmienna środowiskowa) wraca do yt-dlp ytsearch jako
fallback, żeby kolektor dalej działał (wolniej/mniej niezawodnie).
"""

from scripts.download.tiktok.common import VideoInfo
from scripts.download.tiktok.config import YOUTUBE_API_KEY


class YouTubeSearcher:
    """
    Wyszukuje wideo YouTube po zapytaniu tekstowym.

    Użycie:
        searcher = YouTubeSearcher()
        async for video in searcher.search("dog reaction funny face", count=30):
            print(video.url)
    """

    def __init__(self) -> None:
        # Ustawiane raz na cały przebieg, gdy dzienna kwota API się wyczerpie -
        # unika powtarzania tego samego kosztownego wywołania (i błędu) co rundę.
        self._api_quota_exhausted = False

    async def search(self, query: str, count: int):
        """
        Wyszukuje wideo na YouTube dla danego zapytania.

        Args:
            query: Zapytanie tekstowe
            count: Maksymalna liczba wyników

        Yields:
            VideoInfo dla każdego znalezionego wideo
        """
        if YOUTUBE_API_KEY and not self._api_quota_exhausted:
            try:
                async for video in self._search_via_api(query, count):
                    yield video
                return
            except Exception as e:
                if not self._is_quota_error(e):
                    raise
                print(
                    "Dzienna kwota YouTube Data API wyczerpana - przełączam się "
                    "na scrapowany yt-dlp ytsearch do końca tego przebiegu."
                )
                self._api_quota_exhausted = True

        async for video in self._search_via_yt_dlp(query, count):
            yield video

    @staticmethod
    def _is_quota_error(error: Exception) -> bool:
        """Rozpoznaje wyczerpanie dziennej kwoty (HTTP 429 / rateLimitExceeded)."""
        status = getattr(getattr(error, "resp", None), "status", None)
        return status == 429 or "rateLimitExceeded" in str(error)

    @staticmethod
    def _shorts_biased(query: str) -> str:
        """
        Dopisuje "shorts" do zapytania - zmierzone: bez tego ~87% wyników to
        długie kompilacje/reakcje (100-700s), odrzucane potem przez limit
        długości wideo. Sam filtr videoDuration="short" w API (<4 min) nie
        wystarczał.
        """
        return f"{query} shorts"

    async def _search_via_api(self, query: str, count: int):
        """Wyszukuje przez oficjalne YouTube Data API v3 (bez captchy, bez scrapingu)."""
        from googleapiclient.discovery import build

        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            q=self._shorts_biased(query),
            part="id",
            type="video",
            videoDuration="short",  # <4 min - odsiewa długie kompilacje wcześniej
            maxResults=min(count, 50),  # limit API na jedno zapytanie
        )
        response = request.execute()

        for item in response.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                continue
            yield VideoInfo(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                source_label=query,
                platform="youtube",
            )

    async def _search_via_yt_dlp(self, query: str, count: int):
        """
        Fallback: scrapowany ytsearch yt-dlp (bez klucza API).

        Próba użycia strony wyników z parametrem "sp" (filtr długości) dawała
        gorsze wyniki niż zwykły ytsearch - w teście zwracała martwe/niedostępne
        wideo. Filtr długości sprawdzamy więc dopiero po pobraniu metadanych
        (jak wcześniej); "shorts" w zapytaniu (_shorts_biased) i tak podnosi
        trafność bez tego dodatkowego parametru.
        """
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "socket_timeout": 20,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{count}:{self._shorts_biased(query)}", download=False)

        for entry in (info or {}).get("entries", []) or []:
            if entry is None or not entry.get("id"):
                continue
            yield VideoInfo(
                video_id=entry["id"],
                url=f"https://www.youtube.com/watch?v={entry['id']}",
                source_label=query,
                platform="youtube",
            )
