"""Wspólne typy dla wielu źródeł wideo (TikTok, YouTube)."""

from dataclasses import dataclass


@dataclass
class VideoInfo:
    """
    Zwięzły opis wideo znalezionego przez wyszukiwanie (dowolne źródło).

    Attributes:
        video_id: ID wideo (unikalne w obrębie danej platformy)
        url: Pełny URL do wideo
        source_label: Hashtag/zapytanie, z którego wyszukiwania pochodzi wynik
        platform: Nazwa źródła ("tiktok" lub "youtube") - do prefiksu nazwy pliku
    """

    video_id: str
    url: str
    source_label: str
    platform: str
