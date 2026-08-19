"""
Filtry treści dla kolektora TikTok.

AiCaptionFilter - heurystyka odsiewająca opisy oznaczone jako AI-generated.

Filtr widoczności mordy nie jest tu osobnym krokiem - VideoEmotionClassifier
(emotion_classifier.py) i tak wymaga pewnych keypointów twarzy, żeby wybrać
peak frames, więc wideo bez widocznej mordy jest odrzucane przy okazji
klasyfikacji emocji (bez dublowania tej samej detekcji dwa razy).
"""

from scripts.download.tiktok.config import AI_CONTENT_MARKERS


class AiCaptionFilter:
    """Odsiewa wideo, których opis/hashtagi wskazują na treść wygenerowaną przez AI."""

    def __init__(self, markers: list[str] | None = None) -> None:
        self.markers = [m.lower() for m in (markers or AI_CONTENT_MARKERS)]

    def is_likely_ai_generated(self, caption: str, hashtags: list[str]) -> bool:
        """
        Sprawdza opis i hashtagi pod kątem znaczników AI.

        Args:
            caption: Opis wideo
            hashtags: Lista hashtagów wideo (bez '#' lub z '#')

        Returns:
            True, jeśli wideo prawdopodobnie jest wygenerowane przez AI
        """
        haystack = caption.lower() + " " + " ".join(h.lower() for h in hashtags)
        return any(marker in haystack for marker in self.markers)
