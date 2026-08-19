"""
Orkiestrator kolektora wideo z psami z TikToka, z sortowaniem po emocjach.

Przepływ dla każdego hashtagu: wyszukaj -> odsiej AI-caption -> pobierz ->
sklasyfikuj emocję (odrzuca też wideo bez wyraźnie widocznej mordy) -> wyślij
na Google Drive do podfolderu danej emocji -> usuń lokalną kopię.

Uruchomienie:
    python -m scripts.download.tiktok.collect

Wymaga zmiennej środowiskowej TIKTOK_MS_TOKEN oraz pliku
secrets/gdrive_credentials.json (patrz README w tym katalogu).

Stan (już przetworzone ID wideo, liczniki per emocja) jest zapisywany do
data/tiktok_state.json, więc przerwanie i ponowne uruchomienie nie duplikuje
pracy.
"""

import asyncio
import json
import random
import time

from packages.data.schemas import EMOTION_CLASSES
from scripts.download.tiktok.config import (
    DOWNLOAD_TMP_DIR,
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_FOLDER_ID,
    GDRIVE_TOKEN_PATH,
    MAX_CONSECUTIVE_ERRORS,
    MAX_REQUEST_DELAY_SECONDS,
    MIN_REQUEST_DELAY_SECONDS,
    SOFTBAN_COOLDOWN_SECONDS,
    STATE_FILE,
    VIDEOS_PER_HASHTAG_PER_ROUND,
    CollectorConfig,
)
from scripts.download.tiktok.content_filters import AiCaptionFilter
from scripts.download.tiktok.downloader import download_tiktok_video, get_tiktok_video_metadata
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader
from scripts.download.tiktok.emotion_classifier import VideoEmotionClassifier
from scripts.download.tiktok.hashtag_search import TikTokHashtagSearcher


class CollectorState:
    """Persystencja przetworzonych ID wideo i liczników zaakceptowanych per emocja."""

    def __init__(self, state_path) -> None:
        self.state_path = state_path
        self.processed_ids: set[str] = set()
        self.emotion_counts: dict[str, int] = {emotion: 0 for emotion in EMOTION_CLASSES}
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.processed_ids = set(data.get("processed_ids", []))
            saved_counts = data.get("emotion_counts", {})
            for emotion in EMOTION_CLASSES:
                self.emotion_counts[emotion] = saved_counts.get(emotion, 0)

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "processed_ids": sorted(self.processed_ids),
                    "emotion_counts": self.emotion_counts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def mark_processed(self, video_id: str) -> None:
        self.processed_ids.add(video_id)

    def is_processed(self, video_id: str) -> bool:
        return video_id in self.processed_ids

    def is_complete(self, target_per_emotion: int) -> bool:
        return all(count >= target_per_emotion for count in self.emotion_counts.values())

    def needs_more(self, emotion: str, target_per_emotion: int) -> bool:
        return self.emotion_counts[emotion] < target_per_emotion


async def run_collection(config: CollectorConfig) -> None:
    """Uruchamia pełny cykl zbierania aż wszystkie 9 emocji osiągną target_per_emotion."""
    state = CollectorState(STATE_FILE)

    caption_filter = AiCaptionFilter()

    emotion_classifier = VideoEmotionClassifier(device=config.device)
    emotion_classifier.load()

    uploader = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
    uploader.authenticate()

    print("Przygotowuję podfoldery emocji na Google Drive...")
    emotion_folder_ids = {
        emotion: uploader.ensure_folder(emotion, GDRIVE_FOLDER_ID) for emotion in EMOTION_CLASSES
    }

    consecutive_errors = 0

    async with TikTokHashtagSearcher(config.ms_token) as searcher:
        while not state.is_complete(config.target_per_emotion):
            for hashtag in config.hashtags:
                if state.is_complete(config.target_per_emotion):
                    break

                try:
                    async for video in searcher.search(hashtag, VIDEOS_PER_HASHTAG_PER_ROUND):
                        if state.is_complete(config.target_per_emotion):
                            break
                        if state.is_processed(video.video_id):
                            continue

                        _throttle()

                        emotion = _process_video(
                            video, state, config, caption_filter,
                            emotion_classifier, uploader, emotion_folder_ids,
                        )
                        state.mark_processed(video.video_id)
                        if emotion is not None:
                            state.emotion_counts[emotion] += 1
                            counts_str = ", ".join(
                                f"{e}={state.emotion_counts[e]}/{config.target_per_emotion}"
                                for e in EMOTION_CLASSES
                            )
                            print(f"Przyjęto [{emotion}]: {video.url}\n  {counts_str}")
                        state.save()
                        consecutive_errors = 0

                except Exception as e:
                    consecutive_errors += 1
                    print(f"Błąd przy hashtagu #{hashtag}: {e}")
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(
                            f"Zbyt wiele błędów z rzędu - pauza {SOFTBAN_COOLDOWN_SECONDS}s "
                            "(prawdopodobny rate-limit)."
                        )
                        time.sleep(SOFTBAN_COOLDOWN_SECONDS)
                        consecutive_errors = 0

    print(f"Zakończono: wszystkie emocje osiągnęły {config.target_per_emotion} wideo.")


def _process_video(
    video,
    state: CollectorState,
    config: CollectorConfig,
    caption_filter: AiCaptionFilter,
    emotion_classifier: VideoEmotionClassifier,
    uploader: GoogleDriveUploader,
    emotion_folder_ids: dict[str, str],
) -> str | None:
    """
    Sprawdza metadane, pobiera, klasyfikuje emocję i wysyła wideo.

    Returns:
        Nazwa przypisanej emocji, jeśli wideo zostało zaakceptowane i wysłane; inaczej None
    """
    metadata = get_tiktok_video_metadata(video.url)
    if metadata is None:
        return None
    if caption_filter.is_likely_ai_generated(metadata.description, []):
        return None

    result = download_tiktok_video(video.url, DOWNLOAD_TMP_DIR)
    if not result.success or result.path is None:
        return None

    try:
        emotion = emotion_classifier.classify_video(result.path)
        if emotion is None:
            return None
        if not state.needs_more(emotion, config.target_per_emotion):
            return None

        uploader.upload_file(
            result.path,
            remote_name=f"{video.source_hashtag}_{video.video_id}.mp4",
            folder_id=emotion_folder_ids[emotion],
        )
        return emotion
    finally:
        result.path.unlink(missing_ok=True)


def _throttle() -> None:
    """Losowe opóźnienie między żądaniami, żeby ograniczyć ryzyko rate-limitu."""
    time.sleep(random.uniform(MIN_REQUEST_DELAY_SECONDS, MAX_REQUEST_DELAY_SECONDS))


if __name__ == "__main__":
    asyncio.run(run_collection(CollectorConfig()))
