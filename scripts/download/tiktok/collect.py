"""
Orkiestrator kolektora wideo z psami (TikTok i/lub YouTube), z sortowaniem
po emocjach.

Przepływ dla każdego zapytania/hashtagu: wyszukaj -> odsiej AI-caption ->
pobierz -> sklasyfikuj emocję (odrzuca też wideo bez wyraźnie widocznej mordy)
-> wyślij na Google Drive do podfolderu danej emocji -> usuń lokalną kopię.

Uruchomienie:
    python -m scripts.download.tiktok.collect                     # tiktok + youtube
    python -m scripts.download.tiktok.collect --source tiktok     # tylko TikTok
    python -m scripts.download.tiktok.collect --source youtube    # tylko YouTube (bez captchy)

TikTok wymaga zmiennej środowiskowej TIKTOK_MS_TOKEN. Google Drive wymaga
pliku secrets/gdrive_credentials.json (patrz README w tym katalogu).

Stan (już przetworzone ID wideo, liczniki per emocja) jest zapisywany do
data/tiktok_state.json, więc przerwanie i ponowne uruchomienie nie duplikuje
pracy.
"""

import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

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
    VIDEOS_PER_YOUTUBE_QUERY_PER_ROUND,
    CollectorConfig,
)
from scripts.download.tiktok.content_filters import AiCaptionFilter
from scripts.download.tiktok.downloader import download_video, get_video_metadata
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader
from scripts.download.tiktok.emotion_classifier import VideoEmotionClassifier
from scripts.download.tiktok.hashtag_search import TikTokHashtagSearcher
from scripts.download.tiktok.youtube_search import YouTubeSearcher


class CollectorState:
    """Persystencja przetworzonych ID wideo i liczników zaakceptowanych per emocja."""

    def __init__(self, state_path) -> None:
        self.state_path = state_path
        self.processed_ids: set[str] = set()
        # Hash zawartości pliku (SHA-256) już zaakceptowanych wideo - łapie ten sam
        # materiał wgrany ponownie pod innym ID (repost), czego nie widać po ID.
        self.content_hashes: set[str] = set()
        self.emotion_counts: dict[str, int] = {emotion: 0 for emotion in EMOTION_CLASSES}
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.processed_ids = set(data.get("processed_ids", []))
            self.content_hashes = set(data.get("content_hashes", []))
            saved_counts = data.get("emotion_counts", {})
            for emotion in EMOTION_CLASSES:
                self.emotion_counts[emotion] = saved_counts.get(emotion, 0)

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "processed_ids": sorted(self.processed_ids),
                    "content_hashes": sorted(self.content_hashes),
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

    def is_duplicate_content(self, content_hash: str) -> bool:
        return content_hash in self.content_hashes

    def mark_content_hash(self, content_hash: str) -> None:
        self.content_hashes.add(content_hash)

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

    print("Sprawdzam zawartość Dysku pod kątem duplikatów (także od innych osób)...")
    drive_hashes = uploader.collect_content_hashes(list(emotion_folder_ids.values()))
    state.content_hashes |= drive_hashes
    print(f"  → {len(drive_hashes)} plików już na Dysku, zliczonych jako znane treści.")

    # Lokalny licznik per emocja widzi tylko WŁASNE uploady tej maszyny - żeby
    # nie przekroczyć target_per_emotion, gdy inna osoba już dorzuciła swoje,
    # synchronizujemy z faktycznym stanem współdzielonych podfolderów.
    for emotion, folder_id in emotion_folder_ids.items():
        drive_count = len(uploader.list_files(folder_id))
        state.emotion_counts[emotion] = max(state.emotion_counts[emotion], drive_count)
    state.save()

    work_items: list[tuple[str, str, int]] = []
    if "tiktok" in config.sources:
        work_items += [("tiktok", h, VIDEOS_PER_HASHTAG_PER_ROUND) for h in config.hashtags]
    if "youtube" in config.sources:
        work_items += [
            ("youtube", q, VIDEOS_PER_YOUTUBE_QUERY_PER_ROUND) for q in config.youtube_queries
        ]

    common_args = (config, state, caption_filter, emotion_classifier, uploader, emotion_folder_ids)

    if "tiktok" in config.sources:
        async with TikTokHashtagSearcher(config.ms_token) as tiktok_searcher:
            youtube_searcher = YouTubeSearcher()
            await _run_loop(work_items, tiktok_searcher, youtube_searcher, *common_args)
    else:
        youtube_searcher = YouTubeSearcher()
        await _run_loop(work_items, None, youtube_searcher, *common_args)

    print(f"Zakończono: wszystkie emocje osiągnęły {config.target_per_emotion} wideo.")


async def _run_loop(
    work_items: list[tuple[str, str, int]],
    tiktok_searcher: TikTokHashtagSearcher | None,
    youtube_searcher: YouTubeSearcher,
    config: CollectorConfig,
    state: CollectorState,
    caption_filter: AiCaptionFilter,
    emotion_classifier: VideoEmotionClassifier,
    uploader: GoogleDriveUploader,
    emotion_folder_ids: dict[str, str],
) -> None:
    """Główna pętla: przechodzi po źródłach/zapytaniach aż komplet emocji zebrany."""
    consecutive_errors = 0

    while not state.is_complete(config.target_per_emotion):
        for platform, query, round_size in work_items:
            if state.is_complete(config.target_per_emotion):
                break

            searcher = tiktok_searcher if platform == "tiktok" else youtube_searcher

            _throttle()

            try:
                async for video in searcher.search(query, round_size):
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
                        print(f"Przyjęto [{platform}/{emotion}]: {video.url}\n  {counts_str}")
                    state.save()
                    consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                print(f"Błąd przy {platform}:{query}: {e}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(
                        f"Zbyt wiele błędów z rzędu - pauza {SOFTBAN_COOLDOWN_SECONDS}s "
                        "(prawdopodobny rate-limit)."
                    )
                    time.sleep(SOFTBAN_COOLDOWN_SECONDS)
                    consecutive_errors = 0


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
    metadata = get_video_metadata(video.url)
    if metadata is None:
        print(f"  odrzucono [{video.url}]: brak metadanych")
        return None
    if caption_filter.is_likely_ai_generated(metadata.description, []):
        print(f"  odrzucono [{video.url}]: prawdopodobnie AI-generated")
        return None

    result = download_video(video.url, DOWNLOAD_TMP_DIR)
    if not result.success or result.path is None:
        print(f"  odrzucono [{video.url}]: pobieranie nieudane ({result.error})")
        return None

    try:
        content_hash = _hash_file(result.path)
        if state.is_duplicate_content(content_hash):
            print(f"  odrzucono [{video.url}]: duplikat treści (już na Dysku)")
            return None

        emotion = emotion_classifier.classify_video(result.path)
        if emotion is None:
            print(f"  odrzucono [{video.url}]: brak pewnej emocji")
            return None
        if not state.needs_more(emotion, config.target_per_emotion):
            print(f"  odrzucono [{video.url}]: {emotion} już ma komplet ({config.target_per_emotion})")
            return None

        uploader.upload_file(
            result.path,
            remote_name=f"{video.platform}_{video.source_label}_{video.video_id}.mp4",
            folder_id=emotion_folder_ids[emotion],
        )
        state.mark_content_hash(content_hash)
        return emotion
    finally:
        result.path.unlink(missing_ok=True)


def _hash_file(path: Path) -> str:
    """
    Liczy MD5 pliku (do wykrywania duplikatów treści niezależnie od ID).

    MD5, nie SHA-256: Google Drive sam liczy i wystawia md5Checksum dla
    każdego wgranego pliku, więc ten sam algorytm pozwala porównać lokalny
    plik z tym, co już leży na wspólnym Dysku (patrz GoogleDriveUploader.
    collect_content_hashes) - w tym z plikami wgranymi przez inną osobę.
    """
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _throttle() -> None:
    """Losowe opóźnienie między żądaniami, żeby ograniczyć ryzyko rate-limitu."""
    time.sleep(random.uniform(MIN_REQUEST_DELAY_SECONDS, MAX_REQUEST_DELAY_SECONDS))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kolektor wideo z psami (TikTok/YouTube)")
    parser.add_argument(
        "--source",
        choices=["tiktok", "youtube", "all"],
        default="all",
        help="Które źródło uruchomić (domyślnie: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sources = ["tiktok", "youtube"] if args.source == "all" else [args.source]
    asyncio.run(run_collection(CollectorConfig(sources=sources)))
