#!/usr/bin/env python3
"""
Pobiera nagrania z publicznego folderu Google Drive przez oficjalne API.

    python scripts/download/download_drive_folder.py FOLDER_ID_LUB_URL

W odróżnieniu od `sync_drive_gdown.py` NIE wpada w dzienny limit pobrań, przez
który anonimowe ściąganie kończy się po 40-60 plikach („Cannot retrieve the
public link (...) or have had many accesses"). Kosztem jest klucz Google API —
darmowy, ale trzeba go raz założyć; instrukcja w `docs/POBIERANIE_Z_DRIVE.md`.

Schodzi REKURENCYJNIE do podfolderów i ZACHOWUJE ich strukturę, bo ta niesie
informację: anotacja wsadowa czyta etykietę źródłową z nazwy katalogu
nadrzędnego (`batch_annotate.py`, `video_path.parent.name`). Nagranie z
katalogu `angry/` dostanie więc etykietę `angry`, a spłaszczenie wszystkiego
do jednego katalogu bezpowrotnie by ją zgubiło.

Skrypt jest wznawialny: pomija nazwy, które już leżą na dysku (gdziekolwiek
w drzewie), więc przerwanie i ponowne uruchomienie nie kosztuje nic poza
listowaniem.
"""

import argparse
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API: str = "https://www.googleapis.com/drive/v3/files"

DEFAULT_OUTPUT: str = "data/drive_dogs"

# Plik z kluczem. UWAGA: świadomie NIE ma tu `key.txt` — leży w nim klucz
# Kaggle, a wcześniejsza wersja brała go w zastępstwie i wysyłała do Google,
# które odpowiadało nieczytelnym błędem uwierzytelnienia.
KEY_FILE: str = "drive_key.txt"

VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".avi", ".webm", ".mkv")

FOLDER_MIME: str = "application/vnd.google-apps.folder"

# Ile pozycji prosimy w jednym żądaniu listowania (maksimum dopuszczane przez API)
PAGE_SIZE: int = 1000

# Co ile pobranych plików meldujemy postęp
PROGRESS_EVERY: int = 25

# Rozmiar kawałka przy zapisie strumienia [bajty]
CHUNK_BYTES: int = 1 << 20

REQUEST_TIMEOUT_S: int = 30
DOWNLOAD_TIMEOUT_S: int = 300


@dataclass(frozen=True)
class DriveFile:
    """Nagranie na Dysku razem z miejscem, w które ma trafić."""

    file_id: str
    name: str
    relative_dir: str


@dataclass
class SyncStats:
    """Licznik przebiegu pobierania."""

    listed: int = 0
    already_have: int = 0
    downloaded: int = 0
    failed: int = 0
    bytes_downloaded: int = 0


def read_key() -> str:
    """
    Czyta klucz Google Drive API ze zmiennej środowiskowej albo z pliku.

    Returns:
        Klucz API

    Raises:
        SystemExit: Gdy klucza nie ma — z podpowiedzią, jak go założyć
    """
    key = os.environ.get("GDRIVE_API_KEY", "").strip()
    if key:
        return key
    candidate = REPO_ROOT / KEY_FILE
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8").strip()
        if text:
            return text
    raise SystemExit(
        f"Brak klucza Google Drive API.\n"
        f"  Ustaw zmienną GDRIVE_API_KEY albo wpisz klucz do pliku {KEY_FILE}.\n"
        f"  Jak go założyć: docs/POBIERANIE_Z_DRIVE.md"
    )


def folder_id_from(source: str) -> str:
    """
    Wyciąga identyfikator folderu z URL albo przyjmuje sam identyfikator.

    Args:
        source: URL folderu Dysku albo gotowy identyfikator

    Returns:
        Identyfikator folderu
    """
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", source)
    return match.group(1) if match else source


def _get_json(url: str) -> dict:
    """
    Pobiera odpowiedź API i rozpakowuje ją z JSON-a.

    Args:
        url: Pełny adres żądania

    Returns:
        Odpowiedź jako słownik
    """
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as response:
        return json.loads(response.read().decode())


def _list_children(folder_id: str, key: str) -> list[dict]:
    """
    Listuje bezpośrednią zawartość jednego folderu, z paginacją.

    Args:
        folder_id: Identyfikator folderu
        key: Klucz API

    Returns:
        Pozycje (pliki i podfoldery) leżące wprost w tym folderze
    """
    items: list[dict] = []
    token: Optional[str] = None
    query = urllib.parse.quote(f"'{folder_id}' in parents and trashed=false")
    while True:
        url = (
            f"{API}?q={query}&key={key}&pageSize={PAGE_SIZE}"
            f"&fields=nextPageToken,files(id,name,mimeType)"
            f"&supportsAllDrives=true&includeItemsFromAllDrives=true"
        )
        if token:
            url += f"&pageToken={token}"
        data = _get_json(url)
        items.extend(data.get("files", []))
        token = data.get("nextPageToken")
        if not token:
            return items


def list_videos(folder_id: str, key: str, relative_dir: str = "") -> list[DriveFile]:
    """
    Listuje nagrania w folderze i we WSZYSTKICH jego podfolderach.

    Args:
        folder_id: Identyfikator folderu, od którego zaczynamy
        key: Klucz API
        relative_dir: Ścieżka tego folderu względem korzenia (do rekurencji)

    Returns:
        Nagrania razem ze ścieżką katalogu, w którym leżą
    """
    videos: list[DriveFile] = []
    for item in _list_children(folder_id, key):
        name = item["name"]
        if item.get("mimeType") == FOLDER_MIME:
            deeper = f"{relative_dir}/{name}" if relative_dir else name
            videos.extend(list_videos(item["id"], key, deeper))
        elif name.lower().endswith(VIDEO_SUFFIXES):
            videos.append(DriveFile(file_id=item["id"], name=name, relative_dir=relative_dir))
    return videos


def _download_via_gdown(video: DriveFile, destination: Path) -> int:
    """
    Zapasowa droga dla plików, którym klucz API odmawia.

    Klucz API otwiera wyłącznie pliki udostępnione „każdemu z linkiem" WPROST.
    Plik wrzucony do udostępnionego folderu przez inną osobę bywa czytelny
    w przeglądarce (dziedziczy uprawnienia folderu), a przez API zwraca 403.
    Zmierzone 27.08.2026: wszystkie odmowy dotyczyły nagrań `youtube_*`
    wgranych kolektorem — ani jedno inne nagranie nie odmówiło.

    `gdown` chodzi publicznym adresem przeglądarkowym, który dziedziczenie
    uprawnień honoruje, więc te pliki pobiera. Płaci za to limitem dziennym
    (~40-60 plików), dlatego jest DROGĄ ZAPASOWĄ, nie główną.

    Args:
        video: Nagranie do pobrania
        destination: Ścieżka docelowa

    Returns:
        Liczba zapisanych bajtów; 0, gdy i ta droga zawiodła
    """
    try:
        import gdown
    except ImportError:
        return 0
    try:
        result = gdown.download(
            id=video.file_id, output=str(destination), quiet=True, use_cookies=False
        )
    except Exception as error:  # noqa: BLE001 — gdown rzuca czym popadnie
        logger.debug(f"gdown nie dal rady {video.name}: {error}")
        return 0
    if not result or not destination.is_file():
        return 0
    return destination.stat().st_size


def download_file(video: DriveFile, destination: Path, key: str) -> int:
    """
    Pobiera jedno nagranie pod wskazaną ścieżkę.

    Niedokończony plik jest usuwany — inaczej przy następnym uruchomieniu
    wyglądałby na pobrany i nagranie zniknęłoby ze zbioru po cichu.

    Args:
        video: Nagranie do pobrania
        destination: Ścieżka docelowa
        key: Klucz API

    Returns:
        Liczba zapisanych bajtów; 0, gdy pobranie się nie powiodło
    """
    url = f"{API}/{video.file_id}?alt=media&key={key}&supportsAllDrives=true"
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
            with open(destination, "wb") as handle:
                while chunk := response.read(CHUNK_BYTES):
                    handle.write(chunk)
                    written += len(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        destination.unlink(missing_ok=True)
        written = _download_via_gdown(video, destination)
        if written:
            logger.info(f"przez gdown (API odmowilo): {video.name}")
            return written
        logger.warning(f"nie pobrano: {video.name} ({error})")
        return 0
    if written == 0:
        destination.unlink(missing_ok=True)
    return written


def existing_names(output: Path) -> set[str]:
    """
    Zbiera nazwy nagrań już leżących na dysku, w całym drzewie.

    Porównujemy po NAZWIE, a nie po pełnej ścieżce, bo część zbioru pobrano
    kiedyś płasko — inaczej te nagrania pobrałyby się po raz drugi.

    Args:
        output: Katalog docelowy

    Returns:
        Nazwy plików obecnych na dysku
    """
    if not output.is_dir():
        return set()
    return {path.name for path in output.rglob("*") if path.is_file()}


def sync(source: str, output: Path) -> SyncStats:
    """
    Dociąga z folderu Dysku to, czego jeszcze nie ma na dysku lokalnym.

    Args:
        source: URL folderu Dysku albo jego identyfikator
        output: Katalog docelowy

    Returns:
        Statystyki przebiegu
    """
    key = read_key()
    folder_id = folder_id_from(source)
    logger.info(f"Listuje folder {folder_id} (rekurencyjnie)")
    videos = list_videos(folder_id, key)

    have = existing_names(output)
    missing = [video for video in videos if video.name not in have]
    stats = SyncStats(listed=len(videos), already_have=len(videos) - len(missing))
    logger.info(
        f"Na Dysku {stats.listed} nagran, mamy juz {stats.already_have}, "
        f"do pobrania {len(missing)}"
    )

    for index, video in enumerate(missing, start=1):
        target = output / video.relative_dir / video.name if video.relative_dir else output / video.name
        written = download_file(video, target, key)
        if written:
            stats.downloaded += 1
            stats.bytes_downloaded += written
        else:
            stats.failed += 1
        if index % PROGRESS_EVERY == 0:
            logger.info(
                f"  {index}/{len(missing)}  "
                f"({stats.bytes_downloaded / 1e9:.1f} GB, nieudanych {stats.failed})"
            )
    return stats


def main() -> None:
    """Punkt wejścia CLI."""
    parser = argparse.ArgumentParser(
        description="Pobiera nagrania z folderu Google Drive przez oficjalne API"
    )
    parser.add_argument("source", help="URL folderu Drive albo jego identyfikator")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Katalog docelowy")
    args = parser.parse_args()

    stats = sync(args.source, Path(args.out))
    logger.info(f"Pobrano       : {stats.downloaded} plikow ({stats.bytes_downloaded / 1e9:.2f} GB)")
    logger.info(f"Juz mielismy  : {stats.already_have}")
    if stats.failed:
        logger.info(f"Nie pobrano   : {stats.failed} — uruchom ponownie, skrypt jest wznawialny")
        sys.exit(1)


if __name__ == "__main__":
    main()
