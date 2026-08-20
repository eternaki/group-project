#!/usr/bin/env python3
"""
Dociąga z folderu Google Drive to, czego jeszcze nie ma na dysku.

    python scripts/download/sync_drive_gdown.py FOLDER_ID_LUB_URL

W odróżnieniu od `download_drive_folder.py` NIE wymaga klucza Google API — listę
plików bierze z `gdown`, a potem pobiera każdy plik OSOBNO po jego identyfikatorze.
Ta okrężna droga jest konieczna: `gdown` pobierając folder jednym poleceniem tnie
go po ~50 plikach, natomiast samo listowanie i pobieranie pojedynczych plików
limitu nie ma.

Struktura podfolderów jest ZACHOWYWANA, bo niesie informację: anotacja wsadowa
czyta emocję z nazwy katalogu nadrzędnego (`batch_annotate.py`, `video_path.
parent.name`). Wideo z katalogu `angry/` dostanie więc etykietę źródłową `angry`,
a spłaszczenie wszystkiego do jednego katalogu bezpowrotnie by ją zgubiło.

Skrypt jest wznawialny: pobiera wyłącznie brakujące nazwy, więc przerwanie
i ponowne uruchomienie nie kosztuje niczego poza listowaniem.
"""

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gdown

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT: str = "data/drive_dogs"

# Co ile pobranych plików meldujemy postęp. Przy ośmiuset plikach log co sztuce
# jest nieczytelny, a milczenie przez godzinę wygląda jak zawieszenie.
PROGRESS_EVERY: int = 25

# Ile razy próbujemy pobrać plik, zanim go odpuścimy. Drive potrafi odmówić
# pojedynczemu żądaniu i odpowiedzieć normalnie przy następnym.
MAX_ATTEMPTS: int = 3

# Przerwa po nieudanej próbie [s]
RETRY_PAUSE_S: float = 2.0


@dataclass
class SyncStats:
    """Licznik przebiegu pobierania."""

    listed: int = 0
    already_have: int = 0
    downloaded: int = 0
    failed: int = 0
    bytes_downloaded: int = 0


def folder_id_of(source: str) -> str:
    """
    Wyciąga identyfikator folderu z URL albo zwraca podany identyfikator.

    Args:
        source: Pełny adres folderu Drive albo sam identyfikator

    Returns:
        Identyfikator folderu
    """
    match = re.search(r"/folders/([0-9A-Za-z_-]+)", source)
    return match.group(1) if match else source


def existing_names(output: Path) -> set[str]:
    """
    Zwraca nazwy plików, które już leżą na dysku — w DOWOLNYM podkatalogu.

    Porównujemy po samej nazwie, nie po ścieżce: ten sam plik mógł wcześniej
    trafić do innego katalogu i pobieranie go po raz drugi byłoby marnotrawstwem.

    Args:
        output: Katalog docelowy

    Returns:
        Zbiór nazw plików
    """
    return {path.name for path in output.rglob("*") if path.is_file()}


def _download_one(file_id: str, target: Path) -> int:
    """
    Pobiera jeden plik z ponowieniami.

    Args:
        file_id: Identyfikator pliku w Drive
        target: Ścieżka docelowa

    Returns:
        Rozmiar pobranego pliku w bajtach; 0 gdy się nie udało
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            gdown.download(id=file_id, output=str(target), quiet=True)
        except Exception as error:  # noqa: BLE001 — chcemy pominąć plik, nie przerwać przegon
            logger.debug("proba %d dla %s: %s", attempt, target.name, error)
        if target.is_file() and target.stat().st_size > 0:
            return target.stat().st_size
        time.sleep(RETRY_PAUSE_S)
    return 0


def sync(source: str, output: Path) -> SyncStats:
    """
    Dociąga brakujące pliki z folderu Drive.

    Args:
        source: Adres albo identyfikator folderu
        output: Katalog docelowy

    Returns:
        Statystyki przebiegu
    """
    url = f"https://drive.google.com/drive/folders/{folder_id_of(source)}"
    logger.info("Listuje folder %s", url)
    listing = gdown.download_folder(url=url, skip_download=True, quiet=True, use_cookies=False)
    if not listing:
        raise SystemExit("Nie udalo sie wylistowac folderu — czy jest publiczny?")

    output.mkdir(parents=True, exist_ok=True)
    have = existing_names(output)
    stats = SyncStats(listed=len(listing))

    pending = [item for item in listing if Path(item.path).name not in have]
    stats.already_have = stats.listed - len(pending)
    logger.info(
        "W folderze %d plikow, mamy juz %d, do pobrania %d",
        stats.listed, stats.already_have, len(pending),
    )

    for index, item in enumerate(pending, 1):
        relative = Path(item.path.replace("\\", "/"))
        size = _download_one(item.id, output / relative)
        if size:
            stats.downloaded += 1
            stats.bytes_downloaded += size
        else:
            stats.failed += 1
            logger.warning("nie pobrano: %s", relative)
        if index % PROGRESS_EVERY == 0 or index == len(pending):
            logger.info(
                "  %d/%d  (%.1f GB, nieudanych %d)",
                index, len(pending), stats.bytes_downloaded / 1e9, stats.failed,
            )
    return stats


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description="Dociaga folder Google Drive bez klucza API")
    parser.add_argument("source", help="URL folderu Drive albo jego identyfikator")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Katalog docelowy")
    return parser.parse_args()


def main() -> None:
    """Punkt wejścia."""
    args = parse_args()
    output = Path(args.out)
    if not output.is_absolute():
        output = REPO_ROOT / output

    stats = sync(args.source, output)
    logger.info("Pobrano       : %d plikow (%.2f GB)", stats.downloaded, stats.bytes_downloaded / 1e9)
    logger.info("Juz mielismy  : %d", stats.already_have)
    if stats.failed:
        logger.info("Nie pobrano   : %d — uruchom ponownie, skrypt jest wznawialny", stats.failed)
    if stats.failed and stats.downloaded == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
