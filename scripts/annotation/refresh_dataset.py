#!/usr/bin/env python3
"""
Jeden pełny obrót zbierania: Dysk -> anotacja -> kolejka -> git.

    python -m scripts.annotation.refresh_dataset

Pomyślany do uruchamiania CO KILKANAŚCIE GODZIN bez nadzoru: sprawdza, czy na
Dysku przybyło nagrań, i jeśli tak — pobiera je, przerabia, dokłada do kolejki
i publikuje. Gdy nic nie przybyło, kończy się od razu i niczego nie dotyka.

Trzy rzeczy, na których łatwo tu polec, i dlatego są pilnowane wprost:

* **kolejka jest POPOŁNIALNA** — kuracja idzie z `--keep` na kolejce już wydanej
  ludziom, nigdy od zera. Przebudowa od zera zabrała raz 603 z 605 werdyktów;
* **przed publikacją bramka** (`queue_guard`) liczy werdykty bez pary i przerywa,
  gdy przybyło ich choć o jeden. Strata jest cicha — dziennik przeżywa, tylko
  przestaje się z czymkolwiek wiązać;
* **nowy materiał idzie OSOBNĄ falą** do własnego katalogu wyjściowego. Części
  batcha dzielą listę przez `videos[shard::shards]`, więc dołożenie nagrań do
  istniejącego katalogu przesuwa przydziały i nagrania robią się po raz drugi.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation.queue_guard import orphaned_verdicts  # noqa: E402
from scripts.download.download_drive_folder import (  # noqa: E402
    folder_id_from,
    list_videos,
    read_key,
    sync,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DRIVE_FOLDER: str = "https://drive.google.com/drive/folders/1jxUaN3Mq1ge8lFcPzwnN2ISl0E9k9mfQ"

VIDEO_DIR: Path = Path("data/drive_dogs")
DATASET_DIR: Path = Path("data/dataset_final")
LABELS_DIR: Path = Path("data/labels")
QUEUE_PATH: Path = DATASET_DIR / "work" / "curated.json"

# Katalog źródłowy, którego NIE ma po co przerabiać: 414 nagrań `magnific-*`
# dało 8 par po kuracji (0.02 na nagranie) przy 2.23 dla materiału z katalogów
# emocji. Sześć godzin przebiegu za osiem par.
SKIP_PREFIXES: tuple[str, ...] = ("magnific-",)

# Katalog z nagraniami zebranymi wcześniej, poza tym obiegiem
SKIP_DIRS: tuple[str, ...] = ("DOGS",)

# Znacznik "ta fala zostala juz wlana do zbioru". Bez niego nie da sie odroznic
# fali czekajacej na scalenie od takiej, ktora juz wlano — a pomylka w jedna
# strone gubi dane, w druga robi duplikaty.
MERGED_MARKER: str = ".merged"

DEFAULT_WORKERS: int = 12


@dataclass
class CycleResult:
    """Wynik jednego obrotu."""

    downloaded: int = 0
    processed: int = 0
    pairs_before: int = 0
    pairs_after: int = 0
    published: bool = False


def _run(command: list[str], description: str) -> str:
    """
    Uruchamia polecenie i przerywa obieg, gdy się nie powiedzie.

    Args:
        command: Polecenie z argumentami
        description: Opis do komunikatu błędu

    Returns:
        Wyjście standardowe polecenia

    Raises:
        SystemExit: Gdy polecenie zwróci kod różny od zera
    """
    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"PRZERWANE na kroku: {description}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def processed_video_names() -> set[str]:
    """
    Zbiera nazwy nagrań przerobionych w którejkolwiek fali.

    Returns:
        Nazwy nagrań bez rozszerzenia
    """
    done: set[str] = set()
    for progress in Path("data").glob("dataset_*/shard_*/progress.json"):
        with open(progress, encoding="utf-8") as handle:
            done |= set(json.load(handle).get("processed_videos", []))
    return done


def stage_new_videos(target: Path) -> int:
    """
    Rozkłada nieprzerobione nagrania twardymi dowiązaniami, z podziałem na emocje.

    Dowiązania nie zajmują miejsca, a struktura katalogów musi przeżyć, bo
    anotacja wsadowa czyta z niej etykietę źródłową.

    Args:
        target: Katalog kolejki do przerobienia

    Returns:
        Ile nagrań czeka
    """
    done = processed_video_names()
    staged = 0
    for video in sorted(VIDEO_DIR.rglob("*.mp4")):
        if video.stem in done or video.parent.name in SKIP_DIRS:
            continue
        if video.name.startswith(SKIP_PREFIXES):
            continue
        label = "unlabeled" if video.parent.name == VIDEO_DIR.name else video.parent.name
        destination = target / label / video.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(video, destination)
        staged += 1
    return staged


def count_pairs(queue_path: Path) -> int:
    """
    Liczy pary w kolejce.

    Args:
        queue_path: Plik COCO kolejki

    Returns:
        Liczba par (klatek szczytowych)
    """
    if not queue_path.is_file():
        return 0
    with open(queue_path, encoding="utf-8") as handle:
        coco = json.load(handle)
    return sum(
        1
        for annotation in coco.get("annotations", [])
        if annotation.get("neutral_frame_id") not in (None, annotation["image_id"])
    )


def drive_has_new() -> int:
    """
    Sprawdza, ile nagrań z Dysku jeszcze u nas nie leży.

    Returns:
        Liczba brakujących nagrań
    """
    videos = list_videos(folder_id_from(DRIVE_FOLDER), read_key())
    have = {path.name for path in VIDEO_DIR.rglob("*") if path.is_file()}
    return sum(1 for video in videos if video.name not in have)


def publish(result: CycleResult, allowed_orphans: int, push: bool) -> None:
    """
    Sprawdza bramkę i publikuje kolejkę do gita.

    Args:
        result: Wynik obiegu, uzupełniany o informację o publikacji
        allowed_orphans: Ile werdyktów bez pary wolno zastać
        push: Czy wysłać do zdalnego repozytorium

    Raises:
        SystemExit: Gdy kolejka gubi więcej werdyktów, niż wolno
    """
    orphans = orphaned_verdicts(QUEUE_PATH, LABELS_DIR)
    if len(orphans) > allowed_orphans:
        raise SystemExit(
            f"PRZERWANE PRZED PUBLIKACJA: kolejka gubi {len(orphans)} werdyktow, "
            f"wolno {allowed_orphans}. Nic nie wyslane."
        )
    logger.info(f"Bramka przeszla: werdyktow bez pary {len(orphans)} (wolno {allowed_orphans})")

    _run(["git", "add", str(DATASET_DIR / "work")], "git add")
    status = _run(["git", "status", "--porcelain", str(DATASET_DIR / "work")], "git status")
    if not status.strip():
        logger.info("Kolejka bez zmian — nie ma czego publikowac")
        return

    stamp = datetime.now().strftime("%d.%m.%Y %H:%M")
    message = (
        f"[SPRINT-15][TASK] Kolejka po dociagnieciu: {result.pairs_after} par\n\n"
        f"Obieg automatyczny {stamp}. Pobrane {result.downloaded} nagran, "
        f"przerobione {result.processed}.\n"
        f"Kolejka {result.pairs_before} -> {result.pairs_after} par.\n"
        f"Bramka werdyktow przeszla — zadna oceniona para nie wypadla.\n\n"
        f"Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
    )
    _run(["git", "commit", "-m", message], "git commit")
    if push:
        _run(["git", "push", "origin", "HEAD"], "git push")
    result.published = True


def _unmerged_waves() -> list[Path]:
    """
    Wskazuje fale, które SIĘ SKOŃCZYŁY, ale nie zostały wlane do zbioru.

    Drugi sposób na cichą stratę, tuż obok przerwanej fali: batch dobiegł końca
    i zapisał `annotations.json`, po czym obieg padł przed scaleniem. Nagrania
    są już odhaczone w `progress.json`, więc nikt ich nie powtórzy, a ich wynik
    leży odłogiem. Znacznik `.merged` odróżnia „wlane" od „gotowe, ale czeka".

    Returns:
        Katalogi wyników fal czekających na scalenie, od najstarszej
    """
    czeka: list[Path] = []
    for output in sorted(DATASET_DIR.parent.glob("dataset_*")):
        if output == DATASET_DIR:
            continue
        if (output / "annotations.json").is_file() and not (output / MERGED_MARKER).exists():
            czeka.append(output)
    return czeka


def _unfinished_wave() -> Optional[tuple[Path, Path]]:
    """
    Znajduje falę przerwaną w połowie, żeby ją dokończyć zamiast zakładać nową.

    Bez tego przerwany obieg CICHO GUBI pracę: nagrania przerobione przed
    przerwaniem są już zapisane w `progress.json` części, więc następny obieg
    ich nie rozłoży — a ich anotacje leżą w częściach, których nikt nigdy nie
    scali, bo scalanie bierze tylko `annotations.json` świeżej fali. Zmierzone
    28.08.2026: obieg padł po 210 z 306 nagrań i tyle właśnie by przepadło.

    Falę uznajemy za przerwaną, gdy ma katalog kolejki i części z postępem,
    ale NIE MA scalonego wyniku — ten powstaje dopiero na samym końcu.

    Returns:
        Para (katalog kolejki, katalog wyniku) albo None, gdy nie ma czego kończyć
    """
    for output in sorted(DATASET_DIR.parent.glob("dataset_20*"), reverse=True):
        if (output / "annotations.json").is_file():
            continue
        if not any(output.glob("shard_*/progress.json")):
            continue
        wave = output.parent / f"todo_{output.name.removeprefix('dataset_')}"
        if wave.is_dir():
            return wave, output
    return None


def run_cycle(workers: int, allowed_orphans: int, push: bool) -> CycleResult:
    """
    Wykonuje jeden pełny obieg.

    Args:
        workers: Ile procesów anotacji uruchomić
        allowed_orphans: Ile werdyktów bez pary wolno zastać
        push: Czy wysłać wynik do zdalnego repozytorium

    Returns:
        Podsumowanie obiegu
    """
    result = CycleResult(pairs_before=count_pairs(QUEUE_PATH))

    for zaleglosc in _unmerged_waves():
        logger.info(f"Wlewam zalegla fale {zaleglosc.name}")
        merge_into_dataset(zaleglosc / "annotations.json")

    missing = drive_has_new()
    logger.info(f"Na Dysku brakuje u nas {missing} nagran")
    if missing:
        stats = sync(DRIVE_FOLDER, VIDEO_DIR)
        result.downloaded = stats.downloaded

    wave, output = _unfinished_wave() or (None, None)
    if wave is None:
        wave = DATASET_DIR.parent / f"todo_{datetime.now():%Y%m%d_%H%M}"
        output = DATASET_DIR.parent / f"dataset_{wave.name.removeprefix('todo_')}"
        result.processed = stage_new_videos(wave)
    else:
        result.processed = sum(1 for _ in wave.rglob("*.mp4"))
        logger.info(f"Wznawiam przerwana fale {wave.name} zamiast zakladac nowa")

    if not result.processed:
        logger.info("Nic nowego do przerobienia — koniec obiegu")
        return result

    logger.info(f"Przerabiam {result.processed} nagran do {output}")
    _run(
        [
            sys.executable, "-m", "scripts.annotation.run_batch_parallel",
            "--workers", str(workers),
            "--output-dir", str(output),
            "--input-dir", str(wave),
            "--frames-dir", str(DATASET_DIR / "frames"),
            "--resume",
        ],
        "anotacja wsadowa",
    )

    merge_into_dataset(output / "annotations.json")
    _run(
        [
            sys.executable, "-m", "scripts.annotation.curate_for_review",
            "--keep", str(QUEUE_PATH),
        ],
        "kuracja z zachowaniem kolejki",
    )
    _run([sys.executable, "-m", "scripts.annotation.build_work_pack"], "paczka robocza")

    result.pairs_after = count_pairs(QUEUE_PATH)
    publish(result, allowed_orphans, push)
    return result


def merge_into_dataset(wave_coco: Path) -> None:
    """
    Dokłada wynik fali do surowego COCO zbioru, przenumerowując identyfikatory.

    Args:
        wave_coco: Plik COCO świeżej fali

    Raises:
        SystemExit: Gdy po scaleniu klatka szczytowa wskazuje cudzą klatkę neutralną
    """
    from scripts.annotation.run_batch_parallel import merge_shards

    base = DATASET_DIR / "annotations.json"
    merged = merge_shards([base, wave_coco] if base.is_file() else [wave_coco])
    images = {image["id"]: image["file_name"] for image in merged["images"]}
    for annotation in merged["annotations"]:
        neutral_id = annotation.get("neutral_frame_id")
        if neutral_id is None:
            continue
        if neutral_id not in images:
            raise SystemExit("PRZERWANE: neutral_frame_id wskazuje nieistniejacy obraz")
        if images[neutral_id].rsplit("/", 1)[0] != images[annotation["image_id"]].rsplit("/", 1)[0]:
            raise SystemExit("PRZERWANE: peak zwiazany z klatka neutralna INNEGO nagrania")
    base.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    (wave_coco.parent / MERGED_MARKER).write_text(
        datetime.now().isoformat(timespec="seconds"), encoding="utf-8"
    )


def main() -> None:
    """Punkt wejścia CLI."""
    parser = argparse.ArgumentParser(description="Jeden obieg zbierania zbioru")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--allowed-orphans",
        type=int,
        default=2,
        help="Ile werdyktow bez pary wolno zastac (stan zastany, nie cel)",
    )
    parser.add_argument("--no-push", action="store_true", help="Nie wysylaj do zdalnego repo")
    args = parser.parse_args()

    result = run_cycle(args.workers, args.allowed_orphans, not args.no_push)
    logger.info(
        f"OBIEG: pobrane {result.downloaded}, przerobione {result.processed}, "
        f"kolejka {result.pairs_before} -> {result.pairs_after}, "
        f"opublikowane: {'tak' if result.published else 'nie'}"
    )


if __name__ == "__main__":
    main()
