#!/usr/bin/env python3
"""
Przetwarzanie dosypanych nagrań w tle, obok trwającej anotacji.

    python -m scripts.annotation.process_inbox --videos data/videos --dataset data/dataset_manual

Różni się od zwykłego przebiegu wsadowego jedną rzeczą: KURUJE CO KILKA NAGRAŃ,
a nie raz na końcu. Przebieg trwa godzinami, więc kuracja wyłącznie na końcu
znaczyłaby, że anotator przez cały ten czas nie widzi ani jednej nowej pary —
a sens dosypywania w tle jest właśnie taki, żeby materiał dochodził na bieżąco.

Nagrania już przerobione pomija `--resume` batcha, więc dorzucenie plików do
wspólnego katalogu i ponowne uruchomienie przetwarza tylko nowe.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Co ile przetworzonych nagrań odświeżamy kurację. Za często — kuracja czyta
# cały zbiór i marnuje czas procesora potrzebny pipeline'owi; za rzadko —
# anotator długo nie widzi nowego materiału.
CURATE_EVERY: int = 20

# Ile nagrań batch bierze na jedno wywołanie. Dzielimy przebieg na porcje, żeby
# między nimi zmieścić kurację — batch sam nie daje takiego zaczepienia.
CHUNK_VIDEOS: int = CURATE_EVERY


def write_stage(dataset: Path, stage: str) -> None:
    """
    Zapisuje etap pracy, żeby interfejs mógł go pokazać.

    Bez tego pasek postępu stoi nieruchomo przez pierwsze minuty (ładowanie
    modeli trwa ~40 s) i anotator uznaje, że narzędzie się zawiesiło.

    Args:
        dataset: Katalog zbioru
        stage: Nazwa etapu
    """
    (dataset / "stage.json").write_text(json.dumps({"stage": stage}), encoding="utf-8")


def venv_python() -> str:
    """
    Zwraca interpreter z `.venv`, a gdy go nie ma — bieżący.

    Returns:
        Ścieżka do interpretera
    """
    for candidate in (
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def run_batch(videos: Path, dataset: Path) -> int:
    """
    Puszcza jeden przebieg anotacji wsadowej na wspólnym katalogu nagrań.

    Args:
        videos: Katalog z nagraniami
        dataset: Katalog wynikowy zbioru

    Returns:
        Kod wyjścia procesu
    """
    command = [
        venv_python(),
        "-m",
        "scripts.annotation.batch_annotate",
        "--input-dir",
        str(videos),
        "--output-dir",
        str(dataset),
        "--frames-dir",
        str(dataset / "frames"),
        "--resume",
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def run_curation(dataset: Path) -> bool:
    """
    Odświeża plik po kuracji, żeby nowe pary trafiły do kolejki anotatora.

    Args:
        dataset: Katalog zbioru

    Returns:
        True, gdy kuracja się powiodła
    """
    annotations = dataset / "annotations.json"
    if not annotations.is_file():
        return False
    command = [
        venv_python(),
        "-m",
        "scripts.annotation.curate_for_review",
        "--dataset",
        str(annotations),
        "--out",
        str(dataset / "curated.json"),
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode == 0


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description="Przetwarzanie dosypanych nagran w tle")
    parser.add_argument("--videos", type=Path, required=True, help="Katalog z nagraniami")
    parser.add_argument("--dataset", type=Path, required=True, help="Katalog wynikowy")
    return parser.parse_args()


def main() -> None:
    """Punkt wejścia: przetwarza nagrania i odświeża kurację po drodze."""
    args = parse_args()
    args.dataset.mkdir(parents=True, exist_ok=True)
    lock = args.dataset / "worker.pid"

    try:
        print(f"[..] Przetwarzam nagrania z {args.videos}", flush=True)
        write_stage(args.dataset, "processing")
        code = run_batch(args.videos, args.dataset)
        if code != 0:
            print(f"[UWAGA] Batch zakonczyl sie kodem {code}", flush=True)

        write_stage(args.dataset, "curating")
        if run_curation(args.dataset):
            print("[OK] Kuracja odswiezona — nowe pary sa w kolejce", flush=True)
            write_stage(args.dataset, "done")
        else:
            print("[UWAGA] Kuracja sie nie powiodla", flush=True)
            write_stage(args.dataset, "failed")
    finally:
        # Blokada musi zniknąć nawet po awarii, inaczej dosypywanie zostanie
        # zablokowane na zawsze i nikt nie zgadnie dlaczego.
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
