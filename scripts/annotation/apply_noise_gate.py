#!/usr/bin/env python3
"""
Szumowy gejt AU dla całego zbioru 9k — jedną komendą, bez przetwarzania wideo.

    python -m scripts.annotation.apply_noise_gate

Surowa flaga `is_active` z reguł zapala się od drgania keypoints: na materiale
9k klatek 44% jej zapaleń NIE przewyższa zmierzonego szumu treku, czyli jako
etykieta jest bezużyteczna. Ten skrypt nie dotyka wideo ani pomiarów — bierze
gotowe pole `au_analysis` z `curated.json` (ratio + szum + snr są już policzone)
i wyprowadza z niego TRÓJSTANOWĄ etykietę automatyczną (`au_auto_verdict`):
aktywacja tylko gdy sygnał przewyższa szum, ruch utopiony w szumie to „nie
wiadomo", a nie „spoczynek".

Wynik: `data/dataset_final/release/au_auto_labels.csv` — po jednym wierszu na
klatkę szczytową (7119), kodowanie `1` / `0` / puste jak w `au_labels.csv`
(werdykt człowieka), żeby obie tabele dało się porównać wprost. To jest
automatyczna etykieta AU dla dużego zbioru — słaba z natury reguł, ale uczciwa.
"""

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from packages.data.coco import (
    AU_VERDICT_ACTIVE,
    AU_VERDICT_INACTIVE,
    au_auto_verdicts,
)
from packages.models.delta_action_units import ACTION_UNIT_NAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET: str = "dataset_final"
CSV_NAME: str = "au_auto_labels.csv"

# Kodowanie werdyktu do CSV: identyczne z `au_labels.csv` człowieka, więc puste
# pole znaczy „nie wiadomo", a nie zero — trening ma je pominąć, nie uczyć się
# zmyślonego negatywu.
VERDICT_TO_CSV: dict[str, str] = {AU_VERDICT_ACTIVE: "1", AU_VERDICT_INACTIVE: "0"}


def _curated_path(dataset_dir: Path) -> Path:
    """
    Zwraca ścieżkę `curated.json`, akceptując oba układy katalogów.

    Materiał roboczy trzymany jest w `work/`, ale historycznie leżał wprost w
    katalogu zbioru — bierzemy ten, który istnieje.

    Args:
        dataset_dir: Katalog zbioru w `data/`

    Returns:
        Ścieżka pliku kuracji

    Raises:
        SystemExit: Gdy kuracji nie ma w żadnym z miejsc
    """
    for candidate in (dataset_dir / "curated.json", dataset_dir / "work" / "curated.json"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Brak kuracji w {dataset_dir} — uruchom curate_for_review")


def build_rows(coco: dict) -> tuple[list[dict[str, object]], Counter]:
    """
    Buduje wiersze CSV z automatycznymi werdyktami dla klatek szczytowych.

    Args:
        coco: Wczytany `curated.json`

    Returns:
        Para (wiersze CSV, licznik aktywnych AU po gejcie)
    """
    images = {image["id"]: image for image in coco["images"]}
    rows: list[dict[str, object]] = []
    active_counter: Counter = Counter()
    for annotation in coco["annotations"]:
        if annotation.get("frame_role") != "peak":
            continue
        verdicts = au_auto_verdicts(annotation.get("au_analysis", {}))
        active_counter.update(au for au, v in verdicts.items() if v == AU_VERDICT_ACTIVE)
        image = images[annotation["image_id"]]
        row: dict[str, object] = {
            "pair_key": image["file_name"],
            "source_video": image.get("source_video"),
            "emotion": annotation.get("emotion"),
            "n_active": sum(1 for v in verdicts.values() if v == AU_VERDICT_ACTIVE),
        }
        for au in ACTION_UNIT_NAMES:
            row[au] = VERDICT_TO_CSV.get(verdicts.get(au, ""), "")
        rows.append(row)
    return rows, active_counter


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """
    Zapisuje tabelę automatycznych werdyktów AU.

    Args:
        rows: Wiersze z `build_rows`
        path: Ścieżka pliku CSV
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description="Szumowy gejt AU dla zbioru")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Nazwa zbioru w data/")
    parser.add_argument(
        "--output",
        default=None,
        help="Plik CSV wynikowy (domyślnie release/au_auto_labels.csv)",
    )
    return parser.parse_args()


def main() -> None:
    """Punkt wejścia."""
    args = parse_args()
    dataset_dir = REPO_ROOT / "data" / args.dataset
    coco = json.loads(_curated_path(dataset_dir).read_text(encoding="utf-8"))

    rows, active = build_rows(coco)
    expressive = sum(1 for row in rows if row["n_active"] > 0)

    output: Optional[str] = args.output
    path = Path(output) if output else dataset_dir / "release" / CSV_NAME
    if not path.is_absolute():
        path = REPO_ROOT / path
    write_csv(rows, path)

    logger.info("Klatek szczytowych         : %d", len(rows))
    logger.info("Z co najmniej jednym AU    : %d", expressive)
    logger.info("Aktywacji AU po gejcie     : %d", sum(active.values()))
    logger.info(
        "Najczęstsze AU             : %s",
        ", ".join(f"{name} {count}" for name, count in active.most_common(6)),
    )
    logger.info("Zapisano: %s", path)


if __name__ == "__main__":
    main()
