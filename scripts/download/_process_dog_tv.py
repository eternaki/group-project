"""
Jednorazowy obieg dla fali dog_tv_24_7_nareski: anotacja -> scalenie -> kuracja
-> paczka -> commit+push, bez własnego kroku pobierania z Dysku (już zrobione
osobno, patrz _sync_dog_tv.py) - unika wymogu GDRIVE_API_KEY.

Powiela logikę scripts/annotation/refresh_dataset.py (merge_into_dataset,
kolejność kroków), ale bierze WPROST z data/raw/dog_tv_24_7_nareski zamiast
z data/drive_dogs.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation.refresh_dataset import merge_into_dataset, publish, CycleResult, count_pairs

DATASET_DIR = REPO_ROOT / "data" / "dataset_final"
QUEUE_PATH = DATASET_DIR / "work" / "curated.json"
INPUT_DIR = REPO_ROOT / "data" / "raw" / "dog_tv_24_7_nareski"
WORKERS = 12


def run(command: list[str], description: str) -> None:
    print(f"=== {description} ===", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"PRZERWANE na kroku: {description} (kod {result.returncode})")


def _find_unfinished_output() -> Path:
    """
    Zwraca katalog fali w toku (ma progress.json, ale nie ma znacznika
    .merged), zamiast zawsze zakładać nowy ze świeżym znacznikiem czasu -
    inaczej każde wznowienie zaczynało 2106 nagrań od zera.
    """
    for candidate in sorted((REPO_ROOT / "data").glob("dataset_2026*")):
        if (candidate / ".merged").exists():
            continue
        if any(candidate.glob("shard_*/progress.json")):
            return candidate
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return REPO_ROOT / "data" / f"dataset_{stamp}"


def main() -> None:
    result = CycleResult(pairs_before=count_pairs(QUEUE_PATH))

    output = _find_unfinished_output()
    print(f"=== uzywam katalogu fali: {output} ===", flush=True)

    run(
        [
            sys.executable, "-m", "scripts.annotation.run_batch_parallel",
            "--workers", str(WORKERS),
            "--output-dir", str(output),
            "--input-dir", str(INPUT_DIR),
            "--frames-dir", str(DATASET_DIR / "frames"),
            "--resume",
        ],
        "anotacja wsadowa dog_tv_24_7_nareski",
    )

    print("=== scalanie do data/dataset_final/annotations.json ===", flush=True)
    merge_into_dataset(output / "annotations.json")

    run(
        [
            sys.executable, "-m", "scripts.annotation.curate_for_review",
            "--keep", str(QUEUE_PATH),
        ],
        "kuracja z zachowaniem kolejki",
    )
    run([sys.executable, "-m", "scripts.annotation.build_work_pack"], "paczka robocza")

    result.pairs_after = count_pairs(QUEUE_PATH)
    publish(result, allowed_orphans=2, push=True)
    print(
        f"OBIEG ZAKONCZONY: kolejka {result.pairs_before} -> {result.pairs_after} par, "
        f"opublikowane: {'tak' if result.published else 'nie'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
