#!/usr/bin/env python3
"""
Masowa anotacja w kilku procesach naraz.

    python -m scripts.annotation.run_batch_parallel --workers 4

Poprzedni przebieg zajął 9 h 46 min na 1496 nagraniach przy 1 kl./s. Przy
gęstszym próbkowaniu, którego wymaga złapanie kadru z psem patrzącym w obiektyw,
sekwencyjnie byłoby to kilkadziesiąt godzin — więcej, niż zostało do terminu.

Zamiast przerabiać działający pipeline na pracę współbieżną (i ryzykować cichym
rozjechaniem stanu) dzielimy LISTĘ NAGRAŃ na rozłączne części. Każdy proces ma
własny plik COCO i własny postęp, więc procesy nie mają czego popsuć sobie
nawzajem, a przerwany przebieg wznawia się częściami.

Scalanie części wymaga PRZENUMEROWANIA identyfikatorów, bo każda część liczy je
od nowa. Razem z `image_id` trzeba przemapować `neutral_frame_id` — wskazuje on
obraz, więc pozostawiony bez zmian po cichu wiązałby anotację z klatką neutralną
zupełnie innego psa.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DEFAULT_OUTPUT_DIR: Path = Path("data/annotations")
DEFAULT_MERGED_NAME: str = "annotations.json"

# Domyślna liczba procesów.
#
# Poprzednia wartość (4) opierała się na założeniu, że „torch i tak zrównolegla
# pojedynczy przebieg". Pomiar tego nie potwierdza: JEDEN proces z szesnastoma
# wątkami przerabia nagranie w ~120 s, a ten sam proces z czterema — w ~45 s.
# Modele zrównoleglają się wewnętrznie SŁABO, więc rdzeń oddany procesowi robi
# więcej niż rdzeń oddany wątkowi.
#
# Zmierzone na tej maszynie (16 rdzeni, 32 GB, CPU bez CUDA):
#
#     4 procesy x 4 wątki   9.3 rdzenia zajęte   ~0.67 nagrania/min (zgrubnie)
#    12 procesów x 1 wątek  12.2 rdzenia zajęte   1.10 nagrania/min (okno 10 min
#                                                 w stanie ustalonym)
#
# Czyli około półtora raza szybciej, a nie pięć — 1600 nagrań w ~24 h.
#
# UWAGA na sposób mierzenia: pierwsza wersja tego komentarza mówiła o 3.4
# nagrania/min, bo za koniec pierwszego nagrania wzięto POJAWIENIE SIĘ
# PIERWSZEJ KLATKI. To dwie różne rzeczy — klatki zapisuje dopiero trek
# z peakami, więc pierwsza klatka pochodzi z nagrania, które akurat miało
# peaki, a nie z tego, które skończyło się najwcześniej. Tempo trzeba liczyć
# z licznika URUCHOMIONYCH nagrań w oknie czasu, nie z artefaktów na dysku.
#
# Granicą jest PAMIĘĆ, nie procesor: dwanaście procesów zajmuje ~16.5 GB z 32 GB
# i wychodzi na plateau. Na proces składa się komplet modeli (~1.3 GB: HRNet-W48,
# YOLOv8m, EfficientNet-B4) ORAZ bufor klatek nagrania — `max_frames_per_video`
# pełnych klatek naraz, czyli przy 1920x1080 około 300 MB. Ten drugi składnik
# rośnie z rozdzielczością materiału, więc zapas poniżej ~4 GB jest ryzykowny.
DEFAULT_WORKERS: int = 12

# Pole anotacji wskazujące OBRAZ klatki neutralnej — musi jechać razem z mapą
# identyfikatorów, inaczej scalony zbiór wiąże psy z cudzymi bazami AU.
NEUTRAL_REFERENCE_FIELD: str = "neutral_frame_id"


def shard_output_dir(output_dir: Path, shard: int, shards: int) -> Path:
    """
    Zwraca katalog wyników danej części.

    Przy JEDNEJ części nie ma podkatalogu: `BatchConfig.__post_init__` pomija
    wtedy rozdzielanie plików, bo nie ma z czym kolidować. Scalanie musi liczyć
    tak samo — inaczej `--workers 1` przetwarza cały materiał poprawnie, po czym
    wywraca się na szukaniu nieistniejącego `shard_0`.

    Args:
        output_dir: Katalog wyjściowy podany przez użytkownika
        shard: Numer części
        shards: Na ile części dzielony jest materiał

    Returns:
        Katalog, do którego pisze ta część (zgodny z `BatchConfig.__post_init__`)
    """
    if shards == 1:
        return output_dir
    return output_dir / f"shard_{shard}"


def merge_shards(shard_paths: list[Path], allow_missing: bool = False) -> dict:
    """
    Scala zbiory COCO z części, przenumerowując identyfikatory.

    Args:
        shard_paths: Ścieżki do plików annotations.json kolejnych części
        allow_missing: Czy pominąć brakujące części zamiast przerwać. Domyślnie
            NIE, bo przy scalaniu końcowym brak części znaczy, że proces padł,
            a cichy wynik "o jedną dwunastą mniejszy" wygląda jak poprawny.
            Włączane świadomie przy zdejmowaniu MIGAWKI z trwającego przebiegu,
            gdzie części jeszcze niezapisane to normalny stan, a nie awaria.

    Returns:
        Scalony słownik COCO

    Raises:
        FileNotFoundError: Gdy któregoś pliku części nie ma
    """
    merged: dict = {"images": [], "annotations": []}
    next_image_id = 1
    next_annotation_id = 1

    for path in shard_paths:
        if not path.exists():
            if not allow_missing:
                raise FileNotFoundError(f"Brak wyniku części: {path}")
            print(f"[UWAGA] pomijam brakujaca czesc: {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as handle:
            shard = json.load(handle)

        for key in ("info", "licenses", "categories"):
            if key in shard and key not in merged:
                merged[key] = shard[key]

        id_map: dict[int, int] = {}
        for image in shard.get("images", []):
            id_map[image["id"]] = next_image_id
            merged["images"].append({**image, "id": next_image_id})
            next_image_id += 1

        for annotation in shard.get("annotations", []):
            remapped = {
                **annotation,
                "id": next_annotation_id,
                "image_id": id_map[annotation["image_id"]],
            }
            neutral = annotation.get(NEUTRAL_REFERENCE_FIELD)
            if neutral is not None:
                # Brak w mapie znaczy, że część zapisała anotację bez swojej
                # klatki neutralnej — lepiej zostawić None niż wskazać cudzą.
                remapped[NEUTRAL_REFERENCE_FIELD] = id_map.get(neutral)
            merged["annotations"].append(remapped)
            next_annotation_id += 1

    return merged


def launch_shard(shard: int, workers: int, extra_args: list[str]) -> subprocess.Popen:
    """
    Uruchamia jedną część jako osobny proces.

    Args:
        shard: Numer części
        workers: Łączna liczba części
        extra_args: Argumenty przekazywane do `batch_annotate`

    Returns:
        Uruchomiony proces
    """
    environment = dict(os.environ)
    # Bez tego każdy proces bierze wszystkie rdzenie i procesy biją się o nie,
    # przez co wersja "równoległa" bywa wolniejsza od sekwencyjnej.
    threads = max(1, (os.cpu_count() or workers) // workers)
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)

    command = [
        sys.executable,
        "-m",
        "scripts.annotation.batch_annotate",
        "--shard",
        str(shard),
        "--shards",
        str(workers),
        *extra_args,
    ]
    return subprocess.Popen(command, cwd=REPO_ROOT, env=environment)


def wait_for_shards(processes: list[subprocess.Popen]) -> list[int]:
    """
    Czeka na zakończenie wszystkich części.

    Args:
        processes: Uruchomione procesy części

    Returns:
        Kody wyjścia w kolejności części
    """
    codes: list[int] = []
    for shard, process in enumerate(processes):
        code = process.wait()
        codes.append(code)
        status = "OK" if code == 0 else f"BLAD (kod {code})"
        print(f"[{status}] czesc {shard}")
    return codes


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parsuje argumenty; nieznane przekazuje dalej do batch_annotate."""
    parser = argparse.ArgumentParser(
        description="Masowa anotacja w kilku procesach naraz",
        epilog="Nierozpoznane argumenty ida wprost do scripts.annotation.batch_annotate",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Liczba rownoleglych procesow (domyslnie: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Katalog wyjsciowy (czesci laduja w podkatalogach shard_N)",
    )
    parser.add_argument(
        "--merged-name",
        type=str,
        default=DEFAULT_MERGED_NAME,
        help="Nazwa scalonego pliku w katalogu wyjsciowym",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Nie uruchamiaj przetwarzania, tylko scal gotowe czesci",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Pomin czesci bez wyniku zamiast przerwac (migawka z trwajacego przebiegu)",
    )
    return parser.parse_known_args()


def merge_and_save(
    output_dir: Path, workers: int, merged_name: str, allow_missing: bool = False
) -> Optional[Path]:
    """
    Scala części i zapisuje wynik.

    Args:
        output_dir: Katalog wyjściowy
        workers: Liczba części
        merged_name: Nazwa pliku wynikowego
        allow_missing: Czy pominąć części, które jeszcze nie zapisały wyniku

    Returns:
        Ścieżka scalonego pliku albo None, gdy scalenie się nie powiodło
    """
    paths = [
        shard_output_dir(output_dir, i, workers) / "annotations.json" for i in range(workers)
    ]
    try:
        merged = merge_shards(paths, allow_missing=allow_missing)
    except FileNotFoundError as error:
        print(f"[BLAD] {error}", file=sys.stderr)
        return None

    destination = output_dir / merged_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False)
    print(
        f"[OK] Scalono {workers} czesci: {len(merged['images'])} obrazow, "
        f"{len(merged['annotations'])} anotacji -> {destination}"
    )
    return destination


def main() -> None:
    """Punkt wejścia: uruchamia części i scala wyniki."""
    args, extra_args = parse_args()
    if args.workers < 1:
        print("[BLAD] --workers musi byc dodatnie", file=sys.stderr)
        sys.exit(1)

    if not args.merge_only:
        print(f"[..] Uruchamiam {args.workers} czesci")
        started = time.time()
        processes = [
            launch_shard(shard, args.workers, ["--output-dir", str(args.output_dir), *extra_args])
            for shard in range(args.workers)
        ]
        codes = wait_for_shards(processes)
        print(f"[..] Przetwarzanie zajelo {(time.time() - started) / 60:.1f} min")
        if any(code != 0 for code in codes):
            print("[UWAGA] Nie wszystkie czesci skonczyly poprawnie — scalam to, co jest")

    if merge_and_save(
        args.output_dir, args.workers, args.merged_name, args.allow_missing
    ) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
