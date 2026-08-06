#!/usr/bin/env python3
"""
Statystyki wygenerowanego zbioru COCO — z naciskiem na wiarygodność etykiet AU.

Odpowiada na pytanie, którego nie da się zadać samemu plikowi COCO: ile aktywacji
AU (`is_active`) przeżywa konfrontację ze ZMIERZONYM szumem tego AU w tym treku.
Reguły AU są pre-etykietami, a nie celem treningowym, więc przed Sprintem 16
trzeba wiedzieć, jaka część etykiet pochodzi z drgania keypoints.

Użycie:
    python scripts/debug/dataset_stats.py --dataset data/dataset_v2/annotations.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from packages.data.coco import MIN_SIGNAL_TO_NOISE, au_signal_above_noise

# Pola, bez których zbiór nie nadaje się na wejście sieci AU (Sprint 16)
REQUIRED_FIELDS: tuple[str, ...] = (
    "track_id",
    "frame_role",
    "label_source",
    "au_noise",
    "au_sample_count",
    "neutral_source",
    "procrustes_keypoints",
    "keypoints",
    "bbox",
    "au_analysis",
)

# Poniżej tylu próbek odchylenie standardowe jest zbyt niepewne, żeby ważyć nim
# próbkę w treningu (przy n=3 obciążenie ~11%, rozrzut własny ~50%)
RELIABLE_SAMPLE_COUNT: int = 5

# Ile AU pokazać w rozbiciu na jednostki
TOP_AU_COUNT: int = 10


def load_annotations(dataset_file: Path) -> list[dict]:
    """
    Wczytuje anotacje ze zbioru COCO.

    Args:
        dataset_file: Ścieżka do pliku annotations.json

    Returns:
        Lista anotacji

    Raises:
        ValueError: Gdy plik nie zawiera sekcji `annotations`
    """
    data = json.loads(dataset_file.read_text(encoding="utf-8"))
    if "annotations" not in data:
        raise ValueError(f"{dataset_file} nie wygląda na zbiór COCO (brak 'annotations')")
    return list(data["annotations"])


def field_completeness(annotations: list[dict]) -> dict[str, int]:
    """Liczy, ile anotacji ma wypełnione każde z wymaganych pól."""
    return {
        field: sum(
            1
            for ann in annotations
            if field in ann and ann[field] not in (None, [], {})
        )
        for field in REQUIRED_FIELDS
    }


def noise_verdicts(annotations: list[dict]) -> dict:
    """
    Konfrontuje `is_active` ze zmierzonym szumem AU.

    Args:
        annotations: Anotacje klatek szczytowych

    Returns:
        Słownik z licznikami i rozkładami
    """
    stats = {
        "measurements": 0,
        "active": 0,
        "active_above_noise": 0,
        "active_below_noise": 0,
        "noise_unknown": 0,
    }
    noises: list[float] = []
    ratios: list[float] = []
    per_au_active: Counter = Counter()
    per_au_survives: Counter = Counter()

    for ann in annotations:
        for name, value in ann.get("au_analysis", {}).items():
            stats["measurements"] += 1
            verdict = au_signal_above_noise(value)
            if isinstance(value, dict) and value.get("noise") is not None:
                noises.append(float(value["noise"]))
            if isinstance(value, dict) and isinstance(value.get("snr"), (int, float)):
                ratios.append(float(value["snr"]))
            if verdict is None:
                stats["noise_unknown"] += 1
            if not (isinstance(value, dict) and value.get("is_active")):
                continue
            stats["active"] += 1
            per_au_active[name] += 1
            if verdict is True:
                stats["active_above_noise"] += 1
                per_au_survives[name] += 1
            elif verdict is False:
                stats["active_below_noise"] += 1

    stats["noise"] = noises
    stats["snr"] = ratios
    stats["per_au_active"] = per_au_active
    stats["per_au_survives"] = per_au_survives
    return stats


def _percentiles(values: list[float]) -> str:
    """Formatuje rozkład jako mediana i decyle."""
    if not values:
        return "brak danych"
    return (
        f"mediana {np.median(values):.3f}  "
        f"p10 {np.percentile(values, 10):.3f}  "
        f"p90 {np.percentile(values, 90):.3f}"
    )


def print_report(annotations: list[dict]) -> None:
    """Wypisuje raport o zbiorze."""
    peaks = [a for a in annotations if a.get("frame_role") == "peak"]
    print(f"anotacje: {len(annotations)}  (peaki: {len(peaks)})")
    print(f"role klatek: {dict(Counter(a.get('frame_role') for a in annotations))}")
    print(f"emocje: {dict(Counter(a.get('emotion') for a in annotations).most_common())}")

    print("\n--- kompletność pól (wejście sieci AU) ---")
    for field, have in field_completeness(annotations).items():
        mark = "OK " if have == len(annotations) else "BRAK"
        print(f"  [{mark}] {field:22s} {have}/{len(annotations)}")

    counts = [c for a in annotations for c in a.get("au_sample_count", {}).values()]
    if counts:
        weak = 100 * sum(1 for c in counts if c < RELIABLE_SAMPLE_COUNT) / len(counts)
        print(
            f"\nliczba prób szumu: mediana {np.median(counts):.0f}  "
            f"min {min(counts)}  maks {max(counts)}  "
            f"| poniżej {RELIABLE_SAMPLE_COUNT} prób: {weak:.1f}%"
        )

    stats = noise_verdicts(peaks)
    active = max(stats["active"], 1)
    print(f"\n--- wiarygodność etykiet AU (próg snr = {MIN_SIGNAL_TO_NOISE}) ---")
    print(f"  pomiary AU:            {stats['measurements']}")
    print(f"  is_active=True:        {stats['active']}")
    print(
        f"  ...powyżej szumu:      {stats['active_above_noise']} "
        f"({100 * stats['active_above_noise'] / active:.1f}% aktywacji)"
    )
    print(
        f"  ...PONIŻEJ szumu:      {stats['active_below_noise']} "
        f"({100 * stats['active_below_noise'] / active:.1f}% aktywacji)"
    )
    print(f"  szum niezmierzony:     {stats['noise_unknown']}")
    print(f"  szum ratio: {_percentiles(stats['noise'])}")
    print(f"  snr:        {_percentiles(stats['snr'])}")

    print(f"\n--- AU: aktywacje -> ile przeżywa odsiew szumem (top {TOP_AU_COUNT}) ---")
    for name, count in stats["per_au_active"].most_common(TOP_AU_COUNT):
        survives = stats["per_au_survives"][name]
        print(f"  {name:8s} {count:5d} -> {survives:5d}  ({100 * survives / count:.0f}%)")


def main() -> int:
    """Uruchamia analizę zbioru."""
    parser = argparse.ArgumentParser(description="Statystyki wiarygodności zbioru COCO")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dataset_v2/annotations.json"),
        help="Plik annotations.json",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Brak pliku: {args.dataset}")
        return 1

    print_report(load_annotations(args.dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
