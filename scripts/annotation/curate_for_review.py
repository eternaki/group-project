#!/usr/bin/env python3
"""
Kuracja zbioru do weryfikacji ręcznej.

Bierze COCO z batch annotation i zostawia wyłącznie pary (klatka szczytowa,
klatka neutralna), na których w ogóle wolno mierzyć AU — resztę odrzuca bramka
jakości. Wynik jest UPORZĄDKOWANY pod pracę człowieka, a nie pod numer klatki.

Dwie rzeczy decydują o kolejności:

* **Przeplot po wideo** — anotator dostaje kolejne pary z RÓŻNYCH nagrań.
  Gdy praca urwie się w połowie (a przy terminie trzech tygodni urwie się),
  zweryfikowana część jest wtedy przekrojem materiału, a nie pięćdziesięcioma
  klatkami jednego psa.
* **Niepewność w obrębie wideo** — najpierw pary, w których pomiar AU leży
  blisko progu decyzyjnego. Tam głos człowieka zmienia etykietę; tam, gdzie
  sygnał jest dziesięć razy większy od szumu, reguła i tak trafia.

Użycie:
    python -m scripts.annotation.curate_for_review \
        --dataset data/dataset_v2/annotations.json \
        --out data/dataset_v2/curated.json
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from packages.data.coco import (
    FRAME_ROLE_NEUTRAL,
    FRAME_ROLE_PEAK,
    MIN_SIGNAL_TO_NOISE,
)
from packages.pipeline.quality_gate import (
    FrameQuality,
    QualityThresholds,
    assess_frame,
)

# Pas niepewności pomiaru: poniżej sygnał tonie w szumie, powyżej jest
# bezdyskusyjny. Wewnątrz — decyduje człowiek, więc te pary idą pierwsze.
AMBIGUOUS_SNR_MIN: float = 0.5
AMBIGUOUS_SNR_MAX: float = 2.0

DEFAULT_DATASET: str = "data/dataset_v2/annotations.json"
DEFAULT_OUTPUT: str = "data/dataset_v2/curated.json"

# Separator ścieżek w COCO bywa windowsowy — normalizujemy przed rozbiciem
_WINDOWS_SEPARATOR: str = "\\"
_POSIX_SEPARATOR: str = "/"


@dataclass(frozen=True)
class ReviewPair:
    """
    Para klatek gotowa do weryfikacji przez człowieka.

    Attributes:
        peak: Anotacja klatki szczytowej
        neutral: Anotacja klatki neutralnej tego samego psa
        video: Klucz nagrania źródłowego (katalog klatek)
        signal: Ile AU przewyższa szum treku — po tym układa się kolejka
        ambiguity: Ile AU leży w pasie niepewności — rozstrzyga przy równym sygnale
        peak_quality: Ocena jakości klatki szczytowej
        neutral_quality: Ocena jakości klatki neutralnej
    """

    peak: dict
    neutral: dict
    video: str
    signal: int
    ambiguity: int
    peak_quality: FrameQuality
    neutral_quality: FrameQuality


def load_dataset(path: Path) -> dict:
    """
    Wczytuje zbiór COCO.

    Args:
        path: Ścieżka do pliku annotations.json

    Returns:
        Słownik COCO

    Raises:
        FileNotFoundError: Gdy pliku nie ma
    """
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono zbioru: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def video_key(file_name: str) -> str:
    """
    Wyciąga klucz nagrania ze ścieżki klatki.

    Args:
        file_name: Ścieżka klatki z pola `file_name` obrazu COCO

    Returns:
        Katalog nagrania; sama nazwa pliku, gdy klatka leży płasko
    """
    normalized = file_name.replace(_WINDOWS_SEPARATOR, _POSIX_SEPARATOR)
    head, separator, _ = normalized.rpartition(_POSIX_SEPARATOR)
    return head if separator else normalized


def signal_score(annotation: dict) -> int:
    """
    Liczy AU, których pomiar PRZEWYŻSZA szum treku.

    To po tej liczbie układa się kolejkę, a nie po samej niepewności. Zmierzone
    na zbiorze: 42% par nie ma ani jednego AU powyżej szumu, czyli anotator
    ogląda pustą parę i naciska Enter. Puste pary są potrzebne jako przykłady
    negatywne, ale nie muszą iść pierwsze — gdy praca się urwie, zdążone mają
    być te, na których cokolwiek widać.

    Args:
        annotation: Anotacja COCO z polem `au_analysis`

    Returns:
        Liczba aktywnych AU o sygnale powyżej szumu
    """
    count = 0
    for value in annotation.get("au_analysis", {}).values():
        if not isinstance(value, dict) or not value.get("is_active"):
            continue
        snr = value.get("snr")
        if isinstance(snr, (int, float)) and snr >= MIN_SIGNAL_TO_NOISE:
            count += 1
    return count


def ambiguity_score(annotation: dict) -> int:
    """
    Liczy AU, których pomiar leży blisko progu decyzyjnego.

    Args:
        annotation: Anotacja COCO z polem `au_analysis`

    Returns:
        Liczba AU o stosunku sygnału do szumu w pasie niepewności
    """
    count = 0
    for value in annotation.get("au_analysis", {}).values():
        if not isinstance(value, dict):
            continue
        snr = value.get("snr")
        if isinstance(snr, (int, float)) and AMBIGUOUS_SNR_MIN <= snr <= AMBIGUOUS_SNR_MAX:
            count += 1
    return count


def build_pairs(
    coco: dict,
    thresholds: QualityThresholds,
) -> tuple[list[ReviewPair], dict[str, int]]:
    """
    Buduje pary szczyt-neutral przechodzące bramkę jakości.

    Args:
        coco: Wczytany zbiór COCO
        thresholds: Progi bramki jakości

    Returns:
        Para (lista przyjętych par, licznik powodów odrzucenia)
    """
    images = {image["id"]: image for image in coco["images"]}
    annotations = coco["annotations"]
    neutrals = {
        ann["image_id"]: ann
        for ann in annotations
        if ann.get("frame_role") == FRAME_ROLE_NEUTRAL
    }

    pairs: list[ReviewPair] = []
    rejected: dict[str, int] = defaultdict(int)

    for peak in annotations:
        if peak.get("frame_role") != FRAME_ROLE_PEAK:
            continue
        neutral = neutrals.get(peak.get("neutral_frame_id"))
        if neutral is None:
            rejected["brak klatki neutralnej"] += 1
            continue

        peak_quality = assess_frame(peak.get("keypoints"), thresholds)
        neutral_quality = assess_frame(neutral.get("keypoints"), thresholds)
        if not (peak_quality.is_usable and neutral_quality.is_usable):
            for reason in peak_quality.reasons + neutral_quality.reasons:
                rejected[reason] += 1
            continue

        pairs.append(
            ReviewPair(
                peak=peak,
                neutral=neutral,
                video=video_key(images[peak["image_id"]]["file_name"]),
                signal=signal_score(peak),
                ambiguity=ambiguity_score(peak),
                peak_quality=peak_quality,
                neutral_quality=neutral_quality,
            )
        )
    return pairs, dict(rejected)


def order_for_review(pairs: list[ReviewPair]) -> list[ReviewPair]:
    """
    Układa pary do weryfikacji: przeplot po nagraniach, w środku niepewność.

    Args:
        pairs: Pary przyjęte przez bramkę

    Returns:
        Nowa lista w kolejności podawania anotatorowi
    """
    by_video: dict[str, list[ReviewPair]] = defaultdict(list)
    for pair in pairs:
        by_video[pair.video].append(pair)
    for bucket in by_video.values():
        # Najpierw pary z realnym sygnałem, dopiero potem niepewne. Przeplot
        # bierze z każdego nagrania pozycję zerową, więc na początku globalnej
        # kolejki lądują najbardziej wyraziste pary całego zbioru.
        bucket.sort(key=lambda item: (-item.signal, -item.ambiguity))

    queues = sorted(by_video.values(), key=lambda bucket: -len(bucket))
    ordered: list[ReviewPair] = []
    position = 0
    while any(position < len(bucket) for bucket in queues):
        for bucket in queues:
            if position < len(bucket):
                ordered.append(bucket[position])
        position += 1
    return ordered


def _quality_fields(quality: FrameQuality) -> dict:
    """Zamienia ocenę jakości na pola zapisywane w anotacji."""
    return {
        "asymmetry": round(quality.asymmetry, 4),
        "weak_keypoint_ratio": round(quality.weak_ratio, 4),
        "face_width_px": round(quality.face_width, 1),
    }


def build_curated(coco: dict, ordered: list[ReviewPair]) -> dict:
    """
    Składa wynikowy zbiór COCO z par w kolejności weryfikacji.

    Klatka neutralna jedzie razem ze szczytową, bo bez niej człowiek nie ma
    do czego porównać mimiki — AU jest z definicji różnicą względem niej.

    Args:
        coco: Zbiór źródłowy (dla `images` i metadanych)
        ordered: Pary w kolejności podawania anotatorowi

    Returns:
        Słownik COCO gotowy do importu do sesji anotacji
    """
    images = {image["id"]: image for image in coco["images"]}
    out_images: list[dict] = []
    out_annotations: list[dict] = []
    seen_images: set[int] = set()

    for order, pair in enumerate(ordered):
        peak = dict(pair.peak)
        neutral = dict(pair.neutral)
        peak["review_order"] = order
        peak["quality"] = _quality_fields(pair.peak_quality)
        peak["ambiguity"] = pair.ambiguity
        peak["signal"] = pair.signal
        neutral["review_order"] = order
        neutral["quality"] = _quality_fields(pair.neutral_quality)

        for annotation in (neutral, peak):
            image_id = annotation["image_id"]
            if image_id not in seen_images:
                seen_images.add(image_id)
                out_images.append(images[image_id])
            out_annotations.append(annotation)

    return {
        "info": {**coco.get("info", {}), "description": "DogFACS — do weryfikacji ręcznej"},
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": out_images,
        "annotations": out_annotations,
    }


def print_summary(
    pairs: list[ReviewPair],
    rejected: dict[str, int],
    total_peaks: int,
) -> None:
    """
    Wypisuje podsumowanie kuracji.

    Args:
        pairs: Pary przyjęte przez bramkę
        rejected: Licznik powodów odrzucenia
        total_peaks: Ile klatek szczytowych było w zbiorze źródłowym
    """
    videos = {pair.video for pair in pairs}
    print(f"Pary przyjete : {len(pairs)} z {total_peaks} ({100 * len(pairs) / max(total_peaks, 1):.1f}%)")
    print(f"Nagrania      : {len(videos)}")
    if pairs:
        with_signal = sum(1 for pair in pairs if pair.signal > 0)
        ambiguous = sum(1 for pair in pairs if pair.ambiguity > 0)
        print(f"Z AU powyzej szumu      : {with_signal} ({100 * with_signal / len(pairs):.1f}%)")
        print(f"Z AU w pasie niepewnosci: {ambiguous} ({100 * ambiguous / len(pairs):.1f}%)")
    print("\nPowody odrzucenia (kadry, nie pary):")
    for reason, count in sorted(rejected.items(), key=lambda item: -item[1]):
        print(f"  {count:6d}  {reason}")


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description="Kuracja zbioru do weryfikacji ręcznej")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Wejściowy COCO")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Wyjściowy COCO")
    parser.add_argument(
        "--max-asymmetry",
        type=float,
        default=QualityThresholds().max_asymmetry,
        help="Maksymalna asymetria połówek mordy",
    )
    parser.add_argument(
        "--max-weak-ratio",
        type=float,
        default=QualityThresholds().max_weak_ratio,
        help="Maksymalny udział niepewnych keypoints",
    )
    parser.add_argument(
        "--min-face-width",
        type=float,
        default=QualityThresholds().min_face_width,
        help="Minimalna szerokość mordy w pikselach",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Zapisz najwyżej tyle par (przycięcie po uporządkowaniu)",
    )
    return parser.parse_args()


def main() -> None:
    """Punkt wejścia: kuruje zbiór i zapisuje wynik."""
    args = parse_args()
    thresholds = QualityThresholds(
        max_asymmetry=args.max_asymmetry,
        max_weak_ratio=args.max_weak_ratio,
        min_face_width=args.min_face_width,
    )

    coco = load_dataset(Path(args.dataset))
    total_peaks = sum(
        1 for ann in coco["annotations"] if ann.get("frame_role") == FRAME_ROLE_PEAK
    )
    pairs, rejected = build_pairs(coco, thresholds)
    ordered = order_for_review(pairs)
    if args.limit is not None:
        ordered = ordered[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(build_curated(coco, ordered), handle, ensure_ascii=False)

    print_summary(pairs, rejected, total_peaks)
    print(f"\nZapisano {len(ordered)} par do {out_path}")


if __name__ == "__main__":
    main()
