#!/usr/bin/env python3
"""
Kuracja zbioru do weryfikacji ręcznej.

Bierze COCO z batch annotation i zostawia wyłącznie pary (klatka szczytowa,
klatka neutralna), na których w ogóle wolno mierzyć AU — resztę odrzuca bramka
jakości. Wynik jest UPORZĄDKOWANY pod pracę człowieka, a nie pod numer klatki.

Dwie rzeczy decydują o kolejności:

* **Grupowanie po wideo** — anotator kończy całe nagranie, zanim przejdzie do
  następnego. Pary jednego nagrania dzielą klatkę neutralną, więc ocenia je w
  jednym kontekście, a zepsutą bazę poprawia raz dla całej grupy. Gdy praca
  urwie się w połowie (a przy terminie trzech tygodni urwie się), zostaje zbiór
  KOMPLETNYCH nagrań — a takie nadają się do treningu, w odróżnieniu od setki
  nagrań rozgrzebanych po jednej parze.
* **Sygnał** — nagrania idą od najbardziej wyrazistych, wewnątrz nagrania tak
  samo. Pary z pomiarem blisko progu decyzyjnego wyprzedzają te, gdzie sygnał
  jest dziesięć razy większy od szumu i reguła i tak trafia.

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

# Kuracja świadomie luzuje asymetrię ponad `QualityThresholds().max_asymmetry`
# (0.20). Ostry próg chroni POMIAR reguł, który dzieli wszystko przez rozstaw
# oczu i przy obrocie głowy zawyża każde AU. Kolejka trafia jednak do CZŁOWIEKA,
# a człowiek czyta ucho i pysk także na ujęciu trzy czwarte, zaś dla zasłoniętej
# połowy ma werdykt `not_observable`.
#
# Zmierzone na zbiorze: 0.20 daje 271 par, 0.50 daje 1082 pary przy spadku
# udziału par bez żadnego sygnału z 42% do 34%. Wartość MUSI być domyślna, a nie
# podawana flagą — przy dziedziczeniu z `QualityThresholds` ponowna kuracja
# „na domyślnych" po cichu tnie kolejkę czterokrotnie.
REVIEW_MAX_ASYMMETRY: float = 0.50

# Ostry próg 40 px zakłada odczyt z klatki. Anotator ogląda POWIĘKSZONY kadr
# mordy, więc czyta ucho i pysk także przy 30 px. Ten próg i tak nie jest tu
# wiążący: odrzucone peaki mają medianę szerokości mordy 82 px — tracimy je na
# obrocie głowy, nie na rozdzielczości.
REVIEW_MIN_FACE_WIDTH: float = 30.0

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
    Układa pary NAGRANIAMI: najpierw całe jedno nagranie, potem następne.

    Wcześniej kolejka przeplatała nagrania, żeby przerwana praca dawała przekrój
    materiału. Grupowanie jest jednak lepsze z dwóch powodów:

    * **Skończone nagranie to gotowa jednostka, niedokończone nie.** Przy przerwie
      zostaje zbiór kompletnych nagrań, a nie kawałki kilkuset.
    * **Pary jednego nagrania dzielą klatkę neutralną.** Anotator ocenia je w tym
      samym kontekście, a gdy baza okaże się zepsuta, poprawia ją RAZ dla całej
      grupy zamiast wyłapywać ten sam błąd rozrzucony po kolejce.

    Same nagrania idą od najbardziej wyrazistych, więc pierwszeństwo materiału
    z realnym sygnałem zostaje zachowane.

    Args:
        pairs: Pary przyjęte przez bramkę

    Returns:
        Nowa lista w kolejności podawania anotatorowi
    """
    by_video: dict[str, list[ReviewPair]] = defaultdict(list)
    for pair in pairs:
        by_video[pair.video].append(pair)
    for bucket in by_video.values():
        bucket.sort(key=lambda item: (-item.signal, -item.ambiguity))

    # Nagranie reprezentuje jego najlepsza para: nagranie z jednym mocnym
    # sygnałem jest ciekawsze niż nagranie z pięcioma pustymi parami.
    videos = sorted(
        by_video.values(),
        key=lambda bucket: (-max(pair.signal for pair in bucket), -len(bucket)),
    )
    return [pair for bucket in videos for pair in bucket]


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
        default=REVIEW_MAX_ASYMMETRY,
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
        default=REVIEW_MIN_FACE_WIDTH,
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
