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
        --dataset data/dataset_final/annotations.json \
        --out data/dataset_final/curated.json
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
from scripts.annotation.queue_unpack import is_packed, unpack_queue

# Pas niepewności pomiaru: poniżej sygnał tonie w szumie, powyżej jest
# bezdyskusyjny. Wewnątrz — decyduje człowiek, więc te pary idą pierwsze.
AMBIGUOUS_SNR_MIN: float = 0.5
AMBIGUOUS_SNR_MAX: float = 2.0

DEFAULT_DATASET: str = "data/dataset_final/annotations.json"
DEFAULT_OUTPUT: str = "data/dataset_final/curated.json"

# Asymetria jest tu PREFERENCJĄ, a nie realnym sitem, i próg odpowiada temu.
#
# Miara zakłada dokładne keypoints, a nasze takie nie są. Zmierzone: na punktach
# DogFLW stawianych przez człowieka mediana asymetrii wynosi 0.140; po dodaniu
# szumu o skali błędu naszego modelu (NME 0.091) rośnie do 0.307. Pies patrzący
# WPROST czyta się więc u nas jako 0.31 — 61% drogi do progu 0.50 zanim głowa
# w ogóle się obróci. Ranking kadrów przeżywa ten szum z rho 0.37, czyli miara
# ledwie porządkuje i na pewno nie rozstrzyga.
#
# 0.60 to wartość skalibrowana: zachowuje ten sam udział naturalnych póz (94.8%
# zbioru DogFLW), który próg 0.50 miał zachowywać przy dokładnych punktach.
# Obejrzane kadry potwierdzają rozjazd — w paśmie 0.45 (dotąd PRZYJMOWANYM)
# siedzą wyraźne profile, a w paśmie 0.55 (odrzucanym) psy patrzące na wprost.
#
# Wartość MUSI być domyślna, a nie podawana flagą — przy dziedziczeniu
# z `QualityThresholds` ponowna kuracja „na domyślnych" po cichu tnie kolejkę.
REVIEW_MAX_ASYMMETRY: float = 0.60

# Próg 30 px zakładał, że anotator odczyta mordę z powiększenia. PIERWSZE
# WERDYKTY TEGO ZAŁOŻENIA NIE POTWIERDZIŁY. Na 15 parach, które człowiek zdążył
# ocenić, szerokość mordy dzieli je niemal bezbłędnie:
#
#     przyjęte  (6): 100, 148, 172, 189, 465, 494 px
#     odrzucone (9):  31,  32,  37,  48,  64,  66,  84, 112, 117 px
#
# Żadna przyjęta para nie miała mordy poniżej 100 px, a siedem z dziewięciu
# odrzuconych leży poniżej tej wartości. Przy progu 30 px kolejka napełniała się
# więc materiałem, który anotator i tak wyrzuca — z 17 par odrzucił 11.
#
# Drugi, niezależny pomiar wskazuje tę samą liczbę: paczka robocza zmniejsza
# kadr medianowo 0.59 raza, więc morda 100 px w klatce daje anotatorowi około
# 59 px na ekranie — i to jest granica, na której da się jeszcze odczytać ucho
# i pysk. Poniżej pozycji uszu nie widać, a wtedy werdykt brzmi „nie oceniam".
#
# Zastrzeżenie: piętnaście par to mało i próg wolno podnieść dopiero wtedy, gdy
# potwierdzi go większa próbka. Wybrano wartość równą MINIMUM zbioru przyjętego,
# a nie jego medianie, właśnie dlatego, że próbka jest mała — ten wybór nie
# odrzuca żadnej pary, którą człowiek uznał za dobrą.
#
# CENA tego progu, zmierzona na 1477 parach-kandydatach z 389 nagrań nowego
# przebiegu (mediana szerokości mordy u kandydata: 94 px, więc próg tnie tuż
# POWYŻEJ mediany):
#
#     próg   pary   w przeliczeniu na 1601 nagrań
#      30   1080          ~4445
#      50    920          ~3786
#      70    804          ~3309
#      80    738          ~3037
#      90    679          ~2795
#     100    569          ~2342
#     120    474          ~1951
#
# Krzywa jest GŁADKA — nie ma progu, poniżej którego kolejka nagle rośnie.
# Pierwszy pomiar (na 131 parach z 66 nagrań) pokazywał urwisko między 80 a 90
# i dawał prognozy rzędu 600 par; jedno i drugie było artefaktem małej próbki
# złożonej z pierwszych przetworzonych nagrań, a te akurat były pionowymi
# klipami z psem w pełnej postaci. Nie warto czytać tej krzywej z mniej niż
# kilkuset par.
#
# Próg 100 pochodził z 15 pierwszych ocenionych par i te 15 par go NIE POTWIERDZA,
# gdy policzyć je ponownie na dzisiejszym dzienniku (29.08.2026, 55 par z werdyktem
# i zmierzoną mordą):
#
#   szerokość     par   przyjęte przez człowieka   punkty uznane za dobre
#     0-80 px      15            73%                       100%
#    80-100 px      4           100%                       100%
#   100-130 px     10            90%                        89%
#    130+ px       26            90%                       100%
#
# Człowiek przyjmuje wąskie mordy niemal tak samo chętnie jak szerokie, a AU
# oznacza bez „nie widać" — czyli 100 px odcinało materiał, który anotator
# umiał ocenić.
#
# Dolną granicę wyznacza natomiast SZUM POMIARU: 46 punktów na wąskiej mordzie
# leży na kilkunastu pikselach, więc drganie o 1 px jest procentowo ogromne.
# Zmierzona mediana szumu ratio AU wobec progu aktywacji 0.15:
#
#     0-40 px   0.393      40-60 px   0.342      60-80 px   0.278
#    80-100 px  0.245     100-130 px  0.225       200+ px   0.215
#
# Krzywa wypłaszcza się dopiero koło 130 px, ale między 100 a 70 rośnie łagodnie,
# a poniżej 60 skacze. Stąd 70: kolejka rośnie o 36% (2945 -> 4009 par), a szum
# rośnie o 24% wobec poziomu, który i tak jest wysoki w CAŁYM zbiorze.
#
# Asymetria kosztów zostaje w mocy i dlatego NIE schodzimy niżej: za wysoki próg
# cofa się za darmo (surowy materiał leży w `annotations.json`, ponowna kuracja
# to minuty i nie wymaga powtarzania przebiegu przez modele), za niski płaci się
# godzinami pracy człowieka — przy progu 30 px anotator odrzucił 11 par z 17.
REVIEW_MIN_FACE_WIDTH: float = 70.0

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
        peak_name: Ścieżka klatki szczytowej — ten sam klucz, którym dziennik
            werdyktów identyfikuje parę (`pair_key`)
        video: Klucz nagrania źródłowego (katalog klatek)
        signal: Ile AU przewyższa szum treku — po tym układa się kolejka
        ambiguity: Ile AU leży w pasie niepewności — rozstrzyga przy równym sygnale
        peak_quality: Ocena jakości klatki szczytowej
        neutral_quality: Ocena jakości klatki neutralnej
    """

    peak: dict
    neutral: dict
    peak_name: str
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
                peak_name=images[peak["image_id"]]["file_name"],
                video=video_key(images[peak["image_id"]]["file_name"]),
                signal=signal_score(peak),
                ambiguity=ambiguity_score(peak),
                peak_quality=peak_quality,
                neutral_quality=neutral_quality,
            )
        )
    return pairs, dict(rejected)


def with_legacy(coco: dict, published: dict) -> dict:
    """
    Dokłada do surowego COCO te klatki z wydanej kolejki, których w nim nie ma.

    Zbiór bywa generowany na nowo i wtedy `annotations.json` nie zawiera już
    par sprzed przebiegu — zmierzone 29.08.2026: z 4976 par wydanej kolejki
    1007 nie miało odpowiednika w dzisiejszym surowym COCO, a niosły 554
    werdykty. Pełne klatki tych par LEŻĄ na dysku, brakowało tylko wpisów.

    `published` musi być już przeliczona do pełnych klatek (`unpack_queue`),
    inaczej dołożylibyśmy współrzędne w układzie wycinka.

    Args:
        coco: Surowy zbiór po anotacji wsadowej
        published: Wydana kolejka we współrzędnych pełnych klatek

    Returns:
        Zbiór źródłowy uzupełniony o brakujące klatki
    """
    known = {image["file_name"] for image in coco["images"]}
    next_image_id = max((image["id"] for image in coco["images"]), default=0) + 1
    next_annotation_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

    merged = {key: value for key, value in coco.items() if key not in ("images", "annotations")}
    merged["images"] = list(coco["images"])
    merged["annotations"] = list(coco["annotations"])

    # Wskazania rozwiązujemy po NAZWIE klatki, nie po numerze: klatka neutralna
    # pary bywa już obecna w surowym COCO pod INNYM identyfikatorem. Pierwsza
    # wersja pomijała takie pary i gubiła 66 werdyktów — a wystarczy wskazać
    # ten obraz, który już leży w zbiorze.
    by_name = {image["file_name"]: image["id"] for image in merged["images"]}
    published_names = {image["id"]: image["file_name"] for image in published.get("images", [])}

    for image in published.get("images", []):
        if image["file_name"] in known:
            continue
        copied = dict(image)
        copied["id"] = next_image_id
        by_name[copied["file_name"]] = next_image_id
        next_image_id += 1
        merged["images"].append(copied)
        known.add(copied["file_name"])

    # Jedna klatka bywa szczytową w jednej parze i neutralną w innej, więc
    # dopuszczamy po jednym wierszu NA ROLĘ, a nie na klatkę. Ograniczenie do
    # jednego wiersza gubiło klatkę neutralną pary, gdy ta sama klatka weszła
    # wcześniej jako szczytowa — 62 pary z werdyktami przepadały w ten sposób.
    added_rows: set[tuple[str, str]] = set()
    raw_images = {a["image_id"] for a in coco["annotations"]}
    for annotation in published.get("annotations", []):
        name = published_names.get(annotation["image_id"])
        role = annotation.get("frame_role", "")
        if name is None or (name, role) in added_rows:
            continue
        target = by_name.get(name)
        if target is None:
            continue
        # Wiersz dla klatki, która i tak jest w surowym COCO, tylko dublowałby
        # istniejącą anotację — bierzemy wyłącznie te, których tam brakuje.
        if target in raw_images:
            continue
        neutral_name = published_names.get(annotation.get("neutral_frame_id"))
        neutral_target = by_name.get(neutral_name) if neutral_name else None
        if annotation.get("neutral_frame_id") is not None and neutral_target is None:
            continue
        moved = dict(annotation)
        moved["id"] = next_annotation_id
        next_annotation_id += 1
        moved["image_id"] = target
        if neutral_target is not None:
            moved["neutral_frame_id"] = neutral_target
        merged["annotations"].append(moved)
        added_rows.add((name, role))

    # Para bez wiersza NEUTRALNEGO nie da się złożyć. Zdarza się, gdy klatka
    # neutralna istnieje już w surowym COCO, ale opisana tam jest jako szczytowa
    # innego treku — wtedy wiersz neutralny trzeba dołożyć osobno. Bez tego
    # 62 pary z werdyktami wypadały z kolejki mimo obecności obu klatek.
    have_neutral = {
        annotation["image_id"]
        for annotation in merged["annotations"]
        if annotation.get("frame_role") == FRAME_ROLE_NEUTRAL
    }
    published_neutrals = {
        annotation["image_id"]: annotation
        for annotation in published.get("annotations", [])
        if annotation.get("frame_role") == FRAME_ROLE_NEUTRAL
    }
    by_id = {image["id"]: image["file_name"] for image in merged["images"]}
    for annotation in list(merged["annotations"]):
        neutral_id = annotation.get("neutral_frame_id")
        if annotation.get("frame_role") != FRAME_ROLE_PEAK:
            continue
        if neutral_id is None or neutral_id in have_neutral:
            continue
        source = next(
            (
                row
                for image_id, row in published_neutrals.items()
                if published_names.get(image_id) == by_id.get(neutral_id)
            ),
            None,
        )
        if source is None:
            continue
        copied = dict(source)
        copied["id"] = next_annotation_id
        next_annotation_id += 1
        copied["image_id"] = neutral_id
        copied["neutral_frame_id"] = neutral_id
        merged["annotations"].append(copied)
        have_neutral.add(neutral_id)

    return merged


def pairs_for_names(coco: dict, names: set[str], thresholds: QualityThresholds) -> list[ReviewPair]:
    """
    Buduje pary dla WSKAZANYCH klatek szczytowych, z pominięciem bramki.

    Służy do zachowania par już wydanych anotatorom, gdy dzisiejsza bramka
    by ich nie przepuściła. Bramka bywa zaostrzana i luzowana; werdykt raz
    postawiony ma przeżyć jedno i drugie.

    KLUCZOWE: pary odtwarzamy z surowego COCO, a NIE przepisujemy z opublikowanej
    kolejki. Opublikowana kolejka to paczka robocza — jej współrzędne są w
    układzie WYCINKA (obrazy 512 px), a nie pełnej klatki. Przepisane wprost,
    trafiały do ponownego pakowania jako współrzędne pełnej klatki i wycinek
    lądował w lewym górnym rogu: anotator dostawał kadr podłogi z punktami na
    drzwiach zamiast psa. Zmierzone 29.08.2026 na 1818 odziedziczonych parach.

    Args:
        coco: Surowy zbiór po anotacji wsadowej (współrzędne pełnej klatki)
        names: Ścieżki klatek szczytowych, które mają zostać
        thresholds: Progi — służą tylko do opisu jakości, nie do odsiewu

    Returns:
        Pary dla tych klatek, które udało się odtworzyć
    """
    images = {image["id"]: image for image in coco["images"]}
    neutrals = {
        ann["image_id"]: ann
        for ann in coco["annotations"]
        if ann.get("frame_role") == FRAME_ROLE_NEUTRAL
    }
    found: list[ReviewPair] = []
    for peak in coco["annotations"]:
        if peak.get("frame_role") != FRAME_ROLE_PEAK:
            continue
        if images[peak["image_id"]]["file_name"] not in names:
            continue
        neutral = neutrals.get(peak.get("neutral_frame_id"))
        if neutral is None:
            continue
        found.append(
            ReviewPair(
                peak=peak,
                neutral=neutral,
                peak_name=images[peak["image_id"]]["file_name"],
                video=video_key(images[peak["image_id"]]["file_name"]),
                signal=signal_score(peak),
                ambiguity=ambiguity_score(peak),
                peak_quality=assess_frame(peak.get("keypoints"), thresholds),
                neutral_quality=assess_frame(neutral.get("keypoints"), thresholds),
            )
        )
    return found


def peak_names(queue: dict) -> set[str]:
    """
    Wyciąga ścieżki klatek szczytowych z kolejki.

    Args:
        queue: Kolejka w formacie COCO

    Returns:
        Ścieżki klatek szczytowych — te same wartości, którymi posługuje się
        `pair_key` w dzienniku werdyktów
    """
    images = {image["id"]: image["file_name"] for image in queue.get("images", [])}
    # Rolę bierzemy z `frame_role`, a NIE z porównania `neutral_frame_id`
    # z własnym `image_id`. Część wierszy szczytowych wskazuje samą siebie
    # (klatka szczytowa bywa zarazem bazą AU swojego treku) i heurystyka
    # uznawała je za neutralne — 59 par z werdyktami wypadało wtedy z listy
    # do zachowania, choć w kolejce stały jako szczytowe.
    return {
        images[annotation["image_id"]]
        for annotation in queue.get("annotations", [])
        if annotation.get("frame_role") == FRAME_ROLE_PEAK
    }


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


def count_pairs(coco: dict) -> int:
    """
    Liczy pary w kolejce, czyli WIERSZE KLATEK SZCZYTOWYCH.

    Nie wolno tu dzielić liczby wierszy przez dwa: klatka neutralna jest
    wspólna dla wszystkich szczytów jednego treku, więc wierszy neutralnych
    jest mniej niż par. Dzielenie zaniżało meldunek — po obniżeniu progu
    pokazywało 3528 par przy faktycznych 5008.

    Args:
        coco: Kolejka w formacie COCO

    Returns:
        Liczba par
    """
    return sum(
        1
        for annotation in coco.get("annotations", [])
        if annotation.get("neutral_frame_id") not in (None, annotation["image_id"])
    )


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
        "--keep",
        default=None,
        help=(
            "Kolejka JUZ WYDANA anotatorom - jej pary zostaja niezaleznie od "
            "kuracji. Bez tego przebudowa gubi pary, do ktorych przypiete sa "
            "cudze werdykty."
        ),
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
    # Kolejka wydana anotatorom jest PACZKĄ — współrzędne ma w układzie wycinka.
    # Bez odwrócenia trafiłaby do pakowania po raz drugi i wycinek wylądowałby
    # w lewym górnym rogu pełnej klatki: kadr podłogi zamiast psa.
    zachowane: set[str] = set()
    if args.keep:
        published = load_dataset(Path(args.keep))
        if is_packed(published):
            published = unpack_queue(published, Path(args.dataset).parent / "frames")
            print("Kolejka bazowa byla spakowana — przeliczona do pelnych klatek")
        zachowane = peak_names(published)
        przed = len(coco["images"])
        coco = with_legacy(coco, published)
        if len(coco["images"]) > przed:
            print(f"Dolozono {len(coco['images']) - przed} klatek spoza surowego COCO")

    pairs, rejected = build_pairs(coco, thresholds)

    if zachowane:
        maja = {pair.peak_name for pair in pairs}
        odzyskane = pairs_for_names(coco, zachowane - maja, thresholds)
        pairs = pairs + odzyskane
        print(
            f"Zachowano kolejke wydana anotatorom: {len(zachowane)} par "
            f"({len(odzyskane)} odzyskanych spod dzisiejszej bramki), "
            f"nowych {len(maja - zachowane)}"
        )

    ordered = order_for_review(pairs)
    if args.limit is not None:
        ordered = ordered[: args.limit]

    curated = build_curated(coco, ordered)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(curated, handle, ensure_ascii=False)

    print_summary(pairs, rejected, total_peaks)
    print(f"\nZapisano {len(curated['annotations'])} par do {out_path}")


if __name__ == "__main__":
    main()
