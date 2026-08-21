#!/usr/bin/env python3
"""
Paczka robocza — to, co koledzy dostają samym `git clone`.

    python -m scripts.annotation.build_work_pack

Materiał, z którego powstaje zbiór, leży poza repozytorium: klatki ważą 1.3 GB,
a surowe COCO 63 MB. Osoba, która sklonuje repozytorium, widziała więc PUSTĄ
listę zbiorów — narzędzie startowało, ale nie było czego anotować. Każde nowe
stanowisko wymagało ręcznego przesłania gigabajtów, a przy każdej ponownej
kuracji — przesłania jeszcze raz.

Ten skrypt składa z materiału roboczego paczkę, która mieści się w gicie:

    kuracja + pełne klatki  ->  data/<zbior>/work/{curated.json, frames/}

Kadrujemy wokół CAŁEGO psa, nie wokół mordy. Stanowisko i tak liczy sobie kadr
mordy z punktów, więc morda jest dostępna zawsze — ale gdyby paczka niosła samą
mordę, klawisz `F` („szerszy plan") nie miałby czego pokazać, a anotator
straciłby jedyny sposób sprawdzenia, o którego psa w kadrze chodzi.

Ograniczenie 512 px na dłuższym boku i wycięcie `procrustes_keypoints` (pole,
którego nikt jeszcze nie czyta, a które da się odtworzyć z samych punktów) robią
z 1.4 GB około 68 MB. Reszta zostaje bez zmian: `au_analysis`, szum, `tfm_score`,
kolejność pracy — wszystko, na czym stoi stanowisko.
"""

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from packages.pipeline.quality_gate import assess_frame
from scripts.annotation.cropping import (
    CropBox,
    body_box,
    crop_and_scale,
    face_box,
    remap_bbox,
    remap_keypoints,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET: str = "dataset_final"

# Podkatalog paczki. Leży W ŚRODKU zbioru roboczego, tak samo jak `release/`:
# jeden katalog zbioru, w nim wejście pracy i jej wynik. Wzorce `.gitignore`
# (`data/*/frames/`, `data/*/curated.json`) sięgają jednego poziomu, więc
# materiał roboczy zostaje ignorowany, a paczka — nie.
WORK_DIRNAME: str = "work"
FRAMES_DIRNAME: str = "frames"
# Paczka niesie plik POD NAZWĄ KURACJI, bo po tym wzorcu backend rozpoznaje
# zbiór gotowy do weryfikacji (`CURATED_PATTERN`). Nazwa `annotations.json`
# kazałaby stanowisku kurować paczkę jeszcze raz — a ona jest już po kuracji.
PACK_NAME: str = "curated.json"
CURATED_NAME: str = "curated.json"

# Zapas wokół boksu psa — tyle samo, ile pokazuje stanowisko przy podglądzie psa
BODY_MARGIN: float = 0.12

# Zapas wokół rozpiętości punktów. Mniejszy niż przy boksie psa, bo tu chodzi
# wyłącznie o to, żeby żaden punkt nie wypadł poza zdjęcie — nie o kontekst.
KEYPOINT_MARGIN: float = 0.05

# Dłuższy bok kadru. Mediana boksu psa to 596 px, więc 512 prawie nic nie obcina,
# a zbija paczkę z 264 MB do 53 MB.
MAX_CROP_SIDE: int = 512

# Ile pikseli MORDY paczka ma zachować.
#
# Sam limit `MAX_CROP_SIDE` liczy się od boksu CAŁEGO psa, a mierzy się nim
# rzecz, której anotator w ogóle nie ocenia. Zmierzone na kuracji: zmniejszenie
# wychodzi medianowo 0.59 raza, więc morda mająca w klatce 112 px (mediana)
# dociera do stanowiska jako 61 px, a 73.6% mord jest w paczce węższych niż
# 80 px. Informacja o pozycji uszu i pyska JEST w nagraniu i ginie dopiero tutaj.
#
# Stąd drugi warunek na skalę: kadr wolno zmniejszyć najwyżej tak, żeby morda
# zachowała `TARGET_FACE_PX`. Wartość odpowiada progowi kuracji
# (`REVIEW_MIN_FACE_WIDTH`), więc para dopuszczona do kolejki trafia do paczki
# bez utraty szczegółu mordy.
TARGET_FACE_PX: float = 100.0

# Twarda granica rozrostu kadru. Pies zajmujący pół klatki przy małej mordzie
# (profil, pysk odwrócony) kazałby zachować kadr w pełnej rozdzielczości i
# paczka urosłaby o rząd wielkości. Powyżej tej wartości rezygnujemy z pikseli
# mordy — takie pary i tak odrzuca bramka jakości na obrocie głowy.
HARD_MAX_CROP_SIDE: int = 1024

# Jakość JPEG. O dwa stopnie niżej niż w zbiorze do oddania: paczka jest
# materiałem do PATRZENIA, nie do publikacji, a różnicy na oko nie widać.
JPEG_QUALITY: int = 88

# Pole liczone przy anotacji wsadowej, którego nie czyta ani backend, ani żaden
# skrypt — a waży 138 liczb na anotację, czyli 6.5 MB na całą kurację. Da się je
# odtworzyć z samych keypoints, więc do paczki nie jedzie.
DROPPED_FIELDS: tuple[str, ...] = ("procrustes_keypoints",)


@dataclass
class PackStats:
    """Licznik tego, co weszło do paczki i co odpadło."""

    images_written: int = 0
    annotations_written: int = 0
    skipped_missing: int = 0
    skipped_no_box: int = 0
    bytes_written: int = 0


def _annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    """
    Grupuje anotacje po obrazie — WSZYSTKIE, łącznie z powtórzeniami.

    Kuracja powiela anotację klatki neutralnej: jedna neutralna obsługuje kilka
    peaków tego samego psa i dostaje osobny wpis przy KAŻDYM z nich. Wygląda to
    na nadmiarowość, ale nią nie jest — sesja anotacji składa pary z par (klatka
    neutralna, klatka szczytowa) po `track_id`, więc para bez własnego wpisu
    neutralnego nie ma z czego powstać i wypada z kolejki.

    Odsianie tych powtórzeń kosztowało 599 z 1111 wpisów neutralnych i zabrało
    z kolejki ponad połowę par (u jednej osoby 263 na karcie wobec 114 realnie).
    Powtarzają się TYLKO wpisy w JSON — obraz na dysku i tak zapisujemy raz.

    Args:
        coco: Wczytana kuracja

    Returns:
        Mapa `image_id` → anotacje tego obrazu
    """
    grouped: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        grouped.setdefault(annotation["image_id"], []).append(annotation)
    return grouped


def _pack_box(annotations: list[dict], width: int, height: int) -> Optional[CropBox]:
    """
    Wyznacza jeden kadr obejmujący wszystkie psy tego obrazu.

    Jeden obraz bywa dzielony przez kilka treków. Wycinanie go osobno dla
    każdego psa dublowałoby pliki, a wycięcie tylko pierwszego ucięłoby resztę
    — więc kadr obejmuje sumę boksów.

    Args:
        annotations: Anotacje tego obrazu
        width: Szerokość pełnej klatki
        height: Wysokość pełnej klatki

    Returns:
        Kadr albo None, gdy żaden boks nie jest sensowny
    """
    seen: set[tuple[float, ...]] = set()
    unique = []
    for annotation in annotations:
        key = tuple(annotation["bbox"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(annotation)

    # Suma boksu psa I rozpiętości punktów. Sam boks psa nie wystarcza: detektor
    # ciał i model punktów to dwa niezależne pomiary i potrafią się rozjechać.
    # Zmierzone na tym zbiorze — przy kadrze po samym boksie 91 anotacji (4.1%)
    # miało punkty POZA zdjęciem, a jedna wszystkie 46. Anotator nie widziałby
    # ich wcale, więc nie mógłby ich poprawić ani uznać kadru za zły.
    boxes = [
        box
        for annotation in unique
        for box in (
            body_box(annotation["bbox"], width, height, BODY_MARGIN),
            face_box(annotation["keypoints"], width, height, KEYPOINT_MARGIN),
        )
        if box is not None
    ]
    if not boxes:
        return None
    return CropBox(
        x0=min(box.x0 for box in boxes),
        y0=min(box.y0 for box in boxes),
        x1=max(box.x1 for box in boxes),
        y1=max(box.y1 for box in boxes),
    )


def _repacked(annotation: dict, box: CropBox, scale: float) -> dict:
    """
    Przepisuje anotację do układu wycinka, odchudzając ją o pola nieczytane.

    Args:
        annotation: Anotacja z kuracji
        box: Kadr obrazu
        scale: Skala nałożona po wycięciu

    Returns:
        Nowa anotacja
    """
    packed = {key: value for key, value in annotation.items() if key not in DROPPED_FIELDS}
    packed["keypoints"] = remap_keypoints(annotation["keypoints"], box, scale)
    packed["bbox"] = remap_bbox(annotation["bbox"], box, scale)
    packed["area"] = packed["bbox"][2] * packed["bbox"][3]
    return packed


def _within_limit(annotations: list[dict], limit: Optional[int]) -> bool:
    """
    Rozstrzyga, czy obraz należy do pierwszych `limit` par kolejki.

    Kuracja numeruje pary polem `review_order`, a klatka neutralna nosi numer
    KAŻDEJ pary, którą obsługuje. Bierzemy więc numer najmniejszy: neutralna
    wchodzi do paczki razem z pierwszym peakiem, który jej potrzebuje, i nie
    wypada przez to, że któraś z dalszych par nie zmieściła się w limicie.

    Args:
        annotations: Anotacje tego obrazu
        limit: Ile par zostawić; None znaczy „wszystkie"

    Returns:
        Czy obraz wchodzi do paczki
    """
    if limit is None:
        return True
    orders = [
        annotation["review_order"]
        for annotation in annotations
        if annotation.get("review_order") is not None
    ]
    # Brak numeru znaczy kurację starszą niż to pole — wtedy nie ma po czym
    # ciąć i zostawiamy obraz, zamiast po cichu opróżnić paczkę.
    return min(orders) < limit if orders else True


def build_pack(
    dataset_dir: Path, output: Path, limit: Optional[int] = None
) -> PackStats:
    """
    Składa paczkę roboczą ze zbioru po kuracji.

    Args:
        dataset_dir: Katalog zbioru roboczego
        output: Katalog paczki
        limit: Ile pierwszych par kolejki spakować; None znaczy „całą kurację"

    Returns:
        Statystyki przebiegu

    Raises:
        SystemExit: Gdy nie ma pliku kuracji
    """
    curated_path = dataset_dir / CURATED_NAME
    if not curated_path.is_file():
        raise SystemExit(f"Brak kuracji {curated_path} — uruchom curate_for_review")

    coco = json.loads(curated_path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in coco["images"]}
    grouped = _annotations_by_image(coco)

    if output.exists():
        shutil.rmtree(output)
    (output / FRAMES_DIRNAME).mkdir(parents=True)

    stats = PackStats()
    packed_images: list[dict] = []
    packed_annotations: list[dict] = []

    for image_id, annotations in grouped.items():
        if not _within_limit(annotations, limit):
            continue
        entry = images[image_id]
        result = _write_frame(dataset_dir, output, entry, annotations)
        if result is None:
            continue
        image_entry, box, scale, size = result
        packed_images.append(image_entry)
        packed_annotations.extend(_repacked(a, box, scale) for a in annotations)
        stats.images_written += 1
        stats.annotations_written += len(annotations)
        stats.bytes_written += size

    # Liczymy względem obrazów WZIĘTYCH POD UWAGĘ, nie całej kuracji — inaczej
    # obrazy odcięte limitem raportowałyby się jako brakujące klatki.
    considered = sum(
        1 for annotations in grouped.values() if _within_limit(annotations, limit)
    )
    stats.skipped_missing = considered - stats.images_written
    _write_json(output / PACK_NAME, coco, packed_images, packed_annotations)
    return stats


def _max_side_for(annotations: list[dict], box: CropBox) -> int:
    """
    Wylicza dopuszczalny dłuższy bok kadru, tak by morda zachowała szczegół.

    Skala kadru wynosi `max_side / dłuższy_bok`, a szerokość mordy zmniejsza się
    razem z nim. Żeby morda została przy `TARGET_FACE_PX`, dłuższy bok musi
    zostać przy `dłuższy_bok * TARGET_FACE_PX / szerokość_mordy`.

    Args:
        annotations: Anotacje tego obrazu
        box: Kadr wycinany z pełnej klatki

    Returns:
        Dłuższy bok kadru w pikselach, nie mniej niż `MAX_CROP_SIDE`
        i nie więcej niż `HARD_MAX_CROP_SIDE`
    """
    widths = [
        assess_frame(annotation["keypoints"]).face_width
        for annotation in annotations
        if annotation.get("keypoints")
    ]
    face = max(widths, default=0.0)
    # Brak punktów albo zdegenerowana morda: nie ma czego chronić, zostaje limit
    # liczony od boksu psa.
    if face <= 0:
        return MAX_CROP_SIDE

    longest = max(box.width, box.height)
    needed = longest * (TARGET_FACE_PX / face)
    return int(min(HARD_MAX_CROP_SIDE, max(MAX_CROP_SIDE, needed)))


def _write_frame(
    dataset_dir: Path, output: Path, entry: dict, annotations: list[dict]
) -> Optional[tuple[dict, CropBox, float, int]]:
    """
    Wycina i zapisuje jedną klatkę paczki.

    Args:
        dataset_dir: Katalog zbioru roboczego
        output: Katalog paczki
        entry: Wpis obrazu z kuracji
        annotations: Anotacje tego obrazu

    Returns:
        Czwórka (wpis obrazu, kadr, skala, rozmiar pliku) albo None
    """
    source = dataset_dir / FRAMES_DIRNAME / entry["file_name"]
    image = cv2.imread(str(source)) if source.is_file() else None
    if image is None:
        return None

    box = _pack_box(annotations, image.shape[1], image.shape[0])
    if box is None:
        return None

    crop, scale = crop_and_scale(image, box, _max_side_for(annotations, box))
    target = output / FRAMES_DIRNAME / entry["file_name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        return None

    packed = dict(entry)
    packed["width"] = int(crop.shape[1])
    packed["height"] = int(crop.shape[0])
    # Skąd wycięto — bez tego nie da się wrócić do pełnej klatki z nagrania.
    packed["source_bbox"] = [box.x0, box.y0, box.width, box.height]
    packed["source_scale"] = scale
    return packed, box, scale, target.stat().st_size


def _write_json(
    path: Path, coco: dict, images: list[dict], annotations: list[dict]
) -> None:
    """
    Zapisuje COCO paczki bez wcięć — 15 MB zamiast 23 MB przy tej samej treści.

    Args:
        path: Ścieżka pliku
        coco: Oryginalna kuracja (dla `info`, `licenses`, `categories`)
        images: Wpisy obrazów paczki
        annotations: Anotacje paczki
    """
    packed = {
        "info": {**coco.get("info", {}), "description": "DogFACS — paczka robocza (kadry psów)"},
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": images,
        "annotations": annotations,
    }
    path.write_text(
        json.dumps(packed, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description="Paczka robocza do repozytorium")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Nazwa zbioru w data/")
    parser.add_argument("--data-root", default=None, help="Katalog danych (domyslnie data/)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ile pierwszych par kolejki spakowac (domyslnie: cala kuracja)",
    )
    return parser.parse_args()


def main() -> None:
    """Punkt wejścia."""
    args = parse_args()
    data_root = Path(args.data_root) if args.data_root else REPO_ROOT / "data"
    dataset_dir = data_root / args.dataset
    output = dataset_dir / WORK_DIRNAME

    stats = build_pack(dataset_dir, output, args.limit)
    logger.info("Obrazow w paczce   : %d", stats.images_written)
    logger.info("Anotacji w paczce  : %d", stats.annotations_written)
    if stats.skipped_missing:
        logger.info("Pominietych klatek : %d", stats.skipped_missing)
    logger.info("Rozmiar obrazow    : %.1f MB", stats.bytes_written / 1024 / 1024)
    logger.info(
        "Rozmiar COCO       : %.1f MB",
        (output / PACK_NAME).stat().st_size / 1024 / 1024,
    )
    logger.info("Gotowe: %s", output)


if __name__ == "__main__":
    main()
