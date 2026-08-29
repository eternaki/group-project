"""
Odwracanie spakowania: z układu wycinka z powrotem do pełnej klatki.

Paczka robocza (`build_work_pack`) tnie kadr wokół psa i PRZELICZA do niego
wszystkie współrzędne, zapisując użyty wycinek w polach `source_bbox`
i `source_scale`. Dzięki temu operacja jest odwracalna — i ta odwracalność
okazuje się warunkiem, żeby nie stracić cudzej pracy.

Po co to potrzebne. Kolejka wydana anotatorom jest PACZKĄ, więc jej współrzędne
są w układzie wycinka. Podana wprost jako baza do ponownej kuracji trafia do
pakowania po raz drugi: pakowanie traktuje te małe liczby jak współrzędne
pełnej klatki i wycina lewy górny róg. Zmierzone 29.08.2026 — anotator
dostawał kadr podłogi z punktami na drzwiach zamiast psa, a dotyczyło to
wszystkich par odziedziczonych po poprzedniej kolejce.

Odtworzenie z surowego COCO nie jest tu wyjściem: 1007 par z opublikowanej
kolejki pochodzi ze starszego pokolenia zbioru i w dzisiejszym
`annotations.json` ich nie ma, a niosą 554 werdykty. Pełne klatki tych par
LEŻĄ na dysku — brakuje tylko współrzędnych, i właśnie je ten moduł odzyskuje.
"""

from pathlib import Path
from typing import Optional

from scripts.annotation.cropping import read_image

# Pola zapisane przez `build_work_pack`, po których poznajemy spakowany obraz
SOURCE_BBOX: str = "source_bbox"
SOURCE_SCALE: str = "source_scale"


def _unscale(values: list[float], origin: tuple[float, float], scale: float) -> list[float]:
    """
    Przenosi płaską listę współrzędnych z układu wycinka do pełnej klatki.

    Args:
        values: Płaska lista (x, y, widoczność) × N
        origin: Lewy górny róg wycinka w pełnej klatce
        scale: Skala nałożona przy pakowaniu

    Returns:
        Nowa lista w układzie pełnej klatki
    """
    out = list(values)
    for i in range(0, len(out) - 2, 3):
        out[i] = out[i] / scale + origin[0]
        out[i + 1] = out[i + 1] / scale + origin[1]
    return out


def unpack_queue(queue: dict, frames_dir: Path) -> dict:
    """
    Przelicza kolejkę z układu wycinków z powrotem do pełnych klatek.

    Obrazy bez `source_bbox` zostawiamy bez zmian — nie były pakowane.
    Obraz, którego pełnej klatki nie ma na dysku, też zostaje nietknięty:
    lepiej zostawić wpis, jaki jest, niż podać zmyślone wymiary.

    Args:
        queue: Kolejka w formacie COCO (po spakowaniu)
        frames_dir: Katalog pełnych klatek

    Returns:
        Kolejka we współrzędnych pełnych klatek
    """
    origins: dict[int, tuple[tuple[float, float], float]] = {}

    out_images: list[dict] = []
    for image in queue.get("images", []):
        bbox = image.get(SOURCE_BBOX)
        scale = image.get(SOURCE_SCALE)
        full = _full_size(frames_dir / image["file_name"])
        if not bbox or not scale or full is None:
            out_images.append(dict(image))
            continue
        restored = dict(image)
        restored["width"], restored["height"] = full
        restored.pop(SOURCE_BBOX, None)
        restored.pop(SOURCE_SCALE, None)
        out_images.append(restored)
        origins[image["id"]] = ((float(bbox[0]), float(bbox[1])), float(scale))

    out_annotations: list[dict] = []
    for annotation in queue.get("annotations", []):
        moved = dict(annotation)
        found = origins.get(annotation["image_id"])
        if found is not None:
            origin, scale = found
            if moved.get("keypoints"):
                moved["keypoints"] = _unscale(moved["keypoints"], origin, scale)
            if moved.get("bbox"):
                x, y, w, h = (float(v) for v in moved["bbox"])
                moved["bbox"] = [
                    x / scale + origin[0],
                    y / scale + origin[1],
                    w / scale,
                    h / scale,
                ]
        out_annotations.append(moved)

    restored_queue = {
        key: value for key, value in queue.items() if key not in ("images", "annotations")
    }
    restored_queue["images"] = out_images
    restored_queue["annotations"] = out_annotations
    return restored_queue


def _full_size(path: Path) -> Optional[tuple[int, int]]:
    """
    Odczytuje wymiary pełnej klatki.

    Args:
        path: Ścieżka do klatki

    Returns:
        Para (szerokość, wysokość) albo None, gdy klatki nie ma
    """
    image = read_image(path)
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def is_packed(queue: dict) -> bool:
    """
    Mówi, czy kolejka jest spakowana (współrzędne w układzie wycinka).

    Args:
        queue: Kolejka w formacie COCO

    Returns:
        Czy choć jeden obraz niesie ślad pakowania
    """
    return any(image.get(SOURCE_BBOX) for image in queue.get("images", []))
