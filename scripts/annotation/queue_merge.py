"""
Dokładanie par do kolejki weryfikacji BEZ gubienia tych, które już w niej były.

Kolejka jest PUBLIKOWANA anotatorom: `data/dataset_final/work/` idzie do gita,
ludzie ją klonują i zapisują werdykty pod `pair_key`, czyli pod ŚCIEŻKĄ KLATKI.
Wynika z tego reguła, której nie widać w kodzie kuracji: para raz wydana ludziom
musi w kolejce ZOSTAĆ. Zniknięcie jej nie psuje niczego widocznie — dziennik
w `data/labels/` przeżywa, tylko przestaje się z czymkolwiek wiązać.

Zmierzone 27.08.2026: przebudowa kolejki od zera po dołożeniu nowego materiału
wyrzuciła 1533 kadry, a razem z nimi 603 z 605 werdyktów Maszy i Danka. Kadry
leżały na dysku, werdykty leżały w gicie — a mimo to praca dwóch osób nie
miała się do czego przypiąć.

Skąd bierze się rozjazd: `data/dataset_final/annotations.json` jest w
`.gitignore`, więc surowy COCO to plik LOKALNY. Kolejka opublikowana w gicie
mogła powstać z innego jego pokolenia niż to, które akurat leży na dysku.
Dlatego bazą przy dokładaniu jest OPUBLIKOWANA kolejka, a nie ponowna kuracja.
"""

from typing import Optional

# Klucz, po którym rozpoznajemy tę samą klatkę w dwóch kolejkach.
# To ta sama wartość, która trafia do dziennika jako `pair_key`.
FRAME_KEY: str = "file_name"


def _images_by_id(coco: dict) -> dict[int, dict]:
    """
    Indeksuje obrazy po identyfikatorze.

    Args:
        coco: Zbiór w formacie COCO

    Returns:
        Obrazy pod swoimi identyfikatorami
    """
    return {image["id"]: image for image in coco.get("images", [])}


def _next_free_id(coco: dict) -> int:
    """
    Podaje pierwszy wolny identyfikator obrazu.

    Args:
        coco: Zbiór w formacie COCO

    Returns:
        Identyfikator większy od każdego użytego
    """
    used = [image["id"] for image in coco.get("images", [])]
    return max(used, default=0) + 1


def _is_peak(annotation: dict) -> bool:
    """
    Mówi, czy anotacja opisuje klatkę SZCZYTOWĄ, czyli kotwicę pary.

    Kolejka trzyma DWA wiersze na parę: klatkę neutralną i szczytową. Wiersz
    neutralny wskazuje `neutral_frame_id` na SAMEGO SIEBIE (baza AU klatki
    neutralnej to ona sama), więc wzięty za kotwicę udaje osobną parę —
    zajmuje nazwę kadru i prawdziwa para odpada potem jako „już jest".
    Tak właśnie do kolejki weszły 542 osierocone klatki neutralne.

    Args:
        annotation: Anotacja z kolejki

    Returns:
        Czy to klatka szczytowa
    """
    neutral_id = annotation.get("neutral_frame_id")
    return neutral_id is not None and neutral_id != annotation["image_id"]


def _pair_frames(annotation: dict, images: dict[int, dict]) -> list[dict]:
    """
    Zbiera obrazy tworzące parę: klatkę szczytową i jej klatkę neutralną.

    Args:
        annotation: Anotacja klatki szczytowej
        images: Obrazy zbioru źródłowego pod identyfikatorami

    Returns:
        Obrazy pary, bez powtórzeń i bez brakujących
    """
    ids = [annotation["image_id"], annotation["neutral_frame_id"]]
    return [images[image_id] for image_id in ids if image_id in images]


def merge_queues(base: dict, extra: dict, limit: Optional[int] = None) -> dict:
    """
    Dokłada do kolejki `base` te pary z `extra`, których jeszcze w niej nie ma.

    Pary z `base` zostają NIETKNIĘTE razem ze swoimi identyfikatorami — ich
    ścieżki są kluczami werdyktów, więc przenumerowanie po stronie bazy nie
    zaszkodziłoby, ale przesunięcie czegokolwiek innego już tak. Dokładane pary
    dostają nowe identyfikatory, a `neutral_frame_id` jedzie razem z `image_id`:
    rozjechane wiązałoby klatkę szczytową z klatką neutralną INNEGO psa, co
    przesuwa naraz wszystkie 21 AU i nie objawia się niczym widocznym.

    Args:
        base: Kolejka opublikowana anotatorom (ta z gita)
        extra: Świeżo skurowana kolejka, z której bierzemy nowości
        limit: Najwyżej tyle par dołożyć; None znaczy wszystkie

    Returns:
        Kolejka zawierająca wszystko z `base` plus nowe pary z `extra`
    """
    merged = {
        key: value for key, value in base.items() if key not in ("images", "annotations")
    }
    merged["images"] = list(base.get("images", []))
    merged["annotations"] = list(base.get("annotations", []))

    known_frames = {image[FRAME_KEY] for image in merged["images"]}
    extra_images = _images_by_id(extra)
    next_id = _next_free_id(base)
    next_annotation_id = max(
        (annotation["id"] for annotation in merged["annotations"]), default=0
    ) + 1

    # Wiersze opisujące klatki neutralne, pod identyfikatorem ich obrazu.
    # Para potrzebuje ICH TEŻ — bez wiersza neutralnego stanowisko nie ma z czego
    # zbudować lewej strony porównania.
    neutral_rows = {
        annotation["image_id"]: annotation
        for annotation in extra.get("annotations", [])
        if not _is_peak(annotation)
    }

    added = 0
    for annotation in extra.get("annotations", []):
        if not _is_peak(annotation):
            continue
        frames = _pair_frames(annotation, extra_images)
        if len(frames) < 2 or any(frame[FRAME_KEY] in known_frames for frame in frames):
            continue
        if limit is not None and added >= limit:
            break

        remapped: dict[int, int] = {}
        for frame in frames:
            copied = dict(frame)
            remapped[frame["id"]] = next_id
            copied["id"] = next_id
            next_id += 1
            merged["images"].append(copied)
            known_frames.add(copied[FRAME_KEY])

        for row in (neutral_rows.get(annotation["neutral_frame_id"]), annotation):
            if row is None:
                continue
            moved = dict(row)
            moved["id"] = next_annotation_id
            next_annotation_id += 1
            moved["image_id"] = remapped[row["image_id"]]
            neutral_id = row.get("neutral_frame_id")
            if neutral_id in remapped:
                moved["neutral_frame_id"] = remapped[neutral_id]
            merged["annotations"].append(moved)
        added += 1

    return merged
