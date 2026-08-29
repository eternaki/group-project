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
    next_order = max(
        (
            annotation["review_order"]
            for annotation in merged["annotations"]
            if annotation.get("review_order") is not None
        ),
        default=-1,
    )

    # Wiersze opisujące klatki neutralne, pod identyfikatorem ich obrazu.
    # Para potrzebuje ICH TEŻ — bez wiersza neutralnego stanowisko nie ma z czego
    # zbudować lewej strony porównania.
    neutral_rows = {
        annotation["image_id"]: annotation
        for annotation in extra.get("annotations", [])
        if not _is_peak(annotation)
    }

    # Klatka neutralna jest WSPÓLNA dla wszystkich szczytów jednego treku, więc
    # o powtórzeniu decyduje wyłącznie klatka SZCZYTOWA. Wcześniejsza wersja
    # odrzucała parę, gdy w kolejce była już KTÓRAKOLWIEK z jej klatek — czyli
    # drugi i każdy następny szczyt tego samego psa przepadał, mimo że opisuje
    # inną chwilę mimiki. Zmierzone 29.08.2026: z 3230 par czekających na
    # dołożenie weszło 231, a 2429 odpadło właśnie na wspólnej klatce neutralnej.
    frame_to_id = {image[FRAME_KEY]: image["id"] for image in merged["images"]}

    added = 0
    for annotation in extra.get("annotations", []):
        if not _is_peak(annotation):
            continue
        frames = _pair_frames(annotation, extra_images)
        if len(frames) < 2:
            continue
        peak_frame, neutral_frame = frames[0], frames[1]
        if peak_frame[FRAME_KEY] in known_frames:
            continue
        if limit is not None and added >= limit:
            break

        # OBRAZ klatki neutralnej jest współdzielony (nie dublujemy pliku),
        # ale WIERSZ musi być własny dla każdej pary. Stanowisko składa parę
        # po `review_order` (`coco_import._group_pairs`), więc jeden wiersz
        # neutralny może obsłużyć dokładnie jeden numer — przy współdzieleniu
        # wiersza wszystkie szczyty poza jednym zostają bez klatki neutralnej
        # i para po cichu znika z kolejki.
        neutral_id = frame_to_id.get(neutral_frame[FRAME_KEY])
        if neutral_id is None:
            neutral_copy = dict(neutral_frame)
            neutral_id = next_id
            neutral_copy["id"] = next_id
            next_id += 1
            merged["images"].append(neutral_copy)
            known_frames.add(neutral_copy[FRAME_KEY])
            frame_to_id[neutral_copy[FRAME_KEY]] = neutral_id

        peak_copy = dict(peak_frame)
        peak_id = next_id
        peak_copy["id"] = next_id
        next_id += 1
        merged["images"].append(peak_copy)
        known_frames.add(peak_copy[FRAME_KEY])
        frame_to_id[peak_copy[FRAME_KEY]] = peak_id

        # Numer porządkowy MUSI być świeży. Obie kolejki numerują od zera, więc
        # przepisany numer zderza się z numerem pary już obecnej — a wtedy
        # `_group_pairs` nadpisuje wpis i zestawia szczyt jednego psa z klatką
        # neutralną DRUGIEGO. Zmierzone: 1840 zderzeń, numer 0 w trzech parach.
        next_order += 1
        # Braku wiersza neutralnego NIE zastępujemy namiastką: wyszłaby anotacja
        # bez `bbox`, na której składanie paczki wywraca się KeyError-em daleko
        # od miejsca powstania. Para bez kompletu wierszy po prostu nie wchodzi.
        neutral_row = neutral_rows.get(neutral_frame["id"])
        if neutral_row is None:
            continue
        for row, image_id in ((neutral_row, neutral_id), (annotation, peak_id)):
            moved = dict(row)
            moved["id"] = next_annotation_id
            next_annotation_id += 1
            moved["image_id"] = image_id
            moved["neutral_frame_id"] = neutral_id
            moved["review_order"] = next_order
            merged["annotations"].append(moved)
        added += 1

    return merged


def renumber_queue(coco: dict) -> dict:
    """
    Nadaje każdej parze WŁASNY `review_order` i własny wiersz klatki neutralnej.

    Stanowisko skleja parę po `review_order` (`coco_import._group_pairs`), więc
    numer powtórzony znaczy, że jeden wpis NADPISUJE drugi — szczyt jednego psa
    dostaje wtedy klatkę neutralną innego, a część par znika z kolejki. Zmierzone
    29.08.2026 na kolejce z `main`: 7057 wierszy miało tylko 3585 numerów,
    stanowisko złożyło 1628 par z 5008, a 1222 z nich były sparowane błędnie.

    Naprawa jest możliwa bez utraty czegokolwiek, bo psuje się WYŁĄCZNIE
    numeracja: pole `neutral_frame_id` w każdym wierszu szczytowym wskazywało
    właściwą klatkę (sprawdzone: 0 błędów na 5008 par). Obrazy zostają nietknięte
    razem ze ścieżkami, więc `pair_key` w dzienniku dalej pasuje.

    Kolejność zachowujemy po dotychczasowym numerze, żeby anotator wrócił mniej
    więcej tam, gdzie skończył.

    Args:
        coco: Kolejka w formacie COCO

    Returns:
        Kolejka z poprawną numeracją i po dwa wiersze na parę
    """
    neutral_templates = {
        annotation["image_id"]: annotation
        for annotation in coco.get("annotations", [])
        if not _is_peak(annotation)
    }
    peaks = [
        annotation for annotation in coco.get("annotations", []) if _is_peak(annotation)
    ]
    peaks.sort(key=lambda a: (a.get("review_order") or 0, a["image_id"]))

    rows: list[dict] = []
    annotation_id = 1
    for order, peak in enumerate(peaks):
        neutral_id = peak["neutral_frame_id"]
        template = neutral_templates.get(neutral_id)
        if template is None:
            continue
        for row, image_id in ((template, neutral_id), (peak, peak["image_id"])):
            moved = dict(row)
            moved["id"] = annotation_id
            annotation_id += 1
            moved["image_id"] = image_id
            moved["neutral_frame_id"] = neutral_id
            moved["review_order"] = order
            rows.append(moved)

    repaired = {key: value for key, value in coco.items() if key != "annotations"}
    repaired["annotations"] = rows
    return repaired
