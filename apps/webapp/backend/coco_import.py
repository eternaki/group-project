"""
Import zbioru COCO do sesji anotacji.

Domykający brak Sprintu 15: webapp umiał dotąd wyłącznie „wgraj wideo i przepuść
przez pipeline". Zbiór policzony wsadowo (`scripts/annotation/curate_for_review`)
nie miał jak trafić pod ręce anotatora — czyli każda weryfikacja zaczynała się od
ponownego, wielogodzinnego przeliczania tego samego materiału.

Jednostką importu jest PARA (klatka neutralna, klatka szczytowa), a nie klatka:
AU jest z definicji różnicą względem klatki neutralnej, więc bez niej nie ma
czego oceniać. Każda para dostaje własny `track_id` równy pozycji w kolejce
weryfikacji, bo `track_id` z batcha jest unikalny tylko w obrębie nagrania
(w `dataset_v2` na 1359 treków przypada 12 różnych wartości).
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from label_store import latest_by_pair
from session_store import (
    ANNOTATION_STATUS_AUTO,
    ANNOTATION_STATUS_VERIFIED,
    DogTrack,
    FrameAnnotation,
    SessionData,
)

from packages.data.coco import FRAME_ROLE_NEUTRAL, FRAME_ROLE_PEAK

# Prefiks URL, pod którym backend wystawia klatki zbioru
DATASET_URL_PREFIX: str = "/dataset"

# Długość identyfikatora sesji (jak w sesjach z wideo)
SESSION_ID_LENGTH: int = 8

_WINDOWS_SEPARATOR: str = "\\"
_POSIX_SEPARATOR: str = "/"


# Katalog, spod którego wolno importować zbiory. Ścieżka przychodzi z przeglądarki,
# więc bez tego ograniczenia endpoint czytałby dowolny plik z dysku serwera.
IMPORT_ROOT_ENV: str = "DOGFACS_IMPORT_ROOT"
DEFAULT_IMPORT_ROOT: str = "data"

# Zbiory to JSON i tylko JSON — rozszerzenie sprawdzamy, żeby ścieżka nie
# posłużyła do wyciągnięcia czegokolwiek innego, co leży w katalogu danych.
ALLOWED_SUFFIX: str = ".json"


class CocoImportError(ValueError):
    """Wyjątek, gdy zbioru nie da się zaimportować jako sesji anotacji."""


def session_id_for(path: Path) -> str:
    """
    Wylicza STAŁY identyfikator sesji dla danego zbioru.

    Identyfikator losowy znaczyłby, że każde otwarcie narzędzia zakłada nową
    sesję, a wczorajsza praca zostaje w osieroconej — anotator wracałby do
    pierwszej pary i nie miał jak odzyskać swoich werdyktów. Wyprowadzenie
    identyfikatora ze ścieżki zbioru sprawia, że ten sam zbiór to zawsze ta
    sama sesja, więc praca się kumuluje.

    Args:
        path: Ścieżka zbioru po kuracji

    Returns:
        Identyfikator sesji (skrót ścieżki)
    """
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return digest[:SESSION_ID_LENGTH]


def import_root() -> Path:
    """
    Zwraca katalog, spod którego wolno importować zbiory.

    Odczytywany przy każdym wywołaniu, a nie raz przy imporcie modułu, żeby
    dało się go podmienić w testach i w uruchomieniu zespołowym.

    Returns:
        Bezwzględna ścieżka katalogu dozwolonych zbiorów
    """
    return Path(os.environ.get(IMPORT_ROOT_ENV, DEFAULT_IMPORT_ROOT)).resolve()


def resolve_import_path(raw_path: str) -> Path:
    """
    Sprawdza ścieżkę podaną przez klienta i sprowadza ją do pliku pod korzeniem.

    Ścieżka przychodzi z przeglądarki. Bez tego sprawdzenia `../../` albo
    ścieżka bezwzględna pozwoliłyby wczytać dowolny plik z dysku serwera,
    a treść błędu parsowania odesłałaby jego fragment klientowi.

    Args:
        raw_path: Ścieżka podana przez klienta, względna wobec korzenia importu

    Returns:
        Bezwzględna ścieżka pliku leżącego pod korzeniem importu

    Raises:
        CocoImportError: Gdy ścieżka jest bezwzględna, ucieka poza korzeń,
            ma złe rozszerzenie albo nie wskazuje pliku
    """
    root = import_root()
    candidate = Path(raw_path)
    if candidate.is_absolute():
        # Ścieżka bezwzględna bywa poprawna (zespół trzyma zbiór gdzie indziej),
        # ale musi mimo wszystko leżeć pod korzeniem importu.
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()

    if root != resolved and root not in resolved.parents:
        raise CocoImportError(
            f"Zbiór musi leżeć w katalogu {root.name}/ — podana ścieżka wychodzi poza niego"
        )
    if resolved.suffix.lower() != ALLOWED_SUFFIX:
        raise CocoImportError(f"Zbiór musi być plikiem {ALLOWED_SUFFIX}")
    if not resolved.is_file():
        raise CocoImportError("Nie znaleziono zbioru pod podaną ścieżką")
    return resolved


def frames_prefix_for(dataset_path: Path) -> str:
    """
    Wylicza przedrostek URL klatek danego zbioru.

    Backend wystawia CAŁY katalog danych, a nie katalog jednego zbioru — inaczej
    dałoby się pracować tylko na jednym zbiorze naraz, a mamy ich kilka
    (`dataset_v2`, `dataset_v3`). Ścieżki w COCO są względne wobec katalogu
    klatek swojego zbioru, więc przedrostek dokłada brakującą część.

    Args:
        dataset_path: Ścieżka pliku po kuracji

    Returns:
        Przedrostek względem katalogu danych, np. `dataset_v2/frames`
    """
    dataset_dir = dataset_path.resolve().parent
    try:
        relative = dataset_dir.relative_to(import_root()).as_posix()
    except ValueError:
        relative = dataset_dir.name
    return f"{relative}/frames" if relative else "frames"


def _image_url(file_name: str, prefix: str) -> str:
    """
    Buduje URL podglądu klatki.

    Args:
        file_name: Ścieżka klatki z pola `file_name` obrazu COCO
        prefix: Przedrostek katalogu klatek względem katalogu danych

    Returns:
        URL wystawiany przez backend
    """
    normalized = file_name.replace(_WINDOWS_SEPARATOR, _POSIX_SEPARATOR)
    return f"{DATASET_URL_PREFIX}/{prefix}/{normalized}" if prefix else (
        f"{DATASET_URL_PREFIX}/{normalized}"
    )


def _annotation_to_frame(
    annotation: dict,
    image: dict,
    track_id: int,
    review_order: int,
    frames_prefix: str,
) -> FrameAnnotation:
    """
    Przekłada anotację COCO na anotację klatki w sesji.

    Werdykty człowieka (`au_verdicts`) zostają PUSTE. Pomiar reguł jedzie
    w `aus` jako podpowiedź, ale nie jako wypełniona odpowiedź — inaczej
    weryfikacja zamieniłaby się w zatwierdzanie tego, co reguły już orzekły.

    Args:
        annotation: Anotacja COCO
        image: Rekord obrazu COCO
        track_id: Identyfikator pary w sesji
        review_order: Pozycja pary w kolejce weryfikacji
        frames_prefix: Przedrostek katalogu klatek względem katalogu danych

    Returns:
        `FrameAnnotation` gotowa do zapisania w sesji
    """
    return FrameAnnotation(
        frame_idx=int(annotation["image_id"]),
        image_url=_image_url(image["file_name"], frames_prefix),
        track_id=track_id,
        annotation_status=ANNOTATION_STATUS_AUTO,
        source="ai",
        bbox=annotation.get("bbox"),
        keypoints=annotation.get("keypoints"),
        aus=dict(annotation.get("au_analysis", {})),
        emotion=annotation.get("emotion"),
        emotion_confidence=float(
            annotation.get("confidence", {}).get("emotion", 0.0)
            if isinstance(annotation.get("confidence"), dict)
            else 0.0
        ),
        emotion_rule_applied=annotation.get("emotion_rule_applied"),
        breed=annotation.get("breed"),
        breed_confidence=float(
            annotation.get("confidence", {}).get("breed", 0.0)
            if isinstance(annotation.get("confidence"), dict)
            else 0.0
        ),
        tfm_score=float(annotation.get("tfm_score", 0.0)),
        au_verdicts={},
        frame_role=annotation.get("frame_role"),
        quality=dict(annotation.get("quality", {})),
        review_order=review_order,
    )


def _group_pairs(coco: dict) -> list[tuple[dict, dict]]:
    """
    Grupuje anotacje w pary (neutralna, szczytowa) według `review_order`.

    Args:
        coco: Wczytany zbiór COCO po kuracji

    Returns:
        Lista par w kolejności weryfikacji

    Raises:
        CocoImportError: Gdy zbiór nie ma pola `review_order` (nie przeszedł kuracji)
    """
    by_order: dict[int, dict[str, dict]] = {}
    for annotation in coco.get("annotations", []):
        order = annotation.get("review_order")
        role = annotation.get("frame_role")
        if order is None:
            raise CocoImportError(
                "Zbiór nie ma pola `review_order` — przepuść go najpierw przez "
                "scripts/annotation/curate_for_review.py"
            )
        if role in (FRAME_ROLE_NEUTRAL, FRAME_ROLE_PEAK):
            by_order.setdefault(int(order), {})[role] = annotation

    pairs: list[tuple[dict, dict]] = []
    for order in sorted(by_order):
        entry = by_order[order]
        neutral, peak = entry.get(FRAME_ROLE_NEUTRAL), entry.get(FRAME_ROLE_PEAK)
        if neutral is not None and peak is not None:
            pairs.append((neutral, peak))
    if not pairs:
        raise CocoImportError("Zbiór nie zawiera ani jednej kompletnej pary neutral+peak")
    return pairs


def pair_key_for(image: dict) -> str:
    """
    Zwraca stabilny identyfikator pary — ścieżkę klatki szczytowej.

    `image_id` nadaje kuracja i zmienia się przy każdym jej powtórzeniu, więc
    etykiety przypięte do niego rozjechałyby się po pierwszym przeliczeniu
    zbioru. Ścieżka niesie nagranie, trek i numer klatki, więc przeżywa
    i ponowną kurację, i przeniesienie na inną maszynę.

    Args:
        image: Rekord obrazu COCO

    Returns:
        Znormalizowana ścieżka klatki
    """
    return str(image["file_name"]).replace(_WINDOWS_SEPARATOR, _POSIX_SEPARATOR)


def _apply_label(frame: FrameAnnotation, record) -> None:
    """
    Nakłada zapisaną decyzję człowieka na świeżo zbudowaną anotację.

    Args:
        frame: Anotacja klatki szczytowej
        record: Rekord z pliku etykiet
    """
    frame.au_verdicts = dict(record.au_verdicts)
    frame.usable = record.usable
    frame.keypoints_ok = record.keypoints_ok
    if record.breed is not None:
        frame.breed = record.breed
    if record.emotion is not None:
        frame.emotion = record.emotion
    frame.annotation_status = ANNOTATION_STATUS_VERIFIED
    frame.source = "manual"


def build_session(
    coco: dict,
    session_id: str,
    source_name: str,
    limit: Optional[int] = None,
    frames_prefix: str = "",
    dataset: Optional[str] = None,
) -> SessionData:
    """
    Składa sesję anotacji ze zbioru COCO po kuracji.

    Args:
        coco: Wczytany zbiór COCO
        session_id: Identyfikator tworzonej sesji
        source_name: Nazwa źródła zapisywana w sesji (nazwa pliku zbioru)
        limit: Najwyżej tyle par; None znaczy wszystkie
        frames_prefix: Przedrostek katalogu klatek względem katalogu danych
        dataset: Nazwa zbioru — po niej odnajdujemy zapisane etykiety zespołu.
            None znaczy „nie wczytuj etykiet" (używane w testach jednostkowych).

    Returns:
        `SessionData` z parami w kolejności weryfikacji, z nałożonymi etykietami

    Raises:
        CocoImportError: Gdy zbiór nie przeszedł kuracji albo nie ma pełnych par
    """
    images = {image["id"]: image for image in coco.get("images", [])}
    pairs = _group_pairs(coco)
    if limit is not None:
        pairs = pairs[:limit]

    # Etykiety zespołu leżą w repozytorium, sesja jest lokalna — dlatego to
    # pliki etykiet, a nie sesja, są źródłem prawdy o tym, co już zrobione.
    labels = latest_by_pair(dataset) if dataset else {}

    frames: list[FrameAnnotation] = []
    dogs: list[DogTrack] = []

    for track_id, (neutral, peak) in enumerate(pairs):
        neutral_frame = _annotation_to_frame(
            neutral, images[neutral["image_id"]], track_id, track_id, frames_prefix
        )
        peak_frame = _annotation_to_frame(
            peak, images[peak["image_id"]], track_id, track_id, frames_prefix
        )
        record = labels.get(pair_key_for(images[peak["image_id"]]))
        if record is not None:
            _apply_label(peak_frame, record)

        frames.extend((neutral_frame, peak_frame))
        dogs.append(
            DogTrack(
                track_id=track_id,
                neutral_frame_idx=neutral_frame.frame_idx,
                neutral_keypoints=neutral_frame.keypoints,
                neutral_source=peak.get("neutral_source", "auto"),
                peak_frame_indices=[peak_frame.frame_idx],
                au_noise=dict(peak.get("au_noise") or {}),
                au_sample_count=dict(peak.get("au_sample_count") or {}),
            )
        )

    return SessionData(
        session_id=session_id,
        video_filename=source_name,
        created_at=datetime.now().isoformat(),
        total_frames=len(frames),
        neutral_frame_idx=dogs[0].neutral_frame_idx,
        neutral_keypoints=dogs[0].neutral_keypoints,
        frames=frames,
        dogs=dogs,
    )


# Po tym wzorcu poznajemy zbiory gotowe do weryfikacji. Anotator nie ma
# wpisywać ścieżek — narzędzie ma samo znaleźć, co jest do zrobienia.
CURATED_PATTERN: str = "curated*.json"

# Jak głęboko schodzimy szukając zbiorów. Dwa poziomy pokrywają układ
# `data/<zbior>/curated.json`, a nie każą przeczesywać całego katalogu danych.
_SEARCH_DEPTH: int = 2


def find_datasets(root: Optional[Path] = None) -> list[Path]:
    """
    Znajduje zbiory po kuracji, które da się wziąć do weryfikacji.

    Args:
        root: Katalog danych; domyślnie `import_root()`

    Returns:
        Ścieżki zbiorów, posortowane — najświeższe pierwsze
    """
    base = root or import_root()
    if not base.is_dir():
        return []
    found: list[Path] = []
    for depth in range(_SEARCH_DEPTH + 1):
        pattern = "/".join(["*"] * depth + [CURATED_PATTERN]) if depth else CURATED_PATTERN
        found.extend(base.glob(pattern))
    unique = {path.resolve(): path for path in found if path.is_file()}
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def count_pairs(path: Path) -> int:
    """
    Liczy pary w zbiorze bez budowania sesji.

    Args:
        path: Ścieżka zbioru po kuracji

    Returns:
        Liczba par gotowych do weryfikacji; 0 gdy pliku nie da się odczytać
    """
    try:
        coco = load_coco(path)
    except CocoImportError:
        return 0
    orders = {
        annotation.get("review_order")
        for annotation in coco.get("annotations", [])
        if annotation.get("review_order") is not None
    }
    return len(orders)


def load_coco(path: Path) -> dict:
    """
    Wczytuje zbiór COCO z dysku.

    Args:
        path: Ścieżka do pliku JSON

    Returns:
        Słownik COCO

    Raises:
        CocoImportError: Gdy pliku nie ma albo nie jest poprawnym JSON-em.
            Komunikat nie niesie ścieżki ani treści pliku — poszedłby wprost
            do przeglądarki i wyciekłby układ dysku serwera.
    """
    if not path.exists():
        raise CocoImportError("Nie znaleziono zbioru pod podaną ścieżką")
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise CocoImportError("Zbiór nie jest poprawnym plikiem JSON") from error
