"""
Kadrowanie klatek i przenoszenie współrzędnych do układu wycinka.

Wspólne dla dwóch odbiorców, które kadrują INACZEJ i celowo:

* **paczka robocza** (`build_work_pack`) tnie wokół CAŁEGO psa — anotator musi
  widzieć, o którego psa chodzi i co on robi, a pod klawiszem `F` ogląda ten
  właśnie kadr jako „szerszy plan";
* **zbiór do oddania** (`build_final_dataset`) tnie wokół MORDY — publikujemy
  46 punktów twarzy i 21 AU, a piksele tułowia nie niosą żadnej etykiety.

Wspólne zostaje to, co łatwo pomylić: obrazem jest wycinek, więc punkty muszą
przejechać do jego układu. Bez tego zbiór opisywałby punkty w pikselach
oryginału, a zdjęcia byłyby wycinkami — czyli wskazywałby poza własne obrazy.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from packages.data.schemas import NUM_KEYPOINTS

# Przedrostek długiej ścieżki Windows.
#
# Windows tnie ścieżkę na 260 znakach (MAX_PATH) i zgłasza to jako
# FileNotFoundError — czyli limit wygląda jak brakujący katalog, a nie jak
# limit. Gorzej: `Path.is_file()` na takiej ścieżce zwraca po prostu False,
# więc plik LEŻĄCY na dysku wygląda na nieistniejący i znika po cichu.
#
# Przedrostek przełącza wywołania na wariant długi i działa niezależnie od
# ustawienia LongPathsEnabled w rejestrze — nie wymaga więc niczego od maszyny
# kolegi z zespołu. Wymaga ścieżki BEZWZGLĘDNEJ, z ukośnikami wstecznymi
# i bez składników względnych, stąd `resolve()` przed sklejeniem.
_WINDOWS_LONG_PATH_PREFIX: str = "\\\\?\\"


def _os_path(path: Path) -> str:
    """
    Zamienia ścieżkę na postać, którą system otworzy niezależnie od jej długości.

    Args:
        path: Ścieżka do pliku lub katalogu

    Returns:
        Ścieżka w postaci tekstowej; na Windowsie z przedrostkiem długiej ścieżki
    """
    if os.name != "nt":
        return str(path)
    resolved = str(path.resolve())
    if resolved.startswith(_WINDOWS_LONG_PATH_PREFIX):
        return resolved
    return f"{_WINDOWS_LONG_PATH_PREFIX}{resolved}"

# Punkt poniżej tej pewności nie wyznacza kadru — inaczej jedna zabłąkana
# predykcja na tle rozdmuchuje wycinek na pół obrazu.
KEYPOINT_VISIBLE_MIN: float = 0.3

# Minimalna liczba pewnych punktów, żeby w ogóle liczyć kadr mordy
MIN_VISIBLE_KEYPOINTS: int = 6


# Jakość zapisu JPEG używana, gdy wywołujący nie poda własnej
DEFAULT_JPEG_QUALITY: int = 88


def read_image(path: Path) -> Optional[np.ndarray]:
    """
    Wczytuje obraz ścieżką, która MOŻE zawierać znaki spoza ASCII.

    `cv2.imread` na Windowsie idzie przez API ANSI i na ścieżce z cyrylicą
    zwraca None — czyli klatka wygląda jak brakująca, choć leży na dysku.
    Zmierzone na tym zbiorze: nagrania `собака_*` traciły w ten sposób
    wszystkie swoje klatki, a paczka meldowała je tylko jako „pominięte".

    Obejście: plik czyta Python (który ze ścieżkami Unicode radzi sobie
    poprawnie), a OpenCV dostaje gotowy bufor bajtów.

    Args:
        path: Ścieżka do pliku obrazu

    Returns:
        Obraz BGR albo None, gdy pliku nie ma lub nie da się go zdekodować
    """
    native = _os_path(path)
    if not os.path.isfile(native):
        return None
    buffer = np.fromfile(native, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def write_jpeg(path: Path, image: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> bool:
    """
    Zapisuje JPEG ścieżką, która MOŻE zawierać znaki spoza ASCII.

    Odpowiednik `read_image` po stronie zapisu i z tego samego powodu —
    `cv2.imwrite` ma na Windowsie to samo ograniczenie co `cv2.imread`.

    Args:
        path: Ścieżka docelowa
        image: Obraz BGR
        quality: Jakość JPEG

    Returns:
        Czy zapis się powiódł
    """
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    os.makedirs(_os_path(path.parent), exist_ok=True)
    native = _os_path(path)
    encoded.tofile(native)
    return os.path.isfile(native) and os.path.getsize(native) > 0


def file_size(path: Path) -> int:
    """
    Podaje rozmiar pliku, także gdy ścieżka przekracza limit Windows.

    Istnieje, żeby wiedza o długich ścieżkach została w JEDNYM module —
    `Path.stat()` wołane wprost na kadrze wywraca się tam, gdzie zapis
    przez `write_jpeg` przechodzi bez problemu.

    Args:
        path: Ścieżka do pliku

    Returns:
        Rozmiar w bajtach
    """
    return os.path.getsize(_os_path(path))


@dataclass(frozen=True)
class CropBox:
    """Prostokąt kadru w pikselach pełnej klatki."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        """Szerokość kadru."""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """Wysokość kadru."""
        return self.y1 - self.y0


def _clamped(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> Optional[CropBox]:
    """
    Przycina prostokąt do granic klatki i odrzuca zdegenerowany.

    Args:
        x0: Lewa krawędź
        y0: Górna krawędź
        x1: Prawa krawędź
        y1: Dolna krawędź
        width: Szerokość klatki
        height: Wysokość klatki

    Returns:
        Kadr albo None, gdy po przycięciu nic z niego nie zostaje
    """
    box = CropBox(
        x0=int(max(0, x0)),
        y0=int(max(0, y0)),
        x1=int(min(width, x1)),
        y1=int(min(height, y1)),
    )
    return box if box.width > 1 and box.height > 1 else None


def face_box(keypoints: list[float], width: int, height: int, margin: float) -> Optional[CropBox]:
    """
    Wyznacza kadr mordy z rozpiętości pewnych punktów.

    Args:
        keypoints: Płaska lista 138 wartości (x, y, widoczność)
        width: Szerokość pełnej klatki
        height: Wysokość pełnej klatki
        margin: Zapas wokół rozpiętości punktów, ułamek jej rozmiaru

    Returns:
        Kadr albo None, gdy pewnych punktów jest za mało
    """
    points = np.asarray(keypoints, dtype=float).reshape(NUM_KEYPOINTS, 3)
    visible = points[points[:, 2] > KEYPOINT_VISIBLE_MIN]
    if len(visible) < MIN_VISIBLE_KEYPOINTS:
        return None

    x0, y0 = visible[:, 0].min(), visible[:, 1].min()
    x1, y1 = visible[:, 0].max(), visible[:, 1].max()
    pad_x = max((x1 - x0) * margin, 1.0)
    pad_y = max((y1 - y0) * margin, 1.0)
    return _clamped(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y, width, height)


def body_box(bbox: list[float], width: int, height: int, margin: float) -> Optional[CropBox]:
    """
    Wyznacza kadr wokół boksu całego psa.

    Args:
        bbox: Boks psa [x, y, szerokość, wysokość] w pikselach pełnej klatki
        width: Szerokość pełnej klatki
        height: Wysokość pełnej klatki
        margin: Zapas wokół boksu, ułamek jego rozmiaru

    Returns:
        Kadr albo None, gdy boks jest zdegenerowany
    """
    x, y, box_width, box_height = (float(value) for value in bbox)
    # Boks o zerowym boku to błąd detekcji, nie mały pies. Zapas minimalny
    # zrobiłby z niego kadr 2x2 px, czyli zamienił awarię w cichy śmieć.
    if box_width <= 0 or box_height <= 0:
        return None

    pad_x = max(box_width * margin, 1.0)
    pad_y = max(box_height * margin, 1.0)
    return _clamped(
        x - pad_x, y - pad_y, x + box_width + pad_x, y + box_height + pad_y, width, height
    )


def crop_and_scale(image: np.ndarray, box: CropBox, max_side: int) -> tuple[np.ndarray, float]:
    """
    Wycina kadr i zmniejsza go, jeśli dłuższy bok przekracza próg.

    Nigdy nie powiększa: skalowanie w górę dodałoby tylko szum interpolacji,
    a kadr i tak nie zyskałby szczegółu, którego w oryginale nie było.

    Args:
        image: Pełna klatka
        box: Kadr do wycięcia
        max_side: Dopuszczalny dłuższy bok wyniku

    Returns:
        Para (wycinek, współczynnik skali zastosowany po wycięciu)
    """
    crop = image[box.y0 : box.y1, box.x0 : box.x1]
    longest = max(crop.shape[0], crop.shape[1])
    if longest <= max_side:
        return crop, 1.0

    scale = max_side / longest
    resized = cv2.resize(
        crop,
        (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def remap_keypoints(keypoints: list[float], box: CropBox, scale: float) -> list[float]:
    """
    Przenosi punkty z układu pełnej klatki do układu wycinka.

    Widoczność zostaje nietknięta — przemnożona przestałaby być
    prawdopodobieństwem.

    Args:
        keypoints: Płaska lista 138 wartości
        box: Kadr, względem którego liczymy
        scale: Skala nałożona po wycięciu

    Returns:
        Nowa płaska lista 138 wartości
    """
    points = np.asarray(keypoints, dtype=float).reshape(NUM_KEYPOINTS, 3)
    moved = points.copy()
    moved[:, 0] = (points[:, 0] - box.x0) * scale
    moved[:, 1] = (points[:, 1] - box.y0) * scale
    return [float(value) for value in moved.reshape(-1)]


def remap_bbox(bbox: list[float], box: CropBox, scale: float) -> list[float]:
    """
    Przenosi boks psa do układu wycinka.

    Args:
        bbox: Boks [x, y, szerokość, wysokość] w pikselach pełnej klatki
        box: Kadr, względem którego liczymy
        scale: Skala nałożona po wycięciu

    Returns:
        Boks w układzie wycinka
    """
    x, y, width, height = (float(value) for value in bbox)
    return [(x - box.x0) * scale, (y - box.y0) * scale, width * scale, height * scale]


def bbox_from_keypoints(keypoints: list[float]) -> list[float]:
    """
    Liczy obrys pewnych punktów — boks opisujący mordę w układzie wycinka.

    Args:
        keypoints: Płaska lista 138 wartości w układzie wycinka

    Returns:
        Boks [x, y, szerokość, wysokość]; zerowy, gdy nie ma pewnych punktów
    """
    points = np.asarray(keypoints, dtype=float).reshape(NUM_KEYPOINTS, 3)
    visible = points[points[:, 2] > KEYPOINT_VISIBLE_MIN]
    if len(visible) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0 = float(visible[:, 0].min()), float(visible[:, 1].min())
    x1, y1 = float(visible[:, 0].max()), float(visible[:, 1].max())
    return [x0, y0, x1 - x0, y1 - y0]
