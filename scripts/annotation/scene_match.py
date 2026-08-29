"""
Sprawdzenie, czy obie klatki pary pokazują TEGO SAMEGO psa.

Trek psa urywa się na cięciu montażowym, ale numer treku nie — w nagraniu
złożonym z wielu ujęć (kompilacje `youtube_*`) ten sam `track_id` biegnie przez
kilka różnych zwierząt. Klatka neutralna i szczytowa trafiają wtedy do jednej
pary, choć pokazują inne psy.

Dla AU to nie jest usterka kosmetyczna. AU są Z DEFINICJI różnicą względem
klatki neutralnej: gdy neutralna należy do innego psa, wszystkie 21 pomiarów
opisuje różnicę między dwoma zwierzętami, a nie mimikę jednego. Taka para nie
tylko nie wnosi nic — wnosi etykietę fałszywą.

Zmierzone 29.08.2026 na 400 parach: 6% ma podobieństwo mordy poniżej 0.5,
a wśród nagrań `youtube_*` (kompilacje) — 15%.

Porównujemy WYCINEK MORDY, nie całą klatkę. Cała klatka reaguje na ruch psa
w kadrze i na zmianę tła, więc odrzucała pary poprawne: przy pomiarze na całej
klatce próg musiałby wynosić 0.7, przy pomiarze na mordzie wystarcza 0.4.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from scripts.annotation.cropping import face_box, read_image

# Zapas wokół rozpiętości punktów przy wycinaniu mordy do porównania
FACE_MARGIN: float = 0.15

# Rozmiar, do którego sprowadzamy mordę przed liczeniem histogramu
COMPARE_SIZE: int = 64

# Kubełki histogramu: odcień i nasycenie. Jasność pomijamy — to samo zwierzę
# w innym oświetleniu ma inną jasność, a barwa sierści zostaje.
HUE_BINS: int = 24
SATURATION_BINS: int = 24


def _face_histogram(frame: Path, keypoints: list[float]) -> Optional[np.ndarray]:
    """
    Liczy histogram barw wycinka mordy.

    Args:
        frame: Ścieżka pełnej klatki
        keypoints: Punkty twarzy w układzie tej klatki

    Returns:
        Znormalizowany histogram albo None, gdy mordy nie da się wyciąć
    """
    image = read_image(frame)
    if image is None:
        return None
    height, width = image.shape[:2]
    box = face_box(keypoints, width, height, FACE_MARGIN)
    if box is None:
        return None
    crop = image[box.y0 : box.y1, box.x0 : box.x1]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (COMPARE_SIZE, COMPARE_SIZE))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [HUE_BINS, SATURATION_BINS], [0, 180, 0, 256]
    )
    return cv2.normalize(histogram, histogram).flatten()


def face_similarity(
    frames_dir: Path,
    neutral_frame: str,
    neutral_keypoints: list[float],
    peak_frame: str,
    peak_keypoints: list[float],
) -> Optional[float]:
    """
    Mierzy, jak podobne są mordy na obu klatkach pary.

    Args:
        frames_dir: Katalog pełnych klatek
        neutral_frame: Ścieżka klatki neutralnej względem `frames_dir`
        neutral_keypoints: Punkty klatki neutralnej
        peak_frame: Ścieżka klatki szczytowej względem `frames_dir`
        peak_keypoints: Punkty klatki szczytowej

    Returns:
        Korelacja histogramów w zakresie [-1, 1]; None, gdy pomiar niemożliwy.
        None znaczy „nie wiem", a NIE „różne psy" — para bez pomiaru przechodzi,
        bo odrzucanie na podstawie nieudanego odczytu pliku gubiłoby dobre pary.
    """
    first = _face_histogram(frames_dir / neutral_frame, neutral_keypoints)
    second = _face_histogram(frames_dir / peak_frame, peak_keypoints)
    if first is None or second is None:
        return None
    return float(cv2.compareHist(first, second, cv2.HISTCMP_CORREL))
