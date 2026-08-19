"""
Współdzielone fixtury keypoints dla testów pipeline'u.

Realistyczne układy 46 punktów DogFLW używane przez testy klatki neutralnej
i procesora temporalnego — trzymane w jednym miejscu, żeby próg frontalności
testować na tych samych danych.
"""

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS


def make_frontal_kp() -> np.ndarray:
    """
    Tworzy realistyczną tablicę 46 keypoints — frontalna twarz psa.

    Centrum (150, 150), odległość między centrami oczu ~100px.
    Symetria lewo/prawa → minimalna asymetria yaw i minimalny roll.

    Returns:
        Tablica (138,) z wartościami [x0, y0, v0, ...]
    """
    kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    kp[:, 2] = 0.95

    # Oczy
    kp[KP.LEFT_EYE_INNER]  = [105, 150, 0.95]
    kp[KP.LEFT_EYE_TOP]    = [100, 144, 0.95]
    kp[KP.LEFT_EYE_OUTER]  = [95, 150, 0.95]
    kp[KP.LEFT_EYE_BOTTOM] = [100, 156, 0.95]
    kp[KP.RIGHT_EYE_INNER] = [195, 150, 0.95]
    kp[KP.RIGHT_EYE_TOP]   = [200, 144, 0.95]
    kp[KP.RIGHT_EYE_OUTER] = [205, 150, 0.95]
    kp[KP.RIGHT_EYE_BOTTOM]= [200, 156, 0.95]

    # Brwi
    kp[KP.LEFT_BROW_INNER]  = [108, 137, 0.9]
    kp[KP.LEFT_BROW_CENTER] = [100, 134, 0.9]
    kp[KP.LEFT_BROW_OUTER]  = [92, 137, 0.9]
    kp[KP.RIGHT_BROW_INNER] = [192, 137, 0.9]
    kp[KP.RIGHT_BROW_CENTER]= [200, 134, 0.9]
    kp[KP.RIGHT_BROW_OUTER] = [208, 137, 0.9]

    # Uszy (symetryczne)
    kp[KP.LEFT_EAR_BASE_FRONT]  = [80, 120, 0.9]
    kp[KP.LEFT_EAR_BASE_BACK]   = [75, 115, 0.85]
    kp[KP.LEFT_EAR_MID]         = [72, 95, 0.85]
    kp[KP.LEFT_EAR_TIP]         = [68, 70, 0.8]
    kp[KP.RIGHT_EAR_BASE_FRONT] = [220, 120, 0.9]
    kp[KP.RIGHT_EAR_BASE_BACK]  = [225, 115, 0.85]
    kp[KP.RIGHT_EAR_MID]        = [228, 95, 0.85]
    kp[KP.RIGHT_EAR_TIP]        = [232, 70, 0.8]

    # Nos — pośrodku
    kp[KP.NOSE_TIP]       = [150, 200, 0.95]
    kp[KP.NOSE_LEFT_WING] = [143, 196, 0.9]
    kp[KP.NOSE_RIGHT_WING]= [157, 196, 0.9]
    kp[KP.NOSE_BRIDGE]    = [150, 175, 0.9]

    # Usta
    kp[KP.MOUTH_LEFT_CORNER]  = [120, 218, 0.9]
    kp[KP.UPPER_LIP_LEFT]     = [132, 214, 0.9]
    kp[KP.UPPER_LIP_CENTER]   = [150, 212, 0.9]
    kp[KP.UPPER_LIP_RIGHT]    = [168, 214, 0.9]
    kp[KP.MOUTH_RIGHT_CORNER] = [180, 218, 0.9]
    kp[KP.LOWER_LIP_RIGHT]    = [168, 226, 0.9]
    kp[KP.LOWER_LIP_CENTER]   = [150, 228, 0.9]
    kp[KP.LOWER_LIP_LEFT]     = [132, 226, 0.9]

    # Pysk i kontur
    kp[KP.MUZZLE_TOP]        = [150, 207, 0.9]
    kp[KP.MUZZLE_LEFT]       = [122, 222, 0.85]
    kp[KP.MUZZLE_RIGHT]      = [178, 222, 0.85]
    kp[KP.CHIN]              = [150, 250, 0.85]
    kp[KP.FOREHEAD_CENTER]   = [150, 115, 0.8]
    kp[KP.FOREHEAD_LEFT]     = [120, 118, 0.8]
    kp[KP.FOREHEAD_RIGHT]    = [180, 118, 0.8]
    kp[KP.LEFT_CHEEK_UPPER]  = [88, 162, 0.8]
    kp[KP.LEFT_CHEEK_LOWER]  = [88, 202, 0.8]
    kp[KP.RIGHT_CHEEK_UPPER] = [212, 162, 0.8]
    kp[KP.RIGHT_CHEEK_LOWER] = [212, 202, 0.8]
    kp[KP.JAW_CENTER]        = [150, 245, 0.85]

    # Punkty bez stałej w `KP` — dawniej zostawały w (0, 0), czyli dwanaście
    # z czterdziestu sześciu punktów „frontalnej mordy" leżało w rogu układu.
    # Miara `shape_distance` to wyłapała (0.164 przy p99 realnych psów 0.133),
    # więc fixtura nie była realistyczna mimo nazwy.
    #
    # Punkty ucha leżą NA narysowanej krawędzi (baza → środek → czubek), a nie
    # tam, gdzie stawia je średni kształt DogFLW: uśredniony pies ma uszy wiszące
    # i wypchnąłby te punkty szerzej niż czubki uszu tej fixtury, rozdymając boks
    # mordy. Reszta wyliczona z dopasowania średniego kształtu.
    kp[2]  = [70, 82, 0.85]     # ear_left_upper_bend
    kp[3]  = [230, 82, 0.85]    # ear_right_upper_bend
    kp[8]  = [74, 108, 0.85]    # ear_left_lower_mid
    kp[9]  = [226, 108, 0.85]   # ear_right_lower_mid
    kp[10] = [73, 101, 0.85]    # ear_left_lower_23
    kp[11] = [227, 101, 0.85]   # ear_right_lower_23
    kp[26] = [135, 191, 0.85]   # nose_left_upper_edge
    kp[27] = [164, 191, 0.85]   # nose_right_upper_edge
    kp[32] = [150, 205, 0.85]   # nostril_mid
    kp[36] = [120, 207, 0.85]   # whisker_pad_left
    kp[37] = [179, 207, 0.85]   # whisker_pad_right
    kp[KP.TONGUE_TIP] = [150, 239, 0.85]

    return kp.flatten()


def make_turned_kp(shift_x: float) -> np.ndarray:
    """
    Tworzy twarz obróconą w bok przez przesunięcie nosa względem osi symetrii.

    Args:
        shift_x: Przesunięcie czubka nosa w pikselach. Ujemne = nos bliżej
            lewego oka (yaw_asymmetry < 0), dodatnie = bliżej prawego (> 0).

    Returns:
        Tablica (138,) z wartościami [x0, y0, v0, ...]
    """
    flat = make_frontal_kp()
    kp = flat.reshape(NUM_KEYPOINTS, 3)
    kp[KP.NOSE_TIP, 0] += shift_x
    return kp.flatten()


def make_tilted_kp(shift_y: float) -> np.ndarray:
    """
    Tworzy twarz przechyloną — prawe oko przesunięte w pionie o `shift_y`.

    Args:
        shift_y: Przesunięcie prawego oka w pikselach. Ujemne = prawe oko wyżej
            (roll < 0), dodatnie = prawe oko niżej (roll > 0).

    Returns:
        Tablica (138,) z wartościami [x0, y0, v0, ...]
    """
    kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
    kp[KP.RIGHT_EYE_INNER, 1] += shift_y
    kp[KP.RIGHT_EYE_OUTER, 1] += shift_y
    return kp.flatten()


def make_low_visibility_kp(visibility: float = 0.1) -> np.ndarray:
    """
    Tworzy frontalną twarz z jednakowo niską widocznością wszystkich punktów.

    Args:
        visibility: Widoczność wpisana we wszystkie 46 punktów

    Returns:
        Tablica (138,) z wartościami [x0, y0, v0, ...]
    """
    kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
    kp[:, 2] = visibility
    return kp.flatten()
