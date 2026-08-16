"""
Wybór klatki neutralnej musi zgadzać się z bramką, która ją potem ocenia.

Detektor rankuje kandydatów po `head_pose` (yaw/roll), a bramka jakości sądzi je
po `face_asymmetry` liczonej z keypoints. To dwie różne miary i nic nie wymusza
ich zgodności — a rozjazd kosztowałby cały trek, bo neutralna odrzucona przez
bramkę zabiera ze sobą wszystkie pary tego psa.

Testy blokują tę zgodność. Podejrzenie, że detektor faktycznie wybiera źle,
sprawdzono osobno i NIE potwierdziło się: pozorna przewaga peaków nad wybraną
neutralną okazała się artefaktem brania minimum z kilku zaszumionych pomiarów
(przy jednym peaku wynosi +0.008 i 51% przypadków, przy sześciu rośnie do
+0.368 i 100%). Prawdziwą przyczyną odrzuceń jest szum samej miary asymetrii —
patrz `test_shape_distance.py`.
"""

import numpy as np

from packages.data.schemas import NUM_KEYPOINTS
from packages.pipeline.neutral_frame import NeutralFrameDetector
from packages.pipeline.quality_gate import _SYMMETRIC_PAIRS, face_asymmetry

_AXIS_X: float = 100.0


def _frontal_face() -> np.ndarray:
    """
    Buduje idealnie symetryczne keypoints psa patrzącego na wprost.

    Punkty osiowe leżą dokładnie na prostej `x = _AXIS_X`, a każda para
    symetryczna dostaje równe odsunięcie w obie strony — inaczej „frontalny"
    wzorzec sam miałby niezerową asymetrię i test niczego by nie rozstrzygał.

    Returns:
        Płaska tablica 138 wartości (x, y, conf) z pewnością 1.0
    """
    points = np.zeros((NUM_KEYPOINTS, 3), dtype=float)
    points[:, 0] = _AXIS_X
    points[:, 1] = np.linspace(40.0, 160.0, NUM_KEYPOINTS)
    points[:, 2] = 1.0
    for index, (left, right) in enumerate(_SYMMETRIC_PAIRS):
        offset = 20.0 + 5.0 * index
        height = 50.0 + 12.0 * index
        points[left] = (_AXIS_X - offset, height, 1.0)
        points[right] = (_AXIS_X + offset, height, 1.0)
    return points.reshape(-1)


def _skewed(points: np.ndarray, shift: float) -> np.ndarray:
    """
    Przesuwa prawą połowę mordy, psując symetrię bez ruszania osi.

    Args:
        points: Płaskie keypoints (138,)
        shift: O ile przesunąć prawe punkty par symetrycznych

    Returns:
        Nowa płaska tablica keypoints
    """
    skewed = points.reshape(NUM_KEYPOINTS, 3).copy()
    for _, right in _SYMMETRIC_PAIRS:
        skewed[right, 0] += shift
    return skewed.reshape(-1)


def _asymmetry(points: np.ndarray) -> float:
    """
    Liczy asymetrię płaskich keypoints.

    Args:
        points: Płaskie keypoints (138,)

    Returns:
        Wartość miary bramki
    """
    return face_asymmetry(points.reshape(NUM_KEYPOINTS, 3)[:, :2])


def test_wzorzec_frontalny_jest_naprawde_symetryczny() -> None:
    """Bez tego reszta testów mierzyłaby szum wzorca, a nie zachowanie kodu."""
    assert _asymmetry(_frontal_face()) < 1e-6


def test_detektor_nie_wybiera_klatki_gorszej_dla_bramki() -> None:
    """
    Przy równej stabilności wygrywa klatka symetryczniejsza.

    Krzywe kadry leżą pierwsze, żeby wybór „pierwszego z najlepszym wynikiem"
    nie przeszedł testu przypadkiem.
    """
    frontal = _frontal_face()
    crooked = _skewed(frontal, shift=60.0)
    assert _asymmetry(crooked) > _asymmetry(frontal)

    detector = NeutralFrameDetector()
    frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(4)]
    chosen = detector.detect_auto(
        frames=frames,
        keypoints_list=[crooked, crooked.copy(), frontal, frontal.copy()],
    )

    assert chosen in (2, 3), (
        "detektor wybral klatke bardziej asymetryczna, mimo ze obok byla rowniejsza"
    )


def test_wybor_nie_zalezy_od_polozenia_dobrej_klatki() -> None:
    """Wynik nie może zależeć od tego, w którym miejscu treku leży dobra klatka."""
    frontal = _frontal_face()
    crooked = _skewed(frontal, shift=60.0)
    detector = NeutralFrameDetector()
    frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(4)]

    first = detector.detect_auto(
        frames=frames,
        keypoints_list=[frontal, crooked, crooked.copy(), crooked.copy()],
    )
    last = detector.detect_auto(
        frames=frames,
        keypoints_list=[crooked, crooked.copy(), crooked.copy(), frontal],
    )

    assert first == 0
    assert last == 3
