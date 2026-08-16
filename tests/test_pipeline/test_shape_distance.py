"""
Testy miary `shape_distance` — wykrywacza punktów postawionych byle gdzie.

Powstała, bo `face_asymmetry` przy naszej dokładności keypoints przestaje
rozstrzygać: pies patrzący WPROST czyta się jako 0.31 przy progu 0.50, a kadry
z punktami na sierści i potylicy mieściły się w 0.16–0.46, czyli przechodziły.
"""

import numpy as np
import pytest

from packages.data.schemas import NUM_KEYPOINTS
from packages.pipeline.quality_gate import (
    DEFAULT_MAX_SHAPE_DISTANCE,
    REASON_IMPLAUSIBLE_SHAPE,
    QualityThresholds,
    assess_frame,
    face_asymmetry,
    load_mean_shape,
    shape_distance,
)


def _flat(coords: np.ndarray, confidence: float = 1.0) -> np.ndarray:
    """
    Skleja współrzędne z pewnością w płaską tablicę 138 wartości.

    Args:
        coords: Współrzędne (46, 2)
        confidence: Pewność wpisana każdemu punktowi

    Returns:
        Płaska tablica 138 wartości
    """
    return np.column_stack([coords, np.full(len(coords), confidence)]).ravel()


def test_wzorzec_ma_zerowa_odleglosc_od_siebie() -> None:
    """Kształt identyczny ze wzorcem musi dać zero, inaczej skala miary kłamie."""
    assert shape_distance(_flat(load_mean_shape())) == pytest.approx(0.0, abs=1e-6)


def test_miara_nie_zalezy_od_skali_obrotu_i_polozenia() -> None:
    """
    Prokrustes ma zdejmować podobieństwo — bez tego miara mierzyłaby kadrowanie.

    Ta sama morda sfilmowana z bliska i z daleka musi dostać ten sam wynik.
    """
    shape = load_mean_shape()
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    moved = (shape @ rotation) * 4.5 + np.array([120.0, -60.0])

    assert shape_distance(_flat(moved)) == pytest.approx(0.0, abs=1e-6)


def test_punkty_rozrzucone_daja_duza_odleglosc() -> None:
    """Losowa chmura punktów to awaria detektora i miara musi ją pokazać."""
    rng = np.random.default_rng(3)
    scattered = rng.uniform(0.0, 200.0, (NUM_KEYPOINTS, 2))

    assert shape_distance(_flat(scattered)) > DEFAULT_MAX_SHAPE_DISTANCE


def test_asymetria_przepuszcza_smiec_ktory_ksztalt_zatrzymuje() -> None:
    """
    Sedno zmiany: rozrzucone punkty potrafią mieć NISKĄ asymetrię.

    Chmura symetryczna względem pionowej osi czyta się dla `face_asymmetry` jak
    porządny front, choć z psią mordą nie ma nic wspólnego. Dlatego sama
    asymetria nie wystarcza jako bramka.
    """
    rng = np.random.default_rng(11)
    half = rng.uniform(10.0, 90.0, (NUM_KEYPOINTS // 2, 2))
    mirrored = np.column_stack([200.0 - half[:, 0], half[:, 1]])
    cloud = np.empty((NUM_KEYPOINTS, 2), dtype=float)
    cloud[0::2] = half
    cloud[1::2] = mirrored

    thresholds = QualityThresholds(max_asymmetry=0.60, min_face_width=0.0)
    quality = assess_frame(_flat(cloud), thresholds)

    assert face_asymmetry(cloud) < thresholds.max_asymmetry
    assert REASON_IMPLAUSIBLE_SHAPE in quality.reasons
    assert not quality.is_usable


def test_prog_ksztaltu_da_sie_wylaczyc() -> None:
    """Analizy porównawcze muszą móc policzyć miarę bez wetowania kadrów."""
    rng = np.random.default_rng(5)
    scattered = _flat(rng.uniform(0.0, 200.0, (NUM_KEYPOINTS, 2)))
    permissive = QualityThresholds(
        max_asymmetry=9.0, min_face_width=0.0, max_shape_distance=9.0
    )

    quality = assess_frame(scattered, permissive)

    assert REASON_IMPLAUSIBLE_SHAPE not in quality.reasons
    assert quality.shape_distance > 0.0
