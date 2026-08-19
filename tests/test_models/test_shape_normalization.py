"""Testy normalizacji kształtu metodą Prokrustesa."""

import numpy as np
import pytest

from packages.data.schemas import NUM_KEYPOINTS
from packages.models.shape_normalization import (
    DEFAULT_MAX_GPA_ITERATIONS,
    mean_shape,
    procrustes_align,
)


def _shape_to_flat(coords: np.ndarray, visibility: float = 1.0) -> np.ndarray:
    """Zamienia tablicę (46, 2) na płaską [x, y, v, ...] o zadanej widoczności."""
    flat = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
    flat[0::3] = coords[:, 0]
    flat[1::3] = coords[:, 1]
    flat[2::3] = visibility
    return flat


def _random_shape(seed: int) -> np.ndarray:
    """Losowy, ale powtarzalny kształt (46, 2)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 50, (NUM_KEYPOINTS, 2)) + 200.0


def _rotation_matrix(degrees: float) -> np.ndarray:
    """Macierz obrotu 2x2 dla kąta w stopniach."""
    angle = np.radians(degrees)
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def _centroid_size(flat: np.ndarray) -> float:
    """Rozmiar centroidu (norma Frobeniusa) kształtu zapisanego płasko."""
    coords = np.stack([flat[0::3], flat[1::3]], axis=1)
    centered = coords - coords.mean(axis=0)
    return float(np.sqrt(np.sum(centered**2)))


class TestProcrustesAlign:
    """Niezmienniczość na przesunięcie, skalę i obrót."""

    def test_przesuniecie_nie_zmienia_wyniku(self):
        coords = _random_shape(1)
        reference = _random_shape(2)

        first = procrustes_align(_shape_to_flat(coords), reference)
        second = procrustes_align(_shape_to_flat(coords + 137.0), reference)

        assert np.allclose(first, second, atol=1e-6)

    def test_skala_nie_zmienia_wyniku(self):
        coords = _random_shape(3)
        reference = _random_shape(2)

        first = procrustes_align(_shape_to_flat(coords), reference)
        second = procrustes_align(_shape_to_flat(coords * 2.5), reference)

        assert np.allclose(first, second, atol=1e-6)

    def test_obrot_nie_zmienia_wyniku(self):
        coords = _random_shape(4)
        reference = _random_shape(2)
        rotation = _rotation_matrix(37.0)

        first = procrustes_align(_shape_to_flat(coords), reference)
        second = procrustes_align(_shape_to_flat(coords @ rotation.T), reference)

        assert np.allclose(first, second, atol=1e-6)

    def test_zachowuje_widocznosc(self):
        coords = _random_shape(5)

        result = procrustes_align(_shape_to_flat(coords, visibility=0.42), _random_shape(2))

        assert np.allclose(result[2::3], 0.42)

    def test_rozne_ksztalty_daja_rozne_wyniki(self):
        reference = _random_shape(2)

        first = procrustes_align(_shape_to_flat(_random_shape(6)), reference)
        second = procrustes_align(_shape_to_flat(_random_shape(7)), reference)

        assert not np.allclose(first, second, atol=1e-3)

    def test_odbicie_lustrzane_to_inny_ksztalt(self):
        """Prokrustes NIE może usuwać odbicia — lustrzana morda to inny kształt."""
        coords = _random_shape(10)
        mirrored = coords * np.array([-1.0, 1.0])
        reference = _random_shape(2)

        first = procrustes_align(_shape_to_flat(coords), reference)
        second = procrustes_align(_shape_to_flat(mirrored), reference)

        assert not np.allclose(first, second, atol=1e-3)

    def test_wynik_ma_jednostkowy_rozmiar_centroidu(self):
        result = procrustes_align(_shape_to_flat(_random_shape(11)), _random_shape(2))

        assert _centroid_size(result) == pytest.approx(1.0, abs=1e-9)

    def test_bledna_dlugosc_podnosi_blad(self):
        with pytest.raises(ValueError, match="138"):
            procrustes_align(np.zeros(100), _random_shape(2))

    def test_bledny_ksztalt_referencyjny_podnosi_blad(self):
        with pytest.raises(ValueError, match="wymiary"):
            procrustes_align(_shape_to_flat(_random_shape(1)), np.zeros((10, 2)))

    def test_nieskonczone_keypoints_podnosza_czytelny_blad(self):
        """Anotacje DogFLW zawierają NaN dla punktów zasłoniętych — bez kontroli
        numpy wysypuje się głęboko w SVD z nieczytelnym LinAlgError."""
        flat = _shape_to_flat(_random_shape(1))
        flat[9] = np.nan

        with pytest.raises(ValueError, match="skończone"):
            procrustes_align(flat, _random_shape(2))

    def test_nieskonczony_ksztalt_referencyjny_podnosi_blad(self):
        reference = _random_shape(2)
        reference[5, 1] = np.inf

        with pytest.raises(ValueError, match="skończone"):
            procrustes_align(_shape_to_flat(_random_shape(1)), reference)

    def test_zdegenerowany_ksztalt_nie_wysypuje(self):
        """Wszystkie punkty w jednym miejscu — zero wariancji, wynik skończony."""
        coords = np.full((NUM_KEYPOINTS, 2), 250.0)

        result = procrustes_align(_shape_to_flat(coords), _random_shape(2))

        assert np.all(np.isfinite(result))
        assert np.allclose(result[0::3], 0.0)
        assert np.allclose(result[1::3], 0.0)

    def test_niewidoczne_punkty_nie_wplywaja_na_dopasowanie(self):
        """Punkty z widocznością 0 mają nieufne współrzędne — nie mogą ciągnąć fitu."""
        coords = _random_shape(12)
        hidden = [3, 7, 11]
        reference = _random_shape(2)

        first_coords = coords.copy()
        first_coords[hidden] = 0.0
        second_coords = coords.copy()
        second_coords[hidden] = 9999.0

        first = _shape_to_flat(first_coords)
        second = _shape_to_flat(second_coords)
        for index in hidden:
            first[index * 3 + 2] = 0.0
            second[index * 3 + 2] = 0.0

        aligned_first = procrustes_align(first, reference)
        aligned_second = procrustes_align(second, reference)

        visible = [index for index in range(NUM_KEYPOINTS) if index not in hidden]
        assert np.allclose(aligned_first[0::3][visible], aligned_second[0::3][visible], atol=1e-6)
        assert np.allclose(aligned_first[1::3][visible], aligned_second[1::3][visible], atol=1e-6)

    def test_same_niewidoczne_punkty_wracaja_do_wag_jednostkowych(self):
        """Gdy nic nie jest widoczne, wagi degenerują się — wynik ma być skończony."""
        result = procrustes_align(
            _shape_to_flat(_random_shape(13), visibility=0.0), _random_shape(2)
        )

        assert np.all(np.isfinite(result))
        assert _centroid_size(result) == pytest.approx(1.0, abs=1e-9)


    def test_zdegenerowana_referencja_podnosi_blad(self):
        """Referencja bez rozpiętości nie definiuje orientacji — dopasowanie byłoby dowolne."""
        with pytest.raises(ValueError, match="zdegenerowany"):
            procrustes_align(_shape_to_flat(_random_shape(14)), np.zeros((NUM_KEYPOINTS, 2)))


class TestMeanShape:
    """Kształt referencyjny metodą GPA."""

    def test_wspolrzedne_niewidocznych_punktow_nie_wchodza_do_sredniej(self):
        """
        Regresja: ważenie widocznością musi objąć także uśrednianie, nie tylko
        dopasowanie. Punkt zgubiony wpada w (0, 0) i z pełną wagą przesuwał nie
        tylko siebie, ale — przez centroid i rozmiar — cały kształt referencyjny.

        Test wprost: te same kształty, ta sama widoczność, różne śmieci pod
        punktem niewidocznym. Referencja musi wyjść identyczna.
        """
        rng = np.random.default_rng(3)
        base = _random_shape(15)
        hidden = 0

        with_zeros: list[np.ndarray] = []
        with_outliers: list[np.ndarray] = []
        for index in range(20):
            coords = base + rng.normal(0, 2.0, (NUM_KEYPOINTS, 2))
            if index % 2 == 0:
                zeroed = coords.copy()
                zeroed[hidden] = 0.0
                far = coords.copy()
                far[hidden] = 9999.0

                first = _shape_to_flat(zeroed)
                second = _shape_to_flat(far)
                first[hidden * 3 + 2] = 0.0
                second[hidden * 3 + 2] = 0.0
            else:
                first = _shape_to_flat(coords)
                second = _shape_to_flat(coords)

            with_zeros.append(first)
            with_outliers.append(second)

        assert np.allclose(mean_shape(with_zeros), mean_shape(with_outliers), atol=1e-9)

    def test_srednia_z_jednego_ksztaltu_to_ten_ksztalt_po_normalizacji(self):
        coords = _random_shape(8)

        result = mean_shape([_shape_to_flat(coords)])

        assert result.shape == (NUM_KEYPOINTS, 2)
        assert abs(float(np.mean(result))) < 1e-6, "kształt referencyjny jest wyśrodkowany"

    def test_srednia_odporna_na_przesuniecia_wejsc(self):
        coords = _random_shape(9)
        shapes = [_shape_to_flat(coords + offset) for offset in (0.0, 50.0, -30.0)]

        result = mean_shape(shapes)
        expected = mean_shape([_shape_to_flat(coords)])

        assert np.allclose(result, expected, atol=1e-6)

    def test_srednia_odporna_na_obroty_i_skale_wejsc(self):
        """Sedno GPA: te same kształty w różnych pozach dają jeden kształt średni."""
        coords = _random_shape(14)
        shapes = [
            _shape_to_flat(coords),
            _shape_to_flat((coords @ _rotation_matrix(25.0).T) * 3.0),
            _shape_to_flat((coords @ _rotation_matrix(-80.0).T) * 0.2 + 400.0),
        ]

        result = mean_shape(shapes)
        expected = mean_shape([_shape_to_flat(coords)])

        assert np.allclose(result, expected, atol=1e-6)

    def test_srednia_ma_jednostkowy_rozmiar_centroidu(self):
        shapes = [_shape_to_flat(_random_shape(seed)) for seed in (15, 16, 17)]

        result = mean_shape(shapes)

        assert float(np.sqrt(np.sum(result**2))) == pytest.approx(1.0, abs=1e-9)

    def test_gpa_zbiega_w_domyslnym_limicie_iteracji(self):
        """Domyślny budżet iteracji musi wystarczyć nawet na kształtach losowych.

        Gdyby wynik zależał od limitu, kształt referencyjny byłby artefaktem
        ustawienia, a nie danych.
        """
        shapes = [_shape_to_flat(_random_shape(seed)) for seed in (18, 19, 20, 21)]

        converged = mean_shape(shapes)
        with_more_budget = mean_shape(shapes, max_iterations=DEFAULT_MAX_GPA_ITERATIONS * 2)

        assert np.allclose(converged, with_more_budget, atol=1e-8)

    def test_zbyt_maly_limit_iteracji_daje_gorsze_przyblizenie(self):
        """Kontrola czujności testu wyżej: przy 3 iteracjach GPA jeszcze nie zbiega."""
        shapes = [_shape_to_flat(_random_shape(seed)) for seed in (18, 19, 20, 21)]

        assert not np.allclose(mean_shape(shapes, max_iterations=3), mean_shape(shapes), atol=1e-8)

    def test_pusta_lista_podnosi_blad(self):
        with pytest.raises(ValueError, match="co najmniej jednego"):
            mean_shape([])
