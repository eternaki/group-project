"""
Testy złożenia finalnego zbioru.

Sprawdzają dwie rzeczy, na których zbiór stoi: że punkty po przycięciu opisują
TEN obraz, który leży obok nich, oraz że praca człowieka nie ginie po drodze —
w szczególności poprawka punktów klatki NEUTRALNEJ, którą łatwo zgubić, bo
zapisuje się pod własną ścieżką, a nie pod kluczem pary.
"""

import numpy as np
import pytest

from scripts.annotation.build_final_dataset import (
    CropBox,
    bbox_from_keypoints,
    crop_and_scale,
    face_box,
    remap_keypoints,
)

NUM_KEYPOINTS = 46


def _kp(points: list[tuple[float, float]], visibility: float = 0.9) -> list[float]:
    """
    Buduje płaskie keypoints z listy punktów, resztę oznaczając jako niewidoczne.

    Args:
        points: Współrzędne punktów pewnych
        visibility: Pewność wpisana tym punktom

    Returns:
        Płaska lista 138 wartości
    """
    flat = [0.0] * (NUM_KEYPOINTS * 3)
    for index, (x, y) in enumerate(points):
        flat[index * 3] = x
        flat[index * 3 + 1] = y
        flat[index * 3 + 2] = visibility
    return flat


class TestFaceBox:
    """Wyznaczanie kadru mordy z punktów."""

    def test_kadr_obejmuje_wszystkie_pewne_punkty(self) -> None:
        """Punkt poza kadrem znaczyłby, że zbiór wskazuje poza własne zdjęcie."""
        points = [(100.0, 200.0), (140.0, 210.0), (120.0, 240.0),
                  (110.0, 205.0), (130.0, 235.0), (125.0, 220.0)]
        box = face_box(_kp(points), width=640, height=480)

        assert box is not None
        for x, y in points:
            assert box.x0 <= x <= box.x1
            assert box.y0 <= y <= box.y1

    def test_niepewne_punkty_nie_rozdymaja_kadru(self) -> None:
        """Jedna zabłąkana predykcja na tle rozciągnęłaby kadr na pół obrazu."""
        points = [(100.0, 200.0), (140.0, 210.0), (120.0, 240.0),
                  (110.0, 205.0), (130.0, 235.0), (125.0, 220.0)]
        flat = _kp(points)
        flat[40 * 3], flat[40 * 3 + 1], flat[40 * 3 + 2] = 600.0, 20.0, 0.05

        box = face_box(flat, width=640, height=480)

        assert box is not None
        assert box.x1 < 300

    def test_kadr_nie_wychodzi_poza_klatke(self) -> None:
        """Morda przy krawędzi nie może dać kadru o ujemnym początku."""
        points = [(2.0, 3.0), (18.0, 5.0), (10.0, 20.0),
                  (6.0, 4.0), (14.0, 18.0), (11.0, 12.0)]
        box = face_box(_kp(points), width=640, height=480)

        assert box is not None
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= 640 and box.y1 <= 480

    def test_za_malo_pewnych_punktow_daje_brak_kadru(self) -> None:
        """Kadr z trzech punktów byłby zgadywaniem, a nie pomiarem."""
        assert face_box(_kp([(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]), 640, 480) is None


class TestCropAndScale:
    """Wycinanie i ograniczanie rozmiaru."""

    def test_maly_kadr_nie_jest_powiekszany(self) -> None:
        """Skalowanie w górę dodałoby tylko szum interpolacji."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        crop, scale = crop_and_scale(image, CropBox(10, 10, 110, 90))

        assert scale == 1.0
        assert crop.shape[:2] == (80, 100)

    def test_duzy_kadr_schodzi_do_progu_zachowujac_proporcje(self) -> None:
        """Zmiana proporcji zniekształciłaby mordę i zafałszowała geometrię AU."""
        image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        crop, scale = crop_and_scale(image, CropBox(0, 0, 1024, 512))

        assert crop.shape[1] == 512
        assert crop.shape[0] == 256
        assert scale == pytest.approx(0.5)


class TestRemapKeypoints:
    """Przeniesienie punktów do układu wycinka."""

    def test_punkty_ladują_w_ukladzie_wycinka(self) -> None:
        """Bez przeliczenia punkty opisywałyby oryginał, a obraz byłby wycinkiem."""
        flat = _kp([(120.0, 220.0), (140.0, 230.0)])
        moved = remap_keypoints(flat, CropBox(100, 200, 300, 400), scale=1.0)

        assert moved[0] == pytest.approx(20.0)
        assert moved[1] == pytest.approx(20.0)
        assert moved[3] == pytest.approx(40.0)
        assert moved[4] == pytest.approx(30.0)

    def test_skala_dotyczy_wspolrzednych_ale_nie_widocznosci(self) -> None:
        """Przemnożona widoczność przestałaby być prawdopodobieństwem."""
        flat = _kp([(120.0, 220.0)], visibility=0.8)
        moved = remap_keypoints(flat, CropBox(100, 200, 300, 400), scale=0.5)

        assert moved[0] == pytest.approx(10.0)
        assert moved[1] == pytest.approx(10.0)
        assert moved[2] == pytest.approx(0.8)

    def test_punkty_mieszcza_sie_w_wycinku_wyznaczonym_z_nich(self) -> None:
        """Spięcie obu kroków: kadr z punktów i punkty w kadrze muszą się zgadzać."""
        points = [(100.0, 200.0), (140.0, 210.0), (120.0, 240.0),
                  (110.0, 205.0), (130.0, 235.0), (125.0, 220.0)]
        flat = _kp(points)
        box = face_box(flat, width=640, height=480)
        assert box is not None

        moved = remap_keypoints(flat, box, scale=1.0)
        coords = np.asarray(moved, dtype=float).reshape(NUM_KEYPOINTS, 3)
        visible = coords[coords[:, 2] > 0.3]

        assert (visible[:, 0] >= 0).all() and (visible[:, 0] <= box.width).all()
        assert (visible[:, 1] >= 0).all() and (visible[:, 1] <= box.height).all()


class TestBboxFromKeypoints:
    """Boks anotacji liczony w układzie wycinka."""

    def test_boks_obejmuje_pewne_punkty(self) -> None:
        """Boks ma opisywać to, co na obrazie jest — mordę."""
        flat = _kp([(10.0, 20.0), (50.0, 20.0), (30.0, 60.0),
                    (20.0, 30.0), (40.0, 50.0), (30.0, 40.0)])
        x, y, width, height = bbox_from_keypoints(flat)

        assert (x, y) == (10.0, 20.0)
        assert width == pytest.approx(40.0)
        assert height == pytest.approx(40.0)

    def test_brak_pewnych_punktow_daje_pusty_boks(self) -> None:
        """Zamiast wyjątku — pusty boks, żeby jedna klatka nie wysadziła przebiegu."""
        assert bbox_from_keypoints([0.0] * (NUM_KEYPOINTS * 3)) == [0.0, 0.0, 0.0, 0.0]
