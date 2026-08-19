"""Testy pozy głowy psa (metryki niezależne od długości pyska)."""

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS
from packages.models.head_pose import estimate_head_pose


def _keypoints(points: dict[int, tuple[float, float]]) -> np.ndarray:
    """Buduje płaską tablicę keypoints; punkty niepodane są w (0,0) z widocznością 1."""
    flat = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
    flat[2::3] = 1.0
    for index, (x, y) in points.items():
        flat[index * 3] = x
        flat[index * 3 + 1] = y
    return flat


class TestYawAsymmetry:
    """Obrót lewo/prawo mierzony asymetrią odległości kąciki oczu ↔ nos."""

    def test_symetryczna_morda_daje_zero(self):
        pose = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (50.0, 120.0),
            })
        )

        assert abs(pose.yaw_asymmetry) < 1e-6

    def test_dlugi_pysk_nie_wplywa_na_wynik(self):
        """Nos przesunięty w dół (długi pysk) nadal daje pozę frontalną."""
        pose = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (50.0, 300.0),
            })
        )

        assert abs(pose.yaw_asymmetry) < 1e-6
        assert pose.is_frontal is True

    def test_obrot_w_bok_daje_przeciwne_znaki(self):
        left_turn = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (20.0, 120.0),
            })
        )
        right_turn = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (80.0, 120.0),
            })
        )

        assert left_turn.yaw_asymmetry * right_turn.yaw_asymmetry < 0

    def test_metryka_niezalezna_od_skali_obrazu(self):
        small = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (20.0, 120.0),
            })
        )
        large = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (400.0, 500.0),
                KP.RIGHT_EYE_INNER: (600.0, 500.0),
                KP.NOSE_TIP: (200.0, 1200.0),
            })
        )

        assert abs(small.yaw_asymmetry - large.yaw_asymmetry) < 1e-6


class TestRoll:
    """Przechylenie liczone z linii wewnętrznych kącików oczu."""

    def test_pozioma_linia_oczu_daje_zero(self):
        pose = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 50.0),
                KP.RIGHT_EYE_INNER: (60.0, 50.0),
                KP.NOSE_TIP: (50.0, 120.0),
            })
        )

        assert abs(pose.roll) < 1e-6

    def test_przechylona_glowa_nie_jest_frontalna(self):
        pose = estimate_head_pose(
            _keypoints({
                KP.LEFT_EYE_INNER: (40.0, 20.0),
                KP.RIGHT_EYE_INNER: (60.0, 80.0),
                KP.NOSE_TIP: (50.0, 120.0),
            }),
            max_roll=30.0,
        )

        assert pose.is_frontal is False


class TestNaReferencjiDogFLW:
    """Metryka sprawdzona na ręcznie anotowanych landmarkach (nie na predykcjach)."""

    def test_rozklad_na_zbiorze_testowym_jest_wysrodkowany(self, tmp_path):
        import glob
        import json

        files = sorted(glob.glob("data/dogflw_raw/DogFLW/test/labels/*.json"))
        if not files:
            import pytest

            pytest.skip("Brak lokalnej kopii DogFLW")

        values = []
        for path in files:
            landmarks = json.load(open(path, encoding="utf-8"))["landmarks"]
            if len(landmarks) != NUM_KEYPOINTS:
                continue
            flat = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
            for index, (x, y) in enumerate(landmarks):
                flat[index * 3] = x
                flat[index * 3 + 1] = y
                flat[index * 3 + 2] = 1.0
            values.append(estimate_head_pose(flat).yaw_asymmetry)

        median_abs = float(np.median(np.abs(values)))
        share_rejected = float(np.mean(np.abs(values) > 0.35))

        assert median_abs < 0.10, "Fronty muszą dawać wartości bliskie zeru"
        assert share_rejected < 0.05, "Twardy limit nie może odrzucać frontalnych mord"
