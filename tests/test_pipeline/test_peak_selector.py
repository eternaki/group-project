"""
Testy wyboru klatek szczytowych, w szczególności ograniczania kandydatów.

Bramka jakości musi zawężać wybór PRZED wyborem, ale nie wolno jej robić przez
podanie selektorowi krótszej listy klatek: separacja peaków liczona jest
w pozycjach listy, więc na liście przefiltrowanej jedna pozycja odpowiada wielu
klatkom nagrania i twardy odstęp wycina praktycznie wszystkie szczyty. Ta
usterka niczego nie wysypuje — po prostu zbiór wychodzi pusty — więc ma tu
własny test.
"""

import numpy as np
import pytest

from packages.data.schemas import NUM_KEYPOINTS
from packages.models.delta_action_units import DeltaActionUnit
from packages.pipeline.peak_selector import PeakFrameSelector

from .kp_fixtures import make_frontal_kp

# Ile klatek liczy sztuczny trek w tych testach
_TRACK_LENGTH: int = 21
# Pozycja klatki neutralnej
_NEUTRAL: int = 0
# Bok syntetycznej klatki. Musi mieścić keypoints z `kp_fixtures` z zapasem:
# selektor odrzuca kadry, w których morda dotyka krawędzi.
_FRAME_SIDE: int = 320


def _visible_frontal_kp() -> np.ndarray:
    """
    Frontalne keypoints z ukrytymi punktami, których fixtura nie ustawia.

    `make_frontal_kp` zostawia nieustawione punkty w (0, 0) z wysoką
    widocznością, a selektor czyta taki układ jako mordę przyciętą krawędzią
    kadru i odrzuca kadr. Zerujemy im widoczność, żeby test mierzył separację,
    a nie artefakt fixtury.

    Returns:
        Tablica (138,) w układzie [x0, y0, v0, ...]
    """
    kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
    at_origin = (kp[:, 0] == 0.0) & (kp[:, 1] == 0.0)
    kp[at_origin, 2] = 0.0
    return kp.flatten()


def _delta_aus(strength: float) -> dict[str, DeltaActionUnit]:
    """Buduje komplet AU o zadanej sile aktywacji."""
    return {
        "AU25": DeltaActionUnit(
            name="AU25",
            ratio=1.0 + strength,
            delta=strength,
            is_active=strength > 0.0,
            confidence=0.9,
        )
    }


def _selector(min_separation: int) -> PeakFrameSelector:
    """Selektor z wyłączonymi filtrami obrazu (klatki są syntetyczne)."""
    return PeakFrameSelector(
        min_separation_frames=min_separation,
        min_tfm_threshold=0.01,
        frontal_only=False,
        min_keypoint_conf=0.1,
        min_sharpness=0.0,
    )


def _inputs() -> dict:
    """
    Trek, w którym mimika narasta co czwartą klatkę.

    Returns:
        Argumenty wspólne dla wywołań `select`
    """
    keypoints = [_visible_frontal_kp() for _ in range(_TRACK_LENGTH)]
    frames = [
        np.zeros((_FRAME_SIDE, _FRAME_SIDE, 3), dtype=np.uint8)
        for _ in range(_TRACK_LENGTH)
    ]
    deltas = [
        _delta_aus(0.5 if position % 4 == 0 and position != _NEUTRAL else 0.0)
        for position in range(_TRACK_LENGTH)
    ]
    return {
        "frames": frames,
        "keypoints_list": keypoints,
        "neutral_idx": _NEUTRAL,
        "delta_aus_list": deltas,
        "num_peaks": 10,
    }


class TestAllowedPositions:
    """Ograniczanie kandydatów zbiorem dozwolonych pozycji."""

    def test_bez_ograniczenia_wybiera_szczyty(self) -> None:
        selected = _selector(min_separation=4).select(**_inputs())
        assert selected

    def test_wybiera_tylko_z_dozwolonych(self) -> None:
        allowed = {4, 8}
        selected = _selector(min_separation=4).select(**_inputs(), allowed_positions=allowed)
        assert set(selected) <= allowed

    def test_pusty_zbior_dozwolonych_daje_brak_peakow(self) -> None:
        selected = _selector(min_separation=4).select(**_inputs(), allowed_positions=set())
        assert selected == []

    def test_separacja_liczy_sie_w_klatkach_nagrania(self) -> None:
        """
        Sedno: pozycje dozwolone są rzadkie, ale odległe w nagraniu.

        Pozycje 4, 12 i 20 dzieli po 8 klatek, więc przy separacji 8 muszą
        przejść WSZYSTKIE. Gdyby ograniczenie realizować przez skrócenie listy,
        sąsiadowałyby ze sobą (odstęp 1) i separacja wycięłaby dwie z trzech.
        """
        allowed = {4, 12, 20}
        selected = _selector(min_separation=8).select(**_inputs(), allowed_positions=allowed)
        assert set(selected) == allowed

    def test_klatka_neutralna_nigdy_nie_jest_peakiem(self) -> None:
        allowed = set(range(_TRACK_LENGTH))
        selected = _selector(min_separation=1).select(**_inputs(), allowed_positions=allowed)
        assert _NEUTRAL not in selected

    @pytest.mark.parametrize("separation", [1, 2, 4])
    def test_luzniejsza_separacja_nie_zmniejsza_liczby_peakow(self, separation: int) -> None:
        allowed = {4, 8, 12, 16, 20}
        loose = _selector(min_separation=separation).select(
            **_inputs(), allowed_positions=allowed
        )
        tight = _selector(min_separation=separation * 4).select(
            **_inputs(), allowed_positions=allowed
        )
        assert len(loose) >= len(tight)


class TestSelectorSanity:
    """Podstawy, na których opierają się testy ograniczania."""

    def test_keypoints_maja_oczekiwany_rozmiar(self) -> None:
        assert len(_visible_frontal_kp()) == NUM_KEYPOINTS * 3
