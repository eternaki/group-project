"""
Testy bramki jakości kadru (`packages.pipeline.quality_gate`).

Bramka ma rozróżniać trzy sytuacje, które audyt wizualny mieszał ze sobą:
kadr dobry, kadr z profilu i kadr z niepewnymi keypoints. Osobno sprawdzamy,
że bramka odrzuca parę, gdy zepsuta jest sama klatka NEUTRALNA — to był
najtrudniejszy do zauważenia przypadek, bo w interfejsie widać tylko szczyt.
"""

import numpy as np
import pytest

from packages.data.schemas import KP, NUM_KEYPOINTS
from packages.pipeline.quality_gate import (
    REASON_ASYMMETRY,
    REASON_NO_KEYPOINTS,
    REASON_SMALL_FACE,
    REASON_WEAK_KEYPOINTS,
    QualityThresholds,
    assess_frame,
    assess_pair,
    face_asymmetry,
    face_width,
    split_keypoints,
    weak_keypoint_ratio,
)

from .kp_fixtures import make_frontal_kp, make_low_visibility_kp

# Ułamek, do którego ściskamy lewą połówkę mordy, żeby udawać ujęcie z profilu
_PROFILE_SQUEEZE: float = 0.25


def _as_list(flat: np.ndarray) -> list[float]:
    """Zamienia tablicę keypoints na listę, jakiej używa COCO."""
    return [float(value) for value in flat]


def make_profile_kp() -> list[float]:
    """
    Tworzy kadr z profilu — lewa połówka mordy ściśnięta do osi.

    Odwzorowuje skrócenie perspektywiczne: przy obrocie głowy punkty jednej
    połówki zbliżają się do osi mordy, druga zostaje na miejscu.

    Returns:
        Lista 138 wartości COCO
    """
    kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
    axis_x = float(kp[KP.NOSE_TIP, 0])
    left_side = kp[:, 0] < axis_x
    kp[left_side, 0] = axis_x - (axis_x - kp[left_side, 0]) * _PROFILE_SQUEEZE
    return _as_list(kp.flatten())


def make_tiny_face_kp() -> list[float]:
    """
    Tworzy frontalny kadr, na którym morda jest za mała do weryfikacji.

    Returns:
        Lista 138 wartości COCO
    """
    kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
    center = kp[:, :2].mean(axis=0)
    # Skalujemy geometrię wokół środka — proporcje (więc i symetria) zostają
    kp[:, :2] = center + (kp[:, :2] - center) * 0.1
    return _as_list(kp.flatten())


class TestSplitKeypoints:
    """Rozdzielanie płaskiej listy COCO."""

    def test_zwraca_wspolrzedne_i_pewnosci(self) -> None:
        coords, confidences = split_keypoints(_as_list(make_frontal_kp()))
        assert coords.shape == (NUM_KEYPOINTS, 2)
        assert confidences.shape == (NUM_KEYPOINTS,)

    def test_odrzuca_zla_dlugosc(self) -> None:
        with pytest.raises(ValueError, match="138"):
            split_keypoints([0.0, 1.0, 2.0])


class TestFaceAsymmetry:
    """Miara asymetrii połówek mordy."""

    def test_front_jest_prawie_symetryczny(self) -> None:
        coords, _ = split_keypoints(_as_list(make_frontal_kp()))
        assert face_asymmetry(coords) < 0.1

    def test_profil_lamie_symetrie(self) -> None:
        coords, _ = split_keypoints(make_profile_kp())
        assert face_asymmetry(coords) > 0.2

    def test_profil_jest_bardziej_asymetryczny_niz_front(self) -> None:
        frontal, _ = split_keypoints(_as_list(make_frontal_kp()))
        profile, _ = split_keypoints(make_profile_kp())
        assert face_asymmetry(profile) > face_asymmetry(frontal)


class TestWeakKeypointRatio:
    """Udział punktów, którym detektor sam nie ufa."""

    def test_pewne_punkty_daja_zero(self) -> None:
        _, confidences = split_keypoints(_as_list(make_frontal_kp()))
        assert weak_keypoint_ratio(confidences) == 0.0

    def test_niepewne_punkty_daja_jeden(self) -> None:
        _, confidences = split_keypoints(_as_list(make_low_visibility_kp()))
        assert weak_keypoint_ratio(confidences) == 1.0


class TestFaceWidth:
    """Szerokość mordy w pikselach."""

    def test_mierzy_rozstaw_policzkow(self) -> None:
        coords, _ = split_keypoints(_as_list(make_frontal_kp()))
        assert face_width(coords) == pytest.approx(124.0, abs=1.0)


class TestAssessFrame:
    """Ocena pojedynczej klatki."""

    def test_dobry_kadr_przechodzi(self) -> None:
        quality = assess_frame(_as_list(make_frontal_kp()))
        assert quality.is_usable
        assert quality.reasons == ()

    def test_profil_odrzucony_z_powodem(self) -> None:
        quality = assess_frame(make_profile_kp())
        assert not quality.is_usable
        assert REASON_ASYMMETRY in quality.reasons

    def test_niepewne_keypoints_odrzucone_z_powodem(self) -> None:
        quality = assess_frame(_as_list(make_low_visibility_kp()))
        assert not quality.is_usable
        assert REASON_WEAK_KEYPOINTS in quality.reasons

    def test_mala_morda_odrzucona_z_powodem(self) -> None:
        quality = assess_frame(make_tiny_face_kp())
        assert not quality.is_usable
        assert REASON_SMALL_FACE in quality.reasons

    def test_brak_keypoints_nie_wysypuje_bramki(self) -> None:
        quality = assess_frame(None)
        assert not quality.is_usable
        assert quality.reasons == (REASON_NO_KEYPOINTS,)

    def test_progi_da_sie_poluzowac(self) -> None:
        luzne = QualityThresholds(max_asymmetry=1.0, max_weak_ratio=1.0, min_face_width=0.0)
        assert assess_frame(make_profile_kp(), luzne).is_usable


class TestAssessPair:
    """Ocena pary (szczytowa, neutralna) — jednostka pomiaru AU."""

    def test_dwie_dobre_klatki_przechodza(self) -> None:
        good = _as_list(make_frontal_kp())
        assert assess_pair(good, good).is_usable

    def test_zepsuty_szczyt_odrzuca_pare(self) -> None:
        pair = assess_pair(make_profile_kp(), _as_list(make_frontal_kp()))
        assert not pair.is_usable
        assert any(reason.startswith("szczytowa") for reason in pair.reasons)

    def test_zepsuta_neutralna_odrzuca_pare(self) -> None:
        """Zepsuta baza AU unieważnia pomiar tak samo jak zepsuty szczyt."""
        pair = assess_pair(_as_list(make_frontal_kp()), make_profile_kp())
        assert not pair.is_usable
        assert any(reason.startswith("neutralna") for reason in pair.reasons)

    def test_powody_wskazuja_ktora_klatka_zawiodla(self) -> None:
        pair = assess_pair(make_profile_kp(), _as_list(make_low_visibility_kp()))
        assert any(reason.startswith("szczytowa") for reason in pair.reasons)
        assert any(reason.startswith("neutralna") for reason in pair.reasons)
