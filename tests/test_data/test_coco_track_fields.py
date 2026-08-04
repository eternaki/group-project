"""
Testy pól COCO potrzebnych do trenowania własnej sieci AU (Sprint 16).

Sprawdzają dwie rzeczy:

1. Anotacja wie, do którego psa (treku) należy, jaką rolę pełni klatka, skąd
   pochodzi etykieta i jaki szum AU zmierzono w tym treku.
2. Aktywacja AU da się odsiać od drgania keypoints. Zmierzona mediana szumu ratio
   po wygładzaniu to 0.166 przy progu aktywacji 0.15, więc samo `is_active` dla
   większości AU zapala się od szumu i bez odniesienia do zmierzonego szumu jest
   nieinformatywne.
"""

import pytest

from packages.data.coco import (
    FLAT_KEYPOINTS_LENGTH,
    FRAME_ROLE_NEUTRAL,
    FRAME_ROLE_PEAK,
    LABEL_SOURCE_AUTO_RULES,
    LABEL_SOURCE_HUMAN_VERIFIED,
    COCODataset,
    TrackAnnotation,
    au_analysis_from_delta_aus,
    au_signal_above_noise,
)
from packages.models.delta_action_units import DeltaActionUnit


def _make_au(name: str, ratio: float, is_active: bool, confidence: float = 0.9) -> DeltaActionUnit:
    """Buduje pojedynczy AU do testów."""
    return DeltaActionUnit(
        name=name,
        ratio=ratio,
        delta=ratio - 1.0,
        is_active=is_active,
        confidence=confidence,
    )


def _annotation(track: TrackAnnotation) -> dict:
    """Zapisuje anotację z polami treku i zwraca jej słownik."""
    dataset = COCODataset()
    image_id = dataset.add_image(file_name="a.jpg", width=640, height=480)
    dataset.add_annotation(image_id=image_id, bbox=[0, 0, 10, 10], track=track)
    return dataset.to_dict()["annotations"][0]


class TestWalidacjaPolTreku:
    """Pola treku, których nie da się poprawnie zinterpretować, mają nie powstać."""

    def test_odrzuca_nieznana_role_klatki(self):
        with pytest.raises(ValueError, match="Nieznana rola klatki"):
            TrackAnnotation(track_id=1, frame_role="szczyt")

    def test_odrzuca_nieznane_zrodlo_etykiety(self):
        with pytest.raises(ValueError, match="Nieznane źródło etykiety"):
            TrackAnnotation(track_id=1, frame_role=FRAME_ROLE_PEAK, label_source="reguły")

    def test_odrzuca_szum_bez_liczby_prob(self):
        """Szum bez liczby prób jest nieskorygowalny — sigma z 3 klatek jest obciążona."""
        with pytest.raises(ValueError, match="bez liczby prób"):
            TrackAnnotation(
                track_id=1,
                frame_role=FRAME_ROLE_PEAK,
                au_noise={"AU101": 0.12},
                au_sample_count={},
            )

    def test_odrzuca_ksztalt_o_zlej_dlugosci(self):
        with pytest.raises(ValueError, match="wartości"):
            TrackAnnotation(
                track_id=1,
                frame_role=FRAME_ROLE_PEAK,
                procrustes_keypoints=[0.1] * 92,
            )

    def test_domyslne_zrodlo_etykiety_to_reguly(self):
        track = TrackAnnotation(track_id=1, frame_role=FRAME_ROLE_PEAK)

        assert track.label_source == LABEL_SOURCE_AUTO_RULES


class TestPolaTreku:
    """track_id, rola klatki, źródło etykiety, szum AU, kanoniczny kształt."""

    def test_zapisuje_track_id(self):
        annotation = _annotation(TrackAnnotation(track_id=7, frame_role=FRAME_ROLE_PEAK))

        assert annotation["track_id"] == 7

    def test_zapisuje_role_klatki(self):
        annotation = _annotation(TrackAnnotation(track_id=7, frame_role=FRAME_ROLE_NEUTRAL))

        assert annotation["frame_role"] == FRAME_ROLE_NEUTRAL

    def test_zapisuje_zrodlo_etykiety(self):
        annotation = _annotation(
            TrackAnnotation(
                track_id=7,
                frame_role=FRAME_ROLE_PEAK,
                label_source=LABEL_SOURCE_HUMAN_VERIFIED,
            )
        )

        assert annotation["label_source"] == LABEL_SOURCE_HUMAN_VERIFIED

    def test_zapisuje_szum_razem_z_liczba_prob(self):
        annotation = _annotation(
            TrackAnnotation(
                track_id=7,
                frame_role=FRAME_ROLE_PEAK,
                au_noise={"AU101": 0.42},
                au_sample_count={"AU101": 5},
            )
        )

        assert annotation["au_noise"]["AU101"] == 0.42
        assert annotation["au_sample_count"]["AU101"] == 5

    def test_brak_zmierzonego_szumu_jest_zapisany_jawnie(self):
        """Pusty słownik znaczy „nie zmierzono", a nie „szum zerowy" — musi być widoczny."""
        annotation = _annotation(TrackAnnotation(track_id=7, frame_role=FRAME_ROLE_PEAK))

        assert annotation["au_noise"] == {}
        assert annotation["au_sample_count"] == {}

    def test_zapisuje_ksztalt_prokrustesa(self):
        shape = [0.1] * FLAT_KEYPOINTS_LENGTH
        annotation = _annotation(
            TrackAnnotation(
                track_id=7,
                frame_role=FRAME_ROLE_PEAK,
                procrustes_keypoints=shape,
            )
        )

        assert len(annotation["procrustes_keypoints"]) == FLAT_KEYPOINTS_LENGTH

    def test_anotacja_bez_treku_nie_ma_pol_treku(self):
        """Wsteczna zgodność: stary zapis (obraz bez trekowania) zostaje bez zmian."""
        dataset = COCODataset()
        image_id = dataset.add_image(file_name="a.jpg", width=640, height=480)
        dataset.add_annotation(image_id=image_id, bbox=[0, 0, 10, 10])

        annotation = dataset.to_dict()["annotations"][0]

        assert "track_id" not in annotation
        assert "frame_role" not in annotation
        assert "au_noise" not in annotation


class TestSygnalWzgledemSzumu:
    """Porównanie |ratio - 1| ze zmierzonym szumem tego AU w tym treku."""

    def test_liczy_stosunek_sygnalu_do_szumu(self):
        result = au_analysis_from_delta_aus(
            {"AU101": _make_au("AU101", 1.30, True)},
            au_noise={"AU101": 0.10},
        )

        assert result["AU101"]["noise"] == 0.10
        assert result["AU101"]["snr"] == pytest.approx(3.0)

    def test_aktywacja_ponizej_szumu_jest_rozpoznawalna(self):
        """Delta 0.16 przy szumie 0.166 to szum, choć próg aktywacji przekroczony."""
        result = au_analysis_from_delta_aus(
            {"AU101": _make_au("AU101", 1.16, True)},
            au_noise={"AU101": 0.166},
        )

        assert result["AU101"]["is_active"] is True
        assert au_signal_above_noise(result["AU101"]) is False

    def test_sygnal_powyzej_szumu(self):
        result = au_analysis_from_delta_aus(
            {"AU101": _make_au("AU101", 1.50, True)},
            au_noise={"AU101": 0.166},
        )

        assert au_signal_above_noise(result["AU101"]) is True

    def test_brak_pomiaru_szumu_to_brak_wiedzy_a_nie_zero(self):
        """AU o mniej niż MIN_NOISE_SAMPLES pomiarach nie ma klucza w au_noise."""
        result = au_analysis_from_delta_aus(
            {"AU101": _make_au("AU101", 1.30, True)},
            au_noise={"EAD101": 0.10},
        )

        assert result["AU101"]["noise"] is None
        assert result["AU101"]["snr"] is None
        assert au_signal_above_noise(result["AU101"]) is None

    def test_zerowy_szum_nie_daje_nieskonczonosci(self):
        """Ratio stałe w całym treku (np. klamrowane) — dzielenie dałoby inf, czyli zły JSON."""
        result = au_analysis_from_delta_aus(
            {"EAD103": _make_au("EAD103", 3.0, False, confidence=0.0)},
            au_noise={"EAD103": 0.0},
        )

        assert result["EAD103"]["noise"] == 0.0
        assert result["EAD103"]["snr"] is None

    def test_bez_slownika_szumu_zapis_zostaje_stary(self):
        """Wsteczna zgodność: wywołanie bez szumu daje dokładnie stary zestaw pól."""
        result = au_analysis_from_delta_aus({"AU101": _make_au("AU101", 1.30, True)})

        assert result["AU101"] == {"ratio": 1.30, "is_active": True, "confidence": 0.9}


class TestAuSignalAboveNoise:
    """Czytnik odsiewu — trzystanowy, bo brak pomiaru to brak wiedzy."""

    def test_stary_format_float_nie_niesie_wiedzy_o_szumie(self):
        assert au_signal_above_noise(1.42) is None

    def test_stary_format_slownikowy_bez_szumu(self):
        assert au_signal_above_noise({"ratio": 1.42, "is_active": True, "confidence": 0.9}) is None

    def test_nieliczbowy_snr_nie_niesie_wiedzy(self):
        assert au_signal_above_noise({"ratio": 1.42, "snr": None}) is None
