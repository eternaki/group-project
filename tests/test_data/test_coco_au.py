"""
Testy serializacji Action Units w formacie COCO.

Sprawdzają, że eksport zachowuje wiarygodność pomiaru (is_active, confidence),
a odczyt pozostaje wsteczne kompatybilny ze starym formatem (samo ratio).
"""

from packages.data.coco import (
    COCODataset,
    au_analysis_from_delta_aus,
    au_auto_verdict,
    au_auto_verdicts,
    au_ratio,
)
from packages.models.delta_action_units import DeltaActionUnit


def _make_au(name: str, ratio: float, is_active: bool, confidence: float) -> DeltaActionUnit:
    """Buduje pojedynczy AU do testów."""
    return DeltaActionUnit(
        name=name,
        ratio=ratio,
        delta=ratio - 1.0,
        is_active=is_active,
        confidence=confidence,
    )


class TestAuRatio:
    """Odczyt wartości ratio z obu formatów au_analysis."""

    def test_odczytuje_stary_format_float(self):
        assert au_ratio(0.85) == 0.85

    def test_odczytuje_nowy_format_slownikowy(self):
        assert au_ratio({"ratio": 0.85, "is_active": False, "confidence": 0.0}) == 0.85


class TestAuAnalysisFromDeltaAus:
    """Serializacja AU do pola au_analysis."""

    def test_zachowuje_ratio_is_active_i_confidence(self):
        delta_aus = {
            "AU101": _make_au("AU101", 1.42, True, 0.91),
            "EAD103": _make_au("EAD103", 3.0, False, 0.0),
        }

        result = au_analysis_from_delta_aus(delta_aus)

        assert result["AU101"] == {"ratio": 1.42, "is_active": True, "confidence": 0.91}
        assert result["EAD103"] == {"ratio": 3.0, "is_active": False, "confidence": 0.0}

    def test_klamrowane_ucho_nie_wyglada_na_aktywne(self):
        """Klamrowany EAD (confidence 0) musi być odróżnialny od realnej aktywacji."""
        delta_aus = {"EAD103": _make_au("EAD103", 3.0, False, 0.0)}

        result = au_analysis_from_delta_aus(delta_aus)

        assert result["EAD103"]["is_active"] is False
        assert result["EAD103"]["confidence"] == 0.0


class TestWalidatorAkceptujeObaFormaty:
    """Walidator COCO nie może zgłaszać nowego formatu AU jako błędu."""

    def _waliduj(self, au_analysis: dict):
        from scripts.annotation.validate_coco import COCOValidator, ValidationResult

        validator = COCOValidator(strict=True)
        result = ValidationResult()
        annotations = [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "au_analysis": au_analysis,
            }
        ]
        validator.validate_annotations(annotations, {1}, {1}, result)
        return result

    def test_nowy_format_bez_ostrzezen(self):
        result = self._waliduj(
            {"AU101": {"ratio": 1.5, "is_active": True, "confidence": 0.9}}
        )

        assert result.warnings == []
        assert result.errors == []

    def test_stary_format_bez_ostrzezen(self):
        result = self._waliduj({"AU101": 1.5})

        assert result.warnings == []
        assert result.errors == []

    def test_smieciowa_wartosc_daje_ostrzezenie(self):
        result = self._waliduj({"AU101": "nie-liczba"})

        assert len(result.warnings) == 1


class TestStatisticsZNowymFormatem:
    """Statystyki datasetu liczą się z obu formatów au_analysis."""

    def _dataset_z_au(self, au_analysis: dict) -> COCODataset:
        dataset = COCODataset()
        image_id = dataset.add_image(file_name="a.jpg", width=100, height=100)
        dataset.add_annotation(
            image_id=image_id,
            bbox=[0, 0, 10, 10],
            au_analysis=au_analysis,
        )
        return dataset

    def test_liczy_srednia_z_nowego_formatu(self):
        dataset = self._dataset_z_au(
            {"AU101": {"ratio": 1.5, "is_active": True, "confidence": 0.9}}
        )

        stats = dataset.get_statistics()

        assert stats["action_units"]["AU101"]["avg_delta"] == 1.5

    def test_liczy_srednia_ze_starego_formatu(self):
        dataset = self._dataset_z_au({"AU101": 1.5})

        stats = dataset.get_statistics()

        assert stats["action_units"]["AU101"]["avg_delta"] == 1.5

    def test_zapisuje_pelna_strukture_au_w_anotacji(self):
        au_analysis = {"AU101": {"ratio": 1.5, "is_active": True, "confidence": 0.9}}
        dataset = self._dataset_z_au(au_analysis)

        annotation = dataset.to_dict()["annotations"][0]

        assert annotation["au_analysis"]["AU101"]["confidence"] == 0.9


class TestAuAutoVerdict:
    """Trójstanowa etykieta AU z szumowego gejtu."""

    def test_sygnal_powyzej_szumu_daje_active(self):
        assert au_auto_verdict({"ratio": 1.5, "is_active": True, "snr": 2.0}) == "active"

    def test_reguła_zapalona_ale_utopiona_w_szumie_daje_not_observable(self):
        # is_active=True, ale sygnał nie przewyższa szumu — nie potwierdzamy aktywacji
        assert (
            au_auto_verdict({"ratio": 1.2, "is_active": True, "snr": 0.4})
            == "not_observable"
        )

    def test_spoczynek_w_granicach_szumu_daje_inactive(self):
        assert (
            au_auto_verdict({"ratio": 1.01, "is_active": False, "snr": 0.3}) == "inactive"
        )

    def test_brak_zmierzonego_szumu_daje_not_observable(self):
        # Bez snr nie ma jak porównać — to brak wiedzy, nie spoczynek
        assert (
            au_auto_verdict({"ratio": 1.5, "is_active": True}) == "not_observable"
        )

    def test_stary_format_float_daje_not_observable(self):
        assert au_auto_verdict(1.5) == "not_observable"

    def test_mapa_calego_au_analysis(self):
        au_analysis = {
            "AU101": {"ratio": 1.5, "is_active": True, "snr": 2.0},
            "AU25": {"ratio": 1.01, "is_active": False, "snr": 0.2},
        }
        assert au_auto_verdicts(au_analysis) == {"AU101": "active", "AU25": "inactive"}
