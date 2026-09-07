"""
Ręcznie wybrana para musi wejść do kolejki RAZEM ze swoją klatką neutralną.

Człowiek wybiera klatkę SZCZYTOWĄ — widzi ją na stronie wyboru. Klatka
neutralna nie jest przedmiotem wyboru, ale bez niej para jest bezwartościowa:
AU są różnicą względem niej. Gdyby wybór zabierał sam szczyt, dostalibyśmy
osierocone kadry zamiast par.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from packages.pipeline.quality_gate import QualityThresholds  # noqa: E402
from scripts.annotation.curate_for_review import pairs_for_names  # noqa: E402


def _zbior() -> dict:
    """Buduje COCO z jedną parą: klatka neutralna i szczytowa jednego psa."""
    return {
        "images": [
            {"id": 1, "file_name": "psA/neutral.jpg", "width": 1280, "height": 720},
            {"id": 2, "file_name": "psA/peak.jpg", "width": 1280, "height": 720},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "neutral_frame_id": 1,
                "frame_role": "neutral",
                "keypoints": [100.0, 100.0, 1.0] * 46,
                "bbox": [50.0, 50.0, 300.0, 300.0],
            },
            {
                "id": 2,
                "image_id": 2,
                "neutral_frame_id": 1,
                "frame_role": "peak",
                "keypoints": [120.0, 120.0, 1.0] * 46,
                "bbox": [50.0, 50.0, 300.0, 300.0],
            },
        ],
    }


class TestRecznyWybor:
    """Wybór klatki szczytowej zabiera jej bazę AU."""

    def test_wybrany_peak_przynosi_klatke_neutralna(self) -> None:
        pary = pairs_for_names(_zbior(), {"psA/peak.jpg"}, QualityThresholds())

        assert len(pary) == 1
        assert pary[0].peak_name == "psA/peak.jpg"
        assert pary[0].neutral["image_id"] == 1, "para musi wskazywać swoją klatkę neutralną"

    def test_wybor_omija_bramke_jakosci(self) -> None:
        """
        Sedno recznego wyboru: czlowiek widzial kadr i wie o nim wiecej niz prog.

        Progi ustawiamy tak, ze bramka odrzucilaby te pare (morda ma tu zero
        pikseli, bo wszystkie punkty leza w jednym miejscu), a mimo to para
        wchodzi.
        """
        nieosiagalne = QualityThresholds(min_face_width=9999.0, max_asymmetry=0.0)

        pary = pairs_for_names(_zbior(), {"psA/peak.jpg"}, nieosiagalne)

        assert len(pary) == 1, "reczny wybor ma byc mocniejszy niz bramka"

    def test_nieznana_nazwa_nie_wywraca(self) -> None:
        assert pairs_for_names(_zbior(), {"psB/nie-ma.jpg"}, QualityThresholds()) == []

    def test_peak_bez_klatki_neutralnej_nie_wchodzi(self) -> None:
        """Sam szczyt bez bazy AU to nie para — lepiej go pominac niz wpuscic."""
        zbior = _zbior()
        zbior["annotations"] = [a for a in zbior["annotations"] if a["frame_role"] != "neutral"]

        assert pairs_for_names(zbior, {"psA/peak.jpg"}, QualityThresholds()) == []
