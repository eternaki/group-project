"""
Paczka musi zmiescic sie w limicie gita, nie tracac informacji.

GitHub ODRZUCA pliki powyzej 100 MB. Kolejka przy 9519 parach urosla do
129.5 MB i push zostal odbity — 76% pliku to byly liczby zapisane z pelna
precyzja float, na przyklad 1213.4567890123457. Ta precyzja jest fikcyjna:
punkty to pozycje w pikselach przy bledzie modelu rzedu pikseli, a ratio AU
porownuje sie z progiem 0.15.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation.build_work_pack import shrink  # noqa: E402


class TestObcinaniePrecyzji:
    """Zaokrąglanie ma zmniejszać plik, a nie zmieniać treści."""

    def test_punkty_do_dziesiatej_piksela(self) -> None:
        wynik = shrink([{"keypoints": [1213.4567890123457, 2.0, 1.0]}])

        assert wynik[0]["keypoints"] == [1213.5, 2.0, 1.0]

    def test_pomiary_do_czterech_miejsc(self) -> None:
        """Prog aktywacji AU to 0.15 — cztery miejsca to zapas tysiackrotny."""
        wynik = shrink([{"au_analysis": {"AU101": {"ratio": 1.1523456789}}}])

        assert wynik[0]["au_analysis"]["AU101"]["ratio"] == 1.1523

    def test_pola_nieliczbowe_zostaja(self) -> None:
        wpis = {"frame_role": "peak", "image_id": 7, "breed": "beagle"}

        assert shrink([wpis])[0] == wpis

    def test_nie_rusza_oryginalu(self) -> None:
        """Kuracja trzyma pelna precyzje — obcinamy tylko przy zapisie paczki."""
        wpis = {"keypoints": [1.23456789, 2.0, 1.0]}

        shrink([wpis])

        assert wpis["keypoints"][0] == 1.23456789

    def test_zaokraglanie_realnie_zmniejsza(self) -> None:
        import json

        wpisy = [{"keypoints": [1213.4567890123457] * 138} for _ in range(50)]

        przed = len(json.dumps(wpisy))
        po = len(json.dumps(shrink(wpisy)))

        assert po < przed / 2, "zaokrąglenie ma dawać realną oszczędność"
