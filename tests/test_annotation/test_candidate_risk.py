"""
Sortowanie kandydatow wedlug ryzyka: model zmienia KOLEJNOSC, nie decyzje.

Powod, dla ktorego nie odrzuca sam: na odlozonej polowie decyzji czlowieka
precyzja nawet 25 najpewniejszych typow to 64%, czyli co trzeci chybiony.
Sortowanie kosztuje jedno zbedne klikniecie, odrzucanie kosztowaloby setki
dobrych par.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation.candidate_risk import fit, order_by_risk  # noqa: E402


def _anotacja(weak_pewnosc: float) -> dict:
    """Buduje anotację, w której część punktów jest niepewna."""
    punkty: list[float] = []
    for i in range(46):
        pewnosc = 1.0 if i / 46 > weak_pewnosc else 0.05
        punkty += [100.0 + i, 100.0 + (i % 7) * 5.0, pewnosc]
    return {"keypoints": punkty, "bbox": [0.0, 0.0, 500.0, 500.0]}


class TestUczenie:
    """Model musi rozpoznawać kierunek, mimo nierównych klas."""

    def test_wyrownuje_klasy(self) -> None:
        """
        Bez wyrównania model mówiłby „zostaw" na wszystko.

        Odrzuceń jest 20%, więc stała odpowiedź „zostaw" daje 80% trafności
        nie robiąc nic — i taki model jest bezużyteczny do sortowania.
        """
        x = np.hstack([np.array([[0.0]] * 80 + [[1.0]] * 20), np.ones((100, 1))])
        y = np.array([0] * 80 + [1] * 20)

        wagi = fit(x, y)

        assert wagi[0] > 0, "cecha rosnąca z odrzuceniem musi mieć dodatnią wagę"


class TestKolejnosc:
    """Nieprzejrzane idą przodem, przejrzane na koniec."""

    def _dane(self) -> tuple[list[dict], dict[str, bool], dict[str, dict]]:
        kandydaci = [{"peak": f"p{i}.jpg", "neutral": "n.jpg"} for i in range(80)]
        decyzje = {f"p{i}.jpg": i % 4 == 0 for i in range(40)}
        anotacje = {f"p{i}.jpg": _anotacja(0.9 if i % 4 == 0 else 0.1) for i in range(80)}
        return kandydaci, decyzje, anotacje

    def test_przejrzane_ida_na_koniec(self) -> None:
        kandydaci, decyzje, anotacje = self._dane()

        wynik = order_by_risk(kandydaci, decyzje, anotacje)

        ogon = [k["peak"] for k in wynik[-len(decyzje):]]
        assert set(ogon) == set(decyzje), "juz ocenione nie mają wracać na górę"

    def test_nieprzejrzane_dostaja_ryzyko(self) -> None:
        kandydaci, decyzje, anotacje = self._dane()

        wynik = order_by_risk(kandydaci, decyzje, anotacje)

        swieze = [k for k in wynik if "risk" in k]
        assert len(swieze) == 40
        assert swieze == sorted(swieze, key=lambda k: -k["risk"]), "malejąco po ryzyku"

    def test_zadna_para_nie_ginie(self) -> None:
        """Sortowanie ma przestawiać, nie usuwać — to tylko kolejność."""
        kandydaci, decyzje, anotacje = self._dane()

        wynik = order_by_risk(kandydaci, decyzje, anotacje)

        assert {k["peak"] for k in wynik} == {k["peak"] for k in kandydaci}
        assert len(wynik) == len(kandydaci)

    def test_bez_decyzji_kolejnosc_zostaje(self) -> None:
        """Przy garstce decyzji nie ma z czego się uczyć — nie udajemy, że jest."""
        kandydaci, _, anotacje = self._dane()

        wynik = order_by_risk(kandydaci, {"p0.jpg": True}, anotacje)

        assert wynik == kandydaci
