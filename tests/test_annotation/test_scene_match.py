"""
Para musi pokazywać tego samego psa na obu klatkach.

Trek urywa się na cięciu montażowym, ale numer treku nie — w kompilacji ten sam
`track_id` biegnie przez kilka zwierząt. AU są różnicą względem klatki
neutralnej, więc przy innym psie wszystkie 21 pomiarów opisuje różnicę między
zwierzętami zamiast mimiki.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation.cropping import write_jpeg  # noqa: E402
from scripts.annotation.curate_for_review import keypoints_outside_bbox  # noqa: E402
from scripts.annotation.scene_match import face_similarity  # noqa: E402


def _punkty(x: float, y: float, rozmiar: float = 60.0) -> list[float]:
    """Buduje 46 punktów rozłożonych w kwadracie wokół (x, y)."""
    kropki: list[float] = []
    for i in range(46):
        kropki += [x + (i % 7) * rozmiar / 7, y + (i // 7) * rozmiar / 7, 1.0]
    return kropki


def _klatka(sciezka: Path, kolor: tuple[int, int, int]) -> None:
    """Zapisuje klatkę z jednolitym prostokątem 'mordy' w podanym kolorze."""
    obraz = np.full((200, 200, 3), 30, dtype=np.uint8)
    obraz[40:120, 40:120] = kolor
    write_jpeg(sciezka, obraz)


class TestPodobienstwoMordy:
    """Miara ma odróżniać to samo zwierzę od innego."""

    def test_ta_sama_morda_daje_wysokie_podobienstwo(self, tmp_path: Path) -> None:
        _klatka(tmp_path / "a.jpg", (40, 90, 200))
        _klatka(tmp_path / "b.jpg", (40, 90, 200))

        wynik = face_similarity(tmp_path, "a.jpg", _punkty(45, 45), "b.jpg", _punkty(45, 45))

        assert wynik is not None and wynik > 0.9

    def test_inna_morda_daje_niskie_podobienstwo(self, tmp_path: Path) -> None:
        _klatka(tmp_path / "a.jpg", (40, 90, 200))
        _klatka(tmp_path / "b.jpg", (200, 90, 40))

        wynik = face_similarity(tmp_path, "a.jpg", _punkty(45, 45), "b.jpg", _punkty(45, 45))

        assert wynik is not None and wynik < 0.4

    def test_brak_klatki_daje_None_a_nie_zero(self, tmp_path: Path) -> None:
        """
        None znaczy „nie wiem", a nie „różne psy".

        Potraktowanie nieudanego odczytu jako niepodobieństwa wyrzucałoby dobre
        pary za każdym razem, gdy plik chwilowo nie da się otworzyć.
        """
        _klatka(tmp_path / "a.jpg", (40, 90, 200))

        wynik = face_similarity(tmp_path, "a.jpg", _punkty(45, 45), "nie-ma.jpg", _punkty(45, 45))

        assert wynik is None

    def test_za_malo_pewnych_punktow_daje_None(self, tmp_path: Path) -> None:
        _klatka(tmp_path / "a.jpg", (40, 90, 200))
        _klatka(tmp_path / "b.jpg", (40, 90, 200))
        niepewne = [10.0, 10.0, 0.0] * 46

        wynik = face_similarity(tmp_path, "a.jpg", niepewne, "b.jpg", _punkty(45, 45))

        assert wynik is None


class TestPunktyPozaBoksem:
    """
    46 punktów rozdzielonych między dwa psy opisuje dwa zwierzęta i żadnego
    poprawnie. Boks psa pochodzi z detektora ciał — pomiaru NIEZALEŻNEGO od
    modelu punktów — więc nadaje się na sprawdzian.
    """

    def test_punkty_w_boksie_daja_zero(self) -> None:
        anotacja = {"bbox": [100.0, 100.0, 200.0, 200.0], "keypoints": _punkty(150, 150, 40)}

        assert keypoints_outside_bbox(anotacja) == 0.0

    def test_polowa_punktow_u_drugiego_psa(self) -> None:
        """Model rozdzielił punkty: część na swoim psie, część na sąsiednim."""
        swoje = _punkty(150, 150, 40)[: 23 * 3]
        cudze = _punkty(900, 150, 40)[: 23 * 3]
        anotacja = {"bbox": [100.0, 100.0, 200.0, 200.0], "keypoints": swoje + cudze}

        assert keypoints_outside_bbox(anotacja) > 0.4

    def test_brak_boksu_nie_wywraca(self) -> None:
        assert keypoints_outside_bbox({"keypoints": _punkty(10, 10)}) == 0.0

    def test_zdegenerowany_boks_nie_wywraca(self) -> None:
        anotacja = {"bbox": [10.0, 10.0, 0.0, 0.0], "keypoints": _punkty(10, 10)}

        assert keypoints_outside_bbox(anotacja) == 0.0
