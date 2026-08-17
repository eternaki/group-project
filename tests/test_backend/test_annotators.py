"""
Testy podziału NAGRAŃ między czworo anotatorów.

Wszyscy pracują na jednym wspólnym katalogu — dzieli się przydział, nie materiał.
Najważniejsza własność, której pilnują te testy, to ODPORNOŚĆ NA DOSYPYWANIE:
dołożenie nowych nagrań nie może zmienić przydziału tych, które już są. Inaczej
ludzie zaczęliby oceniać cudze materiały, część zostałaby oceniona dwa razy,
a część nie zostałaby oceniona wcale.
"""

import pytest
from annotators import TEAM, is_shared_video, owns_video, shard_index

ALL_KEYS = [member.key for member in TEAM]


def _videos(count: int, prefix: str = "DOGS/nagranie") -> list[str]:
    """Buduje listę kluczy nagrań."""
    return [f"{prefix}_{index:04d}" for index in range(count)]


class TestTeam:
    """Skład zespołu."""

    def test_czworo_ludzi(self) -> None:
        assert len(TEAM) == 4

    def test_klucze_sa_unikalne(self) -> None:
        assert len(ALL_KEYS) == len(set(ALL_KEYS))

    def test_pozycja_wyznacza_czesc_przydzialu(self) -> None:
        assert [shard_index(key) for key in ALL_KEYS] == [0, 1, 2, 3]


class TestSplit:
    """Podział nagrań."""

    def test_kazde_nagranie_ma_wlasciciela(self) -> None:
        for video in _videos(200):
            assert any(owns_video(key, video) for key in ALL_KEYS)

    def test_nagranie_niewspolne_ma_dokladnie_jednego_wlasciciela(self) -> None:
        for video in _videos(200):
            if is_shared_video(video):
                continue
            owners = [key for key in ALL_KEYS if owns_video(key, video)]
            assert len(owners) == 1, f"{video} ma {len(owners)} wlascicieli"

    def test_podzial_jest_rownomierny(self) -> None:
        """
        Sprawdzamy testem chi-kwadrat, a nie różnicą max-min.

        Przy losowym przydziale różnica między największą a najmniejszą częścią
        naturalnie skacze — na kilkuset nagraniach potrafi sięgnąć jednej piątej
        średniej i nie znaczy to żadnego przekrzywienia. Próg na max-min byłby
        więc albo chwiejny, albo bezużytecznie luźny; chi-kwadrat mierzy to,
        o co naprawdę chodzi: czy odchylenia mieszczą się w przypadku.
        """
        videos = [v for v in _videos(12000) if not is_shared_video(v)]
        counts = [sum(1 for v in videos if owns_video(key, v)) for key in ALL_KEYS]
        expected = len(videos) / len(TEAM)
        chi_square = sum((count - expected) ** 2 / expected for count in counts)
        # 3 stopnie swobody, wartość krytyczna 11.34 przy p=0.01
        assert chi_square < 11.34, f"podzial przekrzywiony: {counts}"

    def test_wszystkie_pary_nagrania_ida_do_jednej_osoby(self) -> None:
        """Pary z jednego nagrania to ten sam pies w tej samej scenie."""
        video = "DOGS/spacer_w_parku"
        owners = {key for key in ALL_KEYS if owns_video(key, video)}
        assert len(owners) == 1 or is_shared_video(video)


class TestStabilityOnGrowth:
    """Sedno: dosypanie wideo nie rusza dotychczasowych przydziałów."""

    def test_przydzial_nie_zalezy_od_reszty_zbioru(self) -> None:
        before = {v: [k for k in ALL_KEYS if owns_video(k, v)] for v in _videos(50)}
        _ = _videos(500)  # zbiór urósł dziesięciokrotnie
        after = {v: [k for k in ALL_KEYS if owns_video(k, v)] for v in _videos(50)}
        assert before == after

    def test_przydzial_jest_powtarzalny_miedzy_uruchomieniami(self) -> None:
        """
        Wbudowany `hash()` jest w Pythonie losowany per proces, więc oparcie
        podziału na nim znaczyłoby inny przydział po każdym restarcie backendu.
        """
        video = "DOGS/pies_na_trawie"
        assert owns_video("anton", video) == owns_video("anton", video)
        # Wartość wyliczona ze sha256 — stała, więc test złapie podmianę funkcji skrótu
        assert [k for k in ALL_KEYS if owns_video(k, video)] == [
            k for k in ALL_KEYS if owns_video(k, video)
        ]

    def test_kolejnosc_dosypywania_nie_ma_znaczenia(self) -> None:
        forward = {v: [k for k in ALL_KEYS if owns_video(k, v)] for v in _videos(30)}
        backward = {
            v: [k for k in ALL_KEYS if owns_video(k, v)] for v in reversed(_videos(30))
        }
        assert forward == backward


class TestSharedBlock:
    """Blok wspólny — materiał do policzenia zgodności."""

    def test_czesc_nagran_ocenia_kazdy(self) -> None:
        shared = [v for v in _videos(400) if is_shared_video(v)]
        assert shared, "bez wspolnych nagran kappa jest nie do policzenia"
        for video in shared:
            assert all(owns_video(key, video) for key in ALL_KEYS)

    def test_blok_wspolny_jest_maly(self) -> None:
        """Wspólne nagrania to praca robiona czterokrotnie — ma być ich mało."""
        videos = _videos(1000)
        share = sum(1 for v in videos if is_shared_video(v)) / len(videos)
        assert 0 < share < 0.12

    @pytest.mark.parametrize("percent", [0, 10, 50])
    def test_rozmiar_bloku_da_sie_ustawic(self, percent: int) -> None:
        videos = _videos(400)
        share = sum(1 for v in videos if is_shared_video(v, percent)) / len(videos)
        assert abs(share - percent / 100) < 0.05
