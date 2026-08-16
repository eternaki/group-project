"""
Testy podziału kolejki między czworo anotatorów.

Cztery osoby na jednym zbiorze muszą dostać ROZŁĄCZNE części, inaczej połowa
wysiłku idzie na to samo. Wyjątkiem jest blok wspólny na początku: bez ani
jednej pary ocenionej przez dwie osoby nie da się policzyć zgodności (kappa),
a bez tej liczby zbiór nie ma jak udowodnić, że etykiety są powtarzalne.
"""

import pytest
from annotators import SHARED_PREFIX_PAIRS, TEAM, owns_pair, shard_index

ALL_KEYS = [member.key for member in TEAM]


class TestTeam:
    """Skład zespołu."""

    def test_czworo_ludzi(self) -> None:
        assert len(TEAM) == 4

    def test_klucze_sa_unikalne(self) -> None:
        assert len(ALL_KEYS) == len(set(ALL_KEYS))

    def test_kazdy_ma_nazwe_do_pokazania(self) -> None:
        assert all(member.display for member in TEAM)

    def test_pozycja_wyznacza_czesc_kolejki(self) -> None:
        assert [shard_index(key) for key in ALL_KEYS] == [0, 1, 2, 3]

    def test_nieznana_osoba_dostaje_pierwsza_czesc(self) -> None:
        assert shard_index("ktos-obcy") == 0


class TestSharedPrefix:
    """Blok wspólny — materiał do policzenia zgodności."""

    def test_pierwsze_pary_dostaja_wszyscy(self) -> None:
        for order in range(SHARED_PREFIX_PAIRS):
            assert all(owns_pair(key, order) for key in ALL_KEYS)

    def test_blok_wspolny_jest_niepusty(self) -> None:
        """Bez niego kappa Cohena jest nie do policzenia."""
        assert SHARED_PREFIX_PAIRS > 0


class TestSplit:
    """Podział pozostałej kolejki."""

    @pytest.mark.parametrize("order", range(SHARED_PREFIX_PAIRS, SHARED_PREFIX_PAIRS + 40))
    def test_kazda_para_ma_dokladnie_jednego_wlasciciela(self, order: int) -> None:
        owners = [key for key in ALL_KEYS if owns_pair(key, order)]
        assert len(owners) == 1

    def test_podzial_jest_rownomierny(self) -> None:
        span = SHARED_PREFIX_PAIRS + 400
        counts = {
            key: sum(1 for order in range(SHARED_PREFIX_PAIRS, span) if owns_pair(key, order))
            for key in ALL_KEYS
        }
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_bierze_co_czwarta_pare_a_nie_kolejny_blok(self) -> None:
        """
        Kolejka jest posortowana od par najbardziej niepewnych, więc podział
        blokami dałby jednej osobie same trudne przypadki, a innej same łatwe.
        """
        first = ALL_KEYS[0]
        owned = [
            order
            for order in range(SHARED_PREFIX_PAIRS, SHARED_PREFIX_PAIRS + 20)
            if owns_pair(first, order)
        ]
        gaps = {b - a for a, b in zip(owned, owned[1:])}
        assert gaps == {len(TEAM)}
