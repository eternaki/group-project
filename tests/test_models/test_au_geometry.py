"""Testy modelu AU uczonego z geometrii twarzy."""

import numpy as np
import pytest

from packages.models.au_geometry import (
    VERDICT_ACTIVE,
    VERDICT_INACTIVE,
    VERDICT_NOT_OBSERVABLE,
    AUGeometryModel,
    fit_logistic,
    normalized_points,
    pair_features,
    train_action_unit,
)

LICZBA_PUNKTOW: int = 46


def _punkty(przesuniecie: float = 0.0, widocznosc: float = 1.0) -> list[float]:
    """Buduje płaską listę punktów rozrzuconych wokół zera."""
    krata = np.linspace(-10.0, 10.0, LICZBA_PUNKTOW)
    plaskie: list[float] = []
    for index, wartosc in enumerate(krata):
        plaskie += [float(wartosc), float(wartosc + przesuniecie * (index % 3)), widocznosc]
    return plaskie


def test_normalizacja_nie_zalezy_od_polozenia_ani_skali() -> None:
    """Ten sam pies bliżej i dalej od kamery daje te same znormalizowane punkty."""
    bazowe = np.asarray(_punkty()).reshape(-1, 3)
    przesuniete = bazowe.copy()
    przesuniete[:, :2] = przesuniete[:, :2] * 3.0 + np.array([500.0, -200.0])

    pierwsze = normalized_points(bazowe.ravel().tolist())
    drugie = normalized_points(przesuniete.ravel().tolist())

    assert pierwsze is not None and drugie is not None
    assert np.allclose(pierwsze, drugie)


def test_normalizacja_odmawia_przy_niewidocznych_punktach() -> None:
    """Klatka z większością punktów niewidocznych nie nadaje się do pomiaru."""
    assert normalized_points(_punkty(widocznosc=0.0)) is None


def test_normalizacja_odmawia_przy_pustych_punktach() -> None:
    """Brak punktów to brak pomiaru, a nie wektor zer."""
    assert normalized_points([]) is None


def test_cechy_zmieniaja_znak_przy_zamianie_rol() -> None:
    """Podanie klatek odwrotnie odwraca przesunięcie — dlatego kolejność jest znacząca."""
    wyraz, baza = _punkty(przesuniecie=2.0), _punkty()

    proste = pair_features(wyraz, baza)
    odwrotne = pair_features(baza, wyraz)

    assert proste is not None and odwrotne is not None
    polowa = len(proste) // 2
    assert np.allclose(proste[:polowa], -odwrotne[:polowa])


def test_cechy_bez_jednej_klatki_to_brak_wiedzy() -> None:
    """Nieczytelna klatka bazowa unieważnia całą parę."""
    assert pair_features(_punkty(), _punkty(widocznosc=0.0)) is None


def test_regresja_wyrownuje_klasy() -> None:
    """Przy 2% pozytywów model bez wyrównania milczałby — z wyrównaniem uczy się."""
    rng = np.random.default_rng(0)
    cechy = rng.normal(size=(200, 3))
    etykiety = np.zeros(200, dtype=int)
    etykiety[:4] = 1
    cechy[:4, 0] += 5.0

    wspolczynniki = fit_logistic(cechy, etykiety)
    oceny = np.hstack([cechy, np.ones((200, 1))]) @ wspolczynniki

    assert oceny[:4].mean() > oceny[4:].mean()


def test_model_uczy_sie_rozdzielac_pary() -> None:
    """Na rozdzielalnych danych wyuczone AU orzeka aktywację tam, gdzie trzeba."""
    rng = np.random.default_rng(1)
    cechy = rng.normal(scale=0.1, size=(60, 5))
    etykiety = np.zeros(60, dtype=int)
    etykiety[:20] = 1
    cechy[:20, 0] += 3.0

    wagi = train_action_unit(cechy, etykiety)
    orzeczenia = np.array([wagi.score(wiersz) >= wagi.threshold for wiersz in cechy])

    assert orzeczenia[:20].sum() > orzeczenia[20:].sum()


def test_brak_aktywacji_daje_prog_nieskonczony() -> None:
    """AU, którego nikt nigdy nie potwierdził, nie może orzekać aktywacji."""
    cechy = np.random.default_rng(2).normal(size=(30, 4))

    wagi = train_action_unit(cechy, np.zeros(30, dtype=int))

    assert wagi.threshold == float("inf")
    assert not any(wagi.score(wiersz) >= wagi.threshold for wiersz in cechy)


def test_brak_cech_znaczy_nie_wiadomo_a_nie_spoczynek() -> None:
    """Nieczytelna para dostaje `not_observable` na wszystkich AU."""
    wagi = train_action_unit(
        np.random.default_rng(3).normal(size=(30, 4)), np.array([1] * 10 + [0] * 20)
    )
    model = AUGeometryModel({"AU25": wagi})

    assert model.predict(None) == {"AU25": VERDICT_NOT_OBSERVABLE}


def test_model_orzeka_tylko_o_au_ktore_umie() -> None:
    """AU bez wyuczonych współczynników nie pojawia się w wyniku wcale."""
    wagi = train_action_unit(
        np.random.default_rng(4).normal(size=(30, 4)), np.array([1] * 10 + [0] * 20)
    )
    model = AUGeometryModel({"AU25": wagi})

    wynik = model.predict(np.zeros(4))

    assert set(wynik) == {"AU25"}
    assert wynik["AU25"] in (VERDICT_ACTIVE, VERDICT_INACTIVE)


def test_zapis_i_odczyt_zachowuja_orzeczenia() -> None:
    """Model po przejściu przez JSON orzeka dokładnie to samo."""
    rng = np.random.default_rng(5)
    cechy = rng.normal(size=(40, 6))
    etykiety = np.array([1] * 12 + [0] * 28)
    model = AUGeometryModel({"AU25": train_action_unit(cechy, etykiety)})

    odtworzony = AUGeometryModel.from_dict(model.to_dict())

    for wiersz in cechy:
        assert model.predict(wiersz) == odtworzony.predict(wiersz)


def test_zapis_jest_liczbami_a_nie_tablicami() -> None:
    """`to_dict` musi dać się zserializować do JSON-a bez konwerterów."""
    import json

    wagi = train_action_unit(
        np.random.default_rng(6).normal(size=(30, 4)), np.array([1] * 10 + [0] * 20)
    )

    zapis = json.dumps(AUGeometryModel({"AU25": wagi}).to_dict())

    assert json.loads(zapis)["AU25"]["threshold"] == pytest.approx(wagi.threshold)
