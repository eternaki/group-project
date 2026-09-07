"""Testy uczenia i sprawdzania modelu AU."""

import numpy as np

from scripts.annotation.train_au_model import (
    FOLDS,
    Score,
    TrainingSet,
    assign_folds,
    cross_validate,
    fit_model,
)


def _zbior(pary: int = 60, aktywne: int = 20, odwrotne: bool = False) -> TrainingSet:
    """
    Buduje zbiór, w którym pierwsza cecha rozdziela klasy.

    Args:
        pary: Liczba par
        aktywne: Ile z nich człowiek uznał za aktywne
        odwrotne: Czy wariant „role z pipeline'u" ma mieć odwrócony znak

    Returns:
        Gotowy zbiór uczący
    """
    rng = np.random.default_rng(7)
    zbior = TrainingSet()
    for index in range(pary):
        cechy = rng.normal(scale=0.1, size=4)
        czy_aktywne = index < aktywne
        if czy_aktywne:
            cechy[0] += 3.0
        pipeline = -cechy if odwrotne else cechy
        zbior.add(
            f"wideo_{index}.mp4",
            cechy,
            pipeline,
            {"AU25": "active" if czy_aktywne else "inactive"},
        )
    return zbior


def test_podzial_trzyma_nagranie_w_jednej_czesci() -> None:
    """Nagranie nie może trafić do dwóch części — inaczej model mierzy pamięć."""
    przydzial = assign_folds([f"w{i}.mp4" for i in range(20)])

    assert len(przydzial) == 20
    assert set(przydzial.values()) <= set(range(FOLDS))


def test_podzial_jest_powtarzalny_miedzy_uruchomieniami() -> None:
    """`hash()` jest solony na proces, więc podział musi iść z własnego skrótu."""
    nagrania = [f"w{i}.mp4" for i in range(30)]

    assert assign_folds(nagrania) == assign_folds(list(reversed(nagrania)))


def test_au_bez_potwierdzen_nie_dostaje_modelu() -> None:
    """Bez dostatecznej liczby aktywacji uczciwiej nie orzekać wcale."""
    zbior = _zbior(aktywne=2)

    model = fit_model(zbior)

    assert "AU25" not in model.per_unit


def test_au_z_potwierdzeniami_dostaje_model() -> None:
    """Przy dość licznych potwierdzeniach AU wchodzi do modelu."""
    model = fit_model(_zbior())

    assert "AU25" in model.per_unit


def test_sprawdzian_znajduje_sygnal_gdy_jest() -> None:
    """Na rozdzielalnych danych sprawdzian krzyżowy pokazuje trafienia."""
    zbior = _zbior()

    wynik = cross_validate(zbior, "AU25", assign_folds(zbior.videos))

    assert wynik is not None
    assert wynik.hits > 0
    assert wynik.precision > 0.5


def test_sprawdzian_na_rolach_pipeline_widzi_pogorszenie() -> None:
    """Zamienione role psują wynik — po to mierzy się je osobno."""
    zbior = _zbior(odwrotne=True)
    przydzial = assign_folds(zbior.videos)

    poprawne = cross_validate(zbior, "AU25", przydzial)
    pipeline = cross_validate(zbior, "AU25", przydzial, pipeline_roles=True)

    assert poprawne is not None and pipeline is not None
    assert pipeline.hits < poprawne.hits


def test_sprawdzian_milczy_przy_zbyt_malej_liczbie_potwierdzen() -> None:
    """Poniżej progu potwierdzeń nie ma czego mierzyć — None, a nie zero."""
    zbior = _zbior(aktywne=3)

    assert cross_validate(zbior, "AU25", assign_folds(zbior.videos)) is None


def test_zliczenia_licza_precyzje_i_pokrycie_osobno() -> None:
    """Pomyłka na plus i przeoczenie to różne błędy i nie wolno ich zlewać."""
    wynik = Score()
    wynik.add(True, True)
    wynik.add(True, False)
    wynik.add(False, True)
    wynik.add(False, False)

    assert wynik.precision == 0.5
    assert wynik.recall == 0.5
    assert wynik.f1 == 0.5


def test_puste_zliczenia_nie_dziela_przez_zero() -> None:
    """Model, który nic nie orzekł, ma precyzję zero, a nie wyjątek."""
    wynik = Score()

    assert wynik.precision == 0.0
    assert wynik.recall == 0.0
    assert wynik.f1 == 0.0
