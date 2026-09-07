"""Testy oceny reguł AU względem werdyktu człowieka."""

import json
from pathlib import Path

import pytest

from scripts.annotation.eval_au_rules import (
    FOLDY,
    Komorka,
    Wynik,
    auc,
    podziel_nagrania,
    prog_na_precyzje,
    raport_gejtu,
    raport_regul_surowych,
    sprawdzian_krzyzowy,
    wczytaj_komorki,
)


def _release(tmp_path: Path, anotacje: list[dict], obrazy: list[dict]) -> Path:
    """Zapisuje minimalny plik COCO i zwraca jego ścieżkę."""
    sciezka = tmp_path / "annotations.json"
    sciezka.write_text(
        json.dumps({"images": obrazy, "annotations": anotacje}, ensure_ascii=False),
        encoding="utf-8",
    )
    return sciezka


def test_not_observable_nie_liczy_sie_ani_w_jedna_ani_w_druga_strone(tmp_path: Path) -> None:
    """`not_observable` to brak wiedzy — nie jest ani trafieniem, ani pomyłką."""
    anotacje = [
        {
            "image_id": 1,
            "au_verdicts": {"AU25": "not_observable", "AU26": "active"},
            "au_analysis": {
                "AU25": {"ratio": 1.9, "is_active": True},
                "AU26": {"ratio": 1.9, "is_active": True},
            },
        }
    ]
    obrazy = [{"id": 1, "source_video": "a.mp4"}]
    komorki = wczytaj_komorki(_release(tmp_path, anotacje, obrazy))

    assert "AU25" not in komorki
    assert len(komorki["AU26"]) == 1

    wynik = raport_regul_surowych(_release(tmp_path, anotacje, obrazy))
    assert (wynik.trafione, wynik.falszywe, wynik.przeoczone) == (1, 0, 0)


def test_odchylenie_liczy_sie_w_obie_strony(tmp_path: Path) -> None:
    """AU aktywujące się przez ZMNIEJSZENIE odległości ma dodatnie odchylenie."""
    anotacje = [
        {
            "image_id": 1,
            "au_verdicts": {"AD19": "active"},
            "au_analysis": {"AD19": {"ratio": 0.7, "is_active": True}},
        }
    ]
    komorki = wczytaj_komorki(_release(tmp_path, anotacje, [{"id": 1, "source_video": "a.mp4"}]))

    assert komorki["AD19"][0].odchylenie == pytest.approx(0.3)


def test_gejt_pomija_au_ktorego_nie_orzekl(tmp_path: Path) -> None:
    """AU nieobecne w `au_auto_verdict` nie wchodzi do zliczeń gejtu."""
    anotacje = [
        {
            "image_id": 1,
            "au_verdicts": {"AU25": "active", "AU26": "active"},
            "au_auto_verdict": {"AU25": "active"},
        }
    ]
    wynik = raport_gejtu(_release(tmp_path, anotacje, [{"id": 1}]))

    assert (wynik.trafione, wynik.falszywe, wynik.przeoczone) == (1, 0, 0)


def test_nagranie_nie_rozpada_sie_miedzy_foldy() -> None:
    """Całe nagranie leży po jednej stronie podziału — inaczej próg mierzy pamięć."""
    komorki = {
        "AU25": [Komorka(f"wideo_{i}.mp4", 0.1 * i, i % 2 == 0) for i in range(20)],
    }
    przydzial = podziel_nagrania(komorki)

    assert len(przydzial) == 20
    assert set(przydzial.values()) <= set(range(FOLDY))


def test_podzial_jest_powtarzalny() -> None:
    """Ten sam materiał daje ten sam podział przy każdym uruchomieniu."""
    komorki = {"AU25": [Komorka(f"w{i}.mp4", 0.1, False) for i in range(30)]}

    assert podziel_nagrania(komorki) == podziel_nagrania(komorki)


def test_prog_na_precyzje_nie_lapie_sie_na_pojedyncze_trafienie() -> None:
    """Jedno trafienie na jedno orzeczenie to nie jest precyzja 100%."""
    uczace = [Komorka("a.mp4", 0.9, True)] + [Komorka("b.mp4", 0.1, False) for _ in range(20)]

    assert prog_na_precyzje(uczace, cel=0.9) is None


def test_sprawdzian_bez_aktywacji_w_uczacym_nie_orzeka() -> None:
    """Gdy w części uczącej nie ma ani jednej aktywacji, wariant milczy."""
    komorki = [Komorka(f"w{i}.mp4", 0.5, False) for i in range(10)]
    przydzial = podziel_nagrania({"AU25": komorki})

    wynik = sprawdzian_krzyzowy(komorki, przydzial, lambda _: None)

    assert wynik.trafione == 0 and wynik.falszywe == 0


def test_auc_rozpoznaje_pomiar_bez_informacji() -> None:
    """Identyczne wartości w obu klasach dają AUC 0.5, czyli zero informacji."""
    komorki = [Komorka("a.mp4", 0.4, i % 2 == 0) for i in range(10)]

    assert auc(komorki) == 0.5


def test_auc_rozpoznaje_pomiar_idealny() -> None:
    """Rozdzielone klasy dają AUC 1.0."""
    komorki = [Komorka("a.mp4", 0.9, True), Komorka("a.mp4", 0.1, False)]

    assert auc(komorki) == 1.0


def test_auc_bez_jednej_klasy_zwraca_brak_wiedzy() -> None:
    """Bez aktywacji nie ma czego mierzyć — None, a nie 0.0."""
    assert auc([Komorka("a.mp4", 0.5, False)]) is None


def test_wynik_sumuje_sie_miedzy_au() -> None:
    """Zliczenia różnych AU dodają się bez gubienia pomyłek."""
    pierwszy, drugi = Wynik(), Wynik()
    pierwszy.dodaj(True, True)
    drugi.dodaj(True, False)
    drugi.dodaj(False, True)
    pierwszy += drugi

    assert (pierwszy.trafione, pierwszy.falszywe, pierwszy.przeoczone) == (1, 1, 1)
    assert pierwszy.precyzja == 0.5
    assert pierwszy.pokrycie == 0.5
