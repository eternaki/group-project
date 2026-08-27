"""
Publikacja kolejki musi paść, gdy kolejka gubi cudze werdykty.

Sprawdzian istnieje, bo strata jest CICHA: dziennik w `data/labels/` przeżywa
nietknięty, tylko przestaje się wiązać z czymkolwiek. Zmierzone 27.08.2026 —
przebudowa od zera zabrała 603 z 605 werdyktów dwóch osób i wyszło to na jaw
wyłącznie przez ręczne porównanie liczb przed scaleniem.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation.queue_guard import (  # noqa: E402
    assert_no_new_orphans,
    orphaned_verdicts,
    verdict_keys,
)


def _kolejka(tmp_path: Path, klatki: list[str]) -> Path:
    """Zapisuje kolejkę z podanymi ścieżkami klatek."""
    sciezka = tmp_path / "curated.json"
    sciezka.write_text(
        json.dumps(
            {
                "images": [
                    {"id": i, "file_name": nazwa} for i, nazwa in enumerate(klatki, start=1)
                ],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    return sciezka


def _dziennik(tmp_path: Path, kto: str, klucze: list[str]) -> Path:
    """Zapisuje dziennik werdyktów jednej osoby."""
    katalog = tmp_path / "labels" / "nagranie"
    katalog.mkdir(parents=True, exist_ok=True)
    plik = katalog / f"{kto}.jsonl"
    plik.write_text(
        "\n".join(json.dumps({"pair_key": k, "annotator": kto}) for k in klucze),
        encoding="utf-8",
    )
    return tmp_path / "labels"


class TestOsieroconeWerdykty:
    """Werdykt bez pary w kolejce to praca, która przepadła."""

    def test_kolejka_z_kompletem_nie_gubi_nic(self, tmp_path: Path) -> None:
        kolejka = _kolejka(tmp_path, ["psA/peak.jpg", "psB/peak.jpg"])
        labels = _dziennik(tmp_path, "masha", ["psA/peak.jpg"])

        assert orphaned_verdicts(kolejka, labels) == set()

    def test_brakujaca_para_jest_wykrywana(self, tmp_path: Path) -> None:
        kolejka = _kolejka(tmp_path, ["psB/peak.jpg"])
        labels = _dziennik(tmp_path, "danek", ["psA/peak.jpg", "psB/peak.jpg"])

        assert orphaned_verdicts(kolejka, labels) == {"psA/peak.jpg"}

    def test_wpisy_testowe_sie_nie_licza(self, tmp_path: Path) -> None:
        """Klucze ze stanowiska na danych testowych nigdy nie miały pary."""
        kolejka = _kolejka(tmp_path, ["psA/peak.jpg"])
        labels = _dziennik(
            tmp_path, "anton", ["psA/peak.jpg", "/static/test1234/frame_0010.jpg"]
        )

        assert orphaned_verdicts(kolejka, labels) == set()
        assert "/static/test1234/frame_0010.jpg" not in verdict_keys(labels)

    def test_uciety_wiersz_nie_wywraca_odczytu(self, tmp_path: Path) -> None:
        """Dziennik dopisuje się wierszami — ostatni bywa ucięty."""
        kolejka = _kolejka(tmp_path, ["psA/peak.jpg"])
        labels = _dziennik(tmp_path, "masha", ["psA/peak.jpg"])
        (labels / "nagranie" / "masha.jsonl").write_text(
            json.dumps({"pair_key": "psA/peak.jpg"}) + '\n{"pair_key": "psB',
            encoding="utf-8",
        )

        assert orphaned_verdicts(kolejka, labels) == set()


class TestBramkaPublikacji:
    """Próg pilnuje PRZYROSTU strat, nie stanu zastanego."""

    def test_przepuszcza_straty_zastane(self, tmp_path: Path) -> None:
        kolejka = _kolejka(tmp_path, ["psB/peak.jpg"])
        labels = _dziennik(tmp_path, "danek", ["psA/peak.jpg", "psB/peak.jpg"])

        assert assert_no_new_orphans(kolejka, labels, allowed=1) == {"psA/peak.jpg"}

    def test_przerywa_gdy_strat_przybylo(self, tmp_path: Path) -> None:
        kolejka = _kolejka(tmp_path, [])
        labels = _dziennik(tmp_path, "danek", ["psA/peak.jpg", "psB/peak.jpg"])

        with pytest.raises(SystemExit) as blad:
            assert_no_new_orphans(kolejka, labels, allowed=1)
        assert "gubi 2 werdyktow" in str(blad.value)
