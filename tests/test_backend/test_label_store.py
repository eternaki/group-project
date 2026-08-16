"""
Testy magazynu etykiet — postaci, w jakiej praca anotatorów jeździ w gicie.

Sesja anotacji jest lokalna dla maszyny, więc sama nie przenosi niczego do
zespołu. Te testy pilnują trzech własności, bez których współpraca się sypie:
dopisywanie zamiast nadpisywania (git scala dodane linie bez konfliktu),
zachowanie historii (rozbieżność między anotatorami to materiał na kappę)
oraz odporność na uszkodzoną linię (utrata jednej decyzji jest do przeżycia,
utrata pracy zespołu nie).
"""

from pathlib import Path

import pytest
from label_store import (
    ANNOTATOR_ENV,
    LABELS_ROOT_ENV,
    LabelRecord,
    append_label,
    build_record,
    current_annotator,
    labels_path,
    latest_by_pair,
    read_all,
)

DATASET = "dataset_v2"
PAIR = "DOGS/spacer/spacer_t0_000120.jpg"


@pytest.fixture(autouse=True)
def labels_root(tmp_path: Path, monkeypatch) -> Path:
    """Katalog etykiet w tmp — testy nie mogą pisać do repozytorium."""
    monkeypatch.setenv(LABELS_ROOT_ENV, str(tmp_path / "labels"))
    monkeypatch.setenv(ANNOTATOR_ENV, "anna")
    return tmp_path / "labels"


def _record(pair: str = PAIR, annotator: str = "anna", stamp: str = "2026-08-16T10:00:00+00:00", **kwargs) -> LabelRecord:
    """Buduje rekord etykiety o zadanym autorze i czasie."""
    return LabelRecord(
        pair_key=pair,
        annotator=annotator,
        timestamp=stamp,
        au_verdicts=kwargs.get("au_verdicts", {"AU25": "active"}),
        usable=kwargs.get("usable", True),
        keypoints_ok=kwargs.get("keypoints_ok", True),
        breed=kwargs.get("breed", "Beagle"),
        emotion=kwargs.get("emotion", "happy"),
    )


class TestAppend:
    """Dopisywanie decyzji."""

    def test_kazdy_anotator_pisze_do_wlasnego_pliku(self, labels_root: Path) -> None:
        """Wspólny plik dawałby konflikt gita przy każdym pchnięciu."""
        append_label(DATASET, _record(annotator="anna"))
        append_label(DATASET, _record(annotator="bartek"))
        names = sorted(p.name for p in (labels_root / DATASET).glob("*.jsonl"))
        assert names == ["anna.jsonl", "bartek.jsonl"]

    def test_druga_ocena_dopisuje_linie_zamiast_nadpisywac(self, labels_root: Path) -> None:
        """Historia ocen jest materiałem do policzenia zgodności."""
        append_label(DATASET, _record(stamp="2026-08-16T10:00:00+00:00"))
        append_label(DATASET, _record(stamp="2026-08-16T11:00:00+00:00", breed="Border Collie"))
        lines = labels_path(DATASET, "anna").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_zapis_tworzy_brakujace_katalogi(self, labels_root: Path) -> None:
        append_label("nowy_zbior", _record())
        assert labels_path("nowy_zbior", "anna").is_file()


class TestRead:
    """Odczyt etykiet całego zespołu."""

    def test_czyta_wszystkich_anotatorow(self) -> None:
        append_label(DATASET, _record(annotator="anna"))
        append_label(DATASET, _record(annotator="bartek", pair="inna.jpg"))
        assert {record.annotator for record in read_all(DATASET)} == {"anna", "bartek"}

    def test_brak_katalogu_daje_pusto_zamiast_bledu(self) -> None:
        assert read_all("zbior_ktorego_nie_ma") == []

    def test_uszkodzona_linia_nie_wywraca_odczytu(self, labels_root: Path) -> None:
        """Utrata jednej decyzji jest do przeżycia, utrata pracy zespołu nie."""
        append_label(DATASET, _record())
        with open(labels_path(DATASET, "anna"), "a", encoding="utf-8") as handle:
            handle.write("{to nie jest json\n")
        append_label(DATASET, _record(pair="druga.jpg"))
        assert len(read_all(DATASET)) == 2

    def test_najswiezsza_ocena_wygrywa(self) -> None:
        append_label(DATASET, _record(stamp="2026-08-16T10:00:00+00:00", breed="Beagle"))
        append_label(DATASET, _record(stamp="2026-08-16T12:00:00+00:00", breed="Border Collie"))
        assert latest_by_pair(DATASET)[PAIR].breed == "Border Collie"

    def test_ocena_kolegi_widoczna_po_dociagnieciu(self) -> None:
        """Tak wygląda `git pull` z werdyktami drugiej osoby."""
        append_label(DATASET, _record(annotator="anna", stamp="2026-08-16T10:00:00+00:00"))
        append_label(
            DATASET,
            _record(annotator="bartek", stamp="2026-08-16T13:00:00+00:00", emotion="sad"),
        )
        assert latest_by_pair(DATASET)[PAIR].emotion == "sad"


class TestAnnotatorName:
    """Nazwa anotatora trafia do nazwy pliku, więc musi być bezpieczna."""

    def test_bierze_nazwe_ze_zmiennej(self) -> None:
        assert current_annotator() == "anna"

    def test_spacje_i_ukosniki_nie_wysadzaja_zapisu(self, monkeypatch) -> None:
        monkeypatch.setenv(ANNOTATOR_ENV, "Jan Kowalski/WETI")
        append_label(DATASET, build_record(PAIR, {}, True, None, None, None))
        assert labels_path(DATASET).is_file()

    def test_pusta_nazwa_daje_wartosc_zastepcza(self, monkeypatch) -> None:
        monkeypatch.setenv(ANNOTATOR_ENV, "///")
        assert current_annotator() == "anonim"


class TestBuildRecord:
    """Składanie rekordu z bieżącym anotatorem i czasem."""

    def test_wpisuje_biezacego_anotatora(self) -> None:
        record = build_record(PAIR, {"AU26": "inactive"}, True, False, "Pug", "neutral")
        assert record.annotator == "anna"
        assert record.keypoints_ok is False

    def test_znacznik_czasu_jest_sortowalny(self) -> None:
        first = build_record(PAIR, {}, True, None, None, None)
        second = build_record(PAIR, {}, True, None, None, None)
        assert first.timestamp <= second.timestamp


class TestAnnotatorFromInterface:
    """
    Kto ocenia, przychodzi z INTERFEJSU, nie z konta systemowego.

    Przy jednym komputerze dzielonym przez zespół nazwa konta Windows wpisałaby
    werdykty wszystkich do pliku właściciela maszyny — i praca trzech osób
    zniknęłaby w cudzym pliku bez żadnego komunikatu.
    """

    def test_jawny_anotator_wygrywa_z_kontem_systemowym(self, monkeypatch) -> None:
        monkeypatch.setenv(ANNOTATOR_ENV, "wlasciciel_maszyny")
        record = build_record(PAIR, {}, True, None, None, None, annotator="masha")
        assert record.annotator == "masha"

    def test_brak_jawnego_bierze_konto(self, monkeypatch) -> None:
        monkeypatch.setenv(ANNOTATOR_ENV, "anna")
        assert build_record(PAIR, {}, True, None, None, None).annotator == "anna"

    def test_kazda_osoba_pisze_do_swojego_pliku(self, labels_root: Path) -> None:
        for who in ("anton", "masha", "mafin", "danek"):
            append_label(DATASET, build_record(f"{who}.jpg", {}, True, None, None, None, annotator=who))
        names = sorted(p.stem for p in (labels_root / DATASET).glob("*.jsonl"))
        assert names == ["anton", "danek", "mafin", "masha"]
