"""
Przerwany obieg musi DOKOŃCZYĆ falę, a nie zakładać nową.

Bez tego strata jest cicha i całkowita: nagrania przerobione przed przerwaniem
są już w `progress.json` części, więc następny obieg ich nie rozłoży — a ich
anotacje leżą w częściach, których nikt nigdy nie scali, bo scalanie bierze
tylko `annotations.json` świeżej fali. Zmierzone 28.08.2026: obieg padł po 210
z 306 nagrań i dokładnie tyle by przepadło.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.annotation.refresh_dataset as obieg  # noqa: E402


@pytest.fixture
def dane(tmp_path: Path, monkeypatch) -> Path:
    """Podstawia katalog danych na tymczasowy."""
    katalog = tmp_path / "data"
    katalog.mkdir()
    monkeypatch.setattr(obieg, "DATASET_DIR", katalog / "dataset_final")
    return katalog


def _fala(dane: Path, znacznik: str, *, scalona: bool, z_postepem: bool = True) -> Path:
    """
    Zakłada falę na dysku.

    Args:
        dane: Katalog danych
        znacznik: Znacznik czasu fali
        scalona: Czy fala ma gotowy `annotations.json` (czyli skończyła się)
        z_postepem: Czy części zdążyły cokolwiek zapisać

    Returns:
        Katalog wyniku fali
    """
    wyjscie = dane / f"dataset_{znacznik}"
    kolejka = dane / f"todo_{znacznik}"
    (kolejka / "neutral").mkdir(parents=True)
    (kolejka / "neutral" / "film.mp4").write_bytes(b"x")
    if z_postepem:
        czesc = wyjscie / "shard_0"
        czesc.mkdir(parents=True)
        (czesc / "progress.json").write_text(json.dumps({"processed_videos": ["film"]}))
    else:
        wyjscie.mkdir(parents=True)
    if scalona:
        (wyjscie / "annotations.json").write_text(json.dumps({"images": [], "annotations": []}))
    return wyjscie


class TestWznawianiePrzerwanejFali:
    """Fala bez scalonego wyniku, ale z postępem części, czeka na dokończenie."""

    def test_przerwana_fala_jest_znajdowana(self, dane: Path) -> None:
        _fala(dane, "20260828_0141", scalona=False)

        wynik = obieg._unfinished_wave()

        assert wynik is not None, "przerwana fala musi zostać znaleziona"
        kolejka, wyjscie = wynik
        assert kolejka.name == "todo_20260828_0141"
        assert wyjscie.name == "dataset_20260828_0141"

    def test_skonczona_fala_nie_jest_wznawiana(self, dane: Path) -> None:
        """Scalony `annotations.json` znaczy, że fala doszła do końca."""
        _fala(dane, "20260828_0141", scalona=True)

        assert obieg._unfinished_wave() is None

    def test_fala_bez_postepu_nie_liczy_sie(self, dane: Path) -> None:
        """Katalog zalozony, ale nic nie przerobione — nie ma czego ratować."""
        _fala(dane, "20260828_0141", scalona=False, z_postepem=False)

        assert obieg._unfinished_wave() is None

    def test_brak_katalogu_kolejki_nie_wywraca(self, dane: Path) -> None:
        """Kolejkę mógł ktoś sprzątnąć — wtedy nie ma z czego wznawiać."""
        wyjscie = _fala(dane, "20260828_0141", scalona=False)
        for plik in (wyjscie.parent / "todo_20260828_0141").rglob("*"):
            if plik.is_file():
                plik.unlink()
        (wyjscie.parent / "todo_20260828_0141" / "neutral").rmdir()
        (wyjscie.parent / "todo_20260828_0141").rmdir()

        assert obieg._unfinished_wave() is None

    def test_wybierana_jest_NAJNOWSZA_przerwana(self, dane: Path) -> None:
        _fala(dane, "20260827_1243", scalona=False)
        _fala(dane, "20260828_0141", scalona=False)

        kolejka, _ = obieg._unfinished_wave()

        assert kolejka.name == "todo_20260828_0141", "starsza fala nie może wyprzedzić nowszej"


class TestZalegleScalenia:
    """Fala skończona, ale niewlana, musi zostać wlana przy następnym obiegu."""

    def test_fala_bez_znacznika_czeka_na_scalenie(self, dane: Path) -> None:
        _fala(dane, "20260828_0141", scalona=True)

        czeka = obieg._unmerged_waves()

        assert [p.name for p in czeka] == ["dataset_20260828_0141"]

    def test_fala_ze_znacznikiem_nie_wraca(self, dane: Path) -> None:
        """Drugie wlanie tej samej fali zrobiłoby duplikaty w zbiorze."""
        wyjscie = _fala(dane, "20260828_0141", scalona=True)
        (wyjscie / obieg.MERGED_MARKER).write_text("wlane", encoding="utf-8")

        assert obieg._unmerged_waves() == []

    def test_niedokonczona_fala_nie_jest_scalana(self, dane: Path) -> None:
        """Bez `annotations.json` nie ma czego wlewać — najpierw trzeba dokończyć."""
        _fala(dane, "20260828_0141", scalona=False)

        assert obieg._unmerged_waves() == []

    def test_sam_zbior_nie_wlewa_sie_do_siebie(self, dane: Path) -> None:
        """`dataset_final` jest CELEM scalania, nie jego źródłem."""
        (dane / "dataset_final").mkdir(parents=True)
        (dane / "dataset_final" / "annotations.json").write_text(
            json.dumps({"images": [], "annotations": []})
        )

        assert obieg._unmerged_waves() == []


class TestPodzialPracy:
    """
    Przy podziale folderu miedzy dwie osoby obieg NIE MOZE pobierac z Dysku.

    `sync` sciaga CALY folder, wiec obieg uruchomiony z pobieraniem przyciagnie
    takze polowe kolegi — i obie osoby przerobia te same nagrania, kazda po
    kilkadziesiat godzin. Material dostarcza sie wtedy osobno
    (`download_by_manifest`), a obieg bierze to, co lezy na dysku.
    """

    def test_bez_pobierania_nie_dotyka_dysku(self, dane: Path, monkeypatch) -> None:
        wolane = []
        monkeypatch.setattr(obieg, "sync", lambda *a, **k: wolane.append("sync"))
        monkeypatch.setattr(obieg, "drive_has_new", lambda: wolane.append("lista") or 99)
        monkeypatch.setattr(obieg, "stage_new_videos", lambda wave: 0)
        monkeypatch.setattr(obieg, "_run", lambda *a, **k: None)
        monkeypatch.setattr(obieg, "publish", lambda *a, **k: False)
        monkeypatch.setattr(obieg, "count_pairs", lambda *a, **k: 0)
        monkeypatch.setattr(obieg, "orphaned_verdicts", lambda *a, **k: set())

        obieg.run_cycle(workers=1, allowed_orphans=2, push=False, download=False)

        assert wolane == [], "obieg siegnal po Dysk mimo --no-download"

    def test_z_pobieraniem_siega_po_dysk(self, dane: Path, monkeypatch) -> None:
        """Domyslne zachowanie zostaje nietkniete."""
        wolane = []
        from scripts.download.download_drive_folder import SyncStats
        monkeypatch.setattr(
            obieg, "sync", lambda *a, **k: wolane.append("sync") or SyncStats()
        )
        monkeypatch.setattr(obieg, "drive_has_new", lambda: 99)
        monkeypatch.setattr(obieg, "stage_new_videos", lambda wave: 0)
        monkeypatch.setattr(obieg, "_run", lambda *a, **k: None)
        monkeypatch.setattr(obieg, "publish", lambda *a, **k: False)
        monkeypatch.setattr(obieg, "count_pairs", lambda *a, **k: 0)
        monkeypatch.setattr(obieg, "orphaned_verdicts", lambda *a, **k: set())

        obieg.run_cycle(workers=1, allowed_orphans=2, push=False, download=True)

        assert wolane == ["sync"]
