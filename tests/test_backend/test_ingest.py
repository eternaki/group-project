"""
Testy dosypywania nagrań.

Nazwa pliku przychodzi z przeglądarki i wchodzi zarówno do ścieżek klatek, jak
i do skrótu wyznaczającego właściciela nagrania — dlatego jest tu sprawdzana
najdokładniej. Osobno pilnujemy blokady: plik z PID zostaje po nagłym zamknięciu
i bez weryfikacji, czy proces żyje, zablokowałby dosypywanie na zawsze.
"""

from pathlib import Path

import pytest
from ingest import (
    INBOX_DATASET_ENV,
    VIDEOS_ROOT_ENV,
    IngestError,
    is_running,
    list_videos,
    pairs_ready,
    processed_count,
    safe_video_name,
    save_video,
    status,
)


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch) -> Path:
    """Katalogi nagrań i zbioru w tmp — testy nie mogą pisać do repozytorium."""
    monkeypatch.setenv(VIDEOS_ROOT_ENV, str(tmp_path / "videos"))
    monkeypatch.setenv(INBOX_DATASET_ENV, str(tmp_path / "dataset"))
    return tmp_path


class TestSafeName:
    """Sprowadzanie nazwy pliku do postaci bezpiecznej."""

    def test_zwykla_nazwa_przechodzi(self) -> None:
        assert safe_video_name("piesek.mp4") == "piesek.mp4"

    def test_spacje_zostaja(self) -> None:
        """Spacje są nieszkodliwe, a ich usuwanie zmieniłoby właściciela nagrania."""
        assert safe_video_name("pies w parku.mp4") == "pies w parku.mp4"

    def test_sciezka_zostaje_obcieta(self) -> None:
        """Bez tego plik trafiłby poza katalog nagrań."""
        assert safe_video_name("../../../etc/pies.mp4") == "pies.mp4"

    def test_znaki_niedozwolone_podmienione(self) -> None:
        assert "?" not in safe_video_name("pies?.mp4")

    def test_zly_format_odrzucony(self) -> None:
        with pytest.raises(IngestError, match="format"):
            safe_video_name("zdjecie.png")

    def test_pusta_nazwa_odrzucona(self) -> None:
        with pytest.raises(IngestError):
            safe_video_name("   ")


class TestSaveVideo:
    """Zapis nagrania do wspólnego katalogu."""

    def test_zapisuje_do_wspolnego_katalogu(self, workspace: Path) -> None:
        path = save_video("pies.mp4", b"dane")
        assert path.is_file()
        assert path.read_bytes() == b"dane"

    def test_nagranie_widac_na_liscie(self) -> None:
        save_video("pies.mp4", b"dane")
        assert [p.name for p in list_videos()] == ["pies.mp4"]

    def test_ponowne_wgranie_nadpisuje(self) -> None:
        """Ta sama nazwa to to samo nagranie — duplikat byłby drugim właścicielem."""
        save_video("pies.mp4", b"stare")
        save_video("pies.mp4", b"nowe")
        assert len(list_videos()) == 1
        assert list_videos()[0].read_bytes() == b"nowe"


class TestStatus:
    """Stan dosypywania."""

    def test_pusty_katalog_daje_zera(self) -> None:
        current = status()
        assert current.videos_total == 0
        assert current.videos_processed == 0
        assert current.running is False

    def test_liczy_wgrane_nagrania(self) -> None:
        save_video("a.mp4", b"x")
        save_video("b.mp4", b"x")
        assert status().videos_total == 2

    def test_brak_postepu_nie_wysypuje_odczytu(self) -> None:
        assert processed_count() == 0

    def test_brak_kuracji_nie_wysypuje_odczytu(self) -> None:
        assert pairs_ready() == 0

    def test_uszkodzony_postep_czytany_jako_zero(self, workspace: Path) -> None:
        dataset = workspace / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "progress.json").write_text("{zepsute", encoding="utf-8")
        assert processed_count() == 0


class TestLock:
    """Blokada procesu przetwarzania."""

    def test_brak_blokady_znaczy_nie_pracuje(self) -> None:
        assert is_running() is False

    def test_martwy_pid_nie_blokuje_na_zawsze(self, workspace: Path) -> None:
        """
        Plik blokady zostaje po nagłym zamknięciu maszyny. Gdyby sama jego
        obecność znaczyła „pracuje", dosypywanie byłoby zablokowane na stałe
        i nikt by nie zgadł dlaczego.
        """
        dataset = workspace / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "worker.pid").write_text("999999", encoding="utf-8")
        assert is_running() is False

    def test_smiec_w_pliku_blokady_nie_wywraca_odczytu(self, workspace: Path) -> None:
        dataset = workspace / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "worker.pid").write_text("to nie jest liczba", encoding="utf-8")
        assert is_running() is False
