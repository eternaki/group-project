"""
Testy odstępu próbkowania klatek w anotacji wsadowej.

Sedno: budżet klatek (`max_frames_per_video`) nie może URYWAĆ nagrania.
Próbkowanie w zamówionym tempie z twardym limitem wyczerpuje budżet na
początku klipu, więc do zbioru trafiają systematycznie POCZĄTKI ujęć —
a moment, w którym pies patrzy w obiektyw, wypada gdzie indziej równie często.
Gdy budżet nie starcza, odstęp ma się rozciągnąć na całe nagranie.
"""

import pytest

from scripts.annotation.batch_annotate import BatchAnnotator, BatchConfig

# Typowe nagranie stockowe: 20 s przy 30 kl./s
VIDEO_FPS: float = 30.0
TOTAL_FRAMES_20S: int = 600


def _annotator(fps: float, max_frames: int) -> BatchAnnotator:
    """Buduje anotator o zadanym budżecie klatek (bez ładowania modeli)."""
    return BatchAnnotator(BatchConfig(fps=fps, max_frames_per_video=max_frames))


class TestSamplingInterval:
    """Wyliczanie odstępu między pobieranymi klatkami."""

    def test_gdy_budzet_starcza_probkuje_w_zamowionym_tempie(self) -> None:
        """5 kl./s z 30 kl./s to co szósta klatka — 100 klatek mieści się w 120."""
        interval = _annotator(fps=5.0, max_frames=120)._sampling_interval(
            VIDEO_FPS, TOTAL_FRAMES_20S
        )
        assert interval == 6

    def test_gdy_budzet_nie_starcza_rozciaga_sie_na_cale_nagranie(self) -> None:
        interval = _annotator(fps=5.0, max_frames=50)._sampling_interval(
            VIDEO_FPS, TOTAL_FRAMES_20S
        )
        assert interval == 12

    def test_budzet_pokrywa_cale_nagranie_a_nie_jego_poczatek(self) -> None:
        """Sedno testu: ostatnia pobrana klatka leży blisko końca nagrania."""
        budget = 50
        interval = _annotator(fps=5.0, max_frames=budget)._sampling_interval(
            VIDEO_FPS, TOTAL_FRAMES_20S
        )
        last_sampled = interval * (budget - 1)
        assert last_sampled >= TOTAL_FRAMES_20S * 0.9

    def test_liczba_pobranych_klatek_miesci_sie_w_budzecie(self) -> None:
        budget = 50
        interval = _annotator(fps=5.0, max_frames=budget)._sampling_interval(
            VIDEO_FPS, TOTAL_FRAMES_20S
        )
        sampled = -(-TOTAL_FRAMES_20S // interval)
        assert sampled <= budget

    def test_nieznana_dlugosc_nagrania_nie_wysypuje_probkowania(self) -> None:
        """Niektóre kontenery nie podają liczby klatek — wracamy do tempa z fps."""
        interval = _annotator(fps=5.0, max_frames=50)._sampling_interval(VIDEO_FPS, 0)
        assert interval == 6

    def test_odstep_nigdy_nie_jest_zerowy(self) -> None:
        """fps wyższe niż nagranie nie może dać dzielenia przez zero."""
        interval = _annotator(fps=120.0, max_frames=50)._sampling_interval(
            VIDEO_FPS, TOTAL_FRAMES_20S
        )
        assert interval >= 1

    @pytest.mark.parametrize("budget", [10, 30, 50, 100])
    def test_wiekszy_budzet_nie_daje_rzadszego_probkowania(self, budget: int) -> None:
        loose = _annotator(fps=5.0, max_frames=budget)._sampling_interval(
            VIDEO_FPS, TOTAL_FRAMES_20S
        )
        tight = _annotator(fps=5.0, max_frames=max(1, budget // 2))._sampling_interval(
            VIDEO_FPS, TOTAL_FRAMES_20S
        )
        assert loose <= tight

    def test_krotkie_nagranie_probkuje_sie_w_pelnym_tempie(self) -> None:
        """Trzysekundowy klip mieści się w budżecie, więc nie ma co rozciągać."""
        interval = _annotator(fps=5.0, max_frames=50)._sampling_interval(VIDEO_FPS, 90)
        assert interval == 6
