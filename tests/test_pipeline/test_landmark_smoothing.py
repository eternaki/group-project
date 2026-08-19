"""Testy wygładzania keypoints w obrębie treku."""

import numpy as np
import pytest

from packages.data.schemas import NUM_KEYPOINTS
from packages.pipeline.landmark_smoothing import KeypointSmoother

FACE_BOX: tuple[float, float, float, float] = (100.0, 100.0, 200.0, 200.0)
FPS: float = 5.0
VIDEO_FPS: float = 25.0
NOISE_PX: float = 3.0
NOISE_STEPS: int = 30
WARMUP_FRAMES: int = 5


def _static_frame() -> np.ndarray:
    """Nieruchoma morda: wszystkie punkty w (150, 150), pełna widoczność."""
    frame = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
    frame[0::3] = 150.0
    frame[1::3] = 150.0
    frame[2::3] = 1.0
    return frame


def _noisy_series(rng: np.random.Generator, steps: int, noise: float) -> list[np.ndarray]:
    """Nieruchoma morda z szumem gaussowskim na każdej współrzędnej."""
    base = _static_frame()

    series = []
    for _ in range(steps):
        frame = base.copy()
        frame[0::3] += rng.normal(0, noise, NUM_KEYPOINTS)
        frame[1::3] += rng.normal(0, noise, NUM_KEYPOINTS)
        series.append(frame)
    return series


def _mean_coordinate_std(series: list[np.ndarray]) -> float:
    """Średni po współrzędnych rozrzut w czasie (wszystkie osie x i y)."""
    stacked = np.array(series)
    coords = np.r_[0 : NUM_KEYPOINTS * 3 : 3, 1 : NUM_KEYPOINTS * 3 : 3]
    return float(np.std(stacked[:, coords], axis=0).mean())


class TestKeypointSmoother:
    """Filtr One Euro na trek."""

    def test_tlumi_drganie_nieruchomej_mordy(self) -> None:
        """
        Amplituda drgania ma spaść co najmniej dwukrotnie.

        Rozrzut liczymy na wszystkich 92 współrzędnych naraz, a nie na jednej:
        odchylenie z 25 próbek jednej współrzędnej ma kilkunastoprocentowy błąd
        własny, więc pojedyncza współrzędna czyniłaby z testu loterię zależną
        od ziarna losowania.
        """
        rng = np.random.default_rng(0)
        series = _noisy_series(rng, steps=NOISE_STEPS, noise=NOISE_PX)
        smoother = KeypointSmoother()

        smoothed = [
            smoother.smooth(frame, FACE_BOX, timestamp=i / FPS)
            for i, frame in enumerate(series)
        ]

        raw_std = _mean_coordinate_std(series)
        smooth_std = _mean_coordinate_std(smoothed[WARMUP_FRAMES:])

        assert smooth_std < raw_std / 2, f"szum ma spaść, było {raw_std}, jest {smooth_std}"

    def test_przepuszcza_szybki_ruch_mimiki(self) -> None:
        """Skok (szczyt mimiki) nie może zostać zatarty przez filtr."""
        smoother = KeypointSmoother()
        base = _static_frame()

        for i in range(10):
            smoother.smooth(base, FACE_BOX, timestamp=i / FPS)

        jumped = base.copy()
        jumped[1::3] = 180.0  # wyraźne otwarcie pyska
        result = smoother.smooth(jumped, FACE_BOX, timestamp=10 / FPS)

        assert result[1] > 170.0, "szybka zmiana ma zostać przepuszczona"

    def test_zachowuje_widocznosc_punktow(self) -> None:
        smoother = KeypointSmoother()
        frame = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
        frame[2::3] = 0.7

        result = smoother.smooth(frame, FACE_BOX, timestamp=0.0)

        assert np.allclose(result[2::3], 0.7)

    def test_ruch_psa_po_kadrze_nie_rozmywa_punktow(self) -> None:
        """Wygładzanie działa w układzie mordy, więc przesunięcie boksu nie szkodzi."""
        smoother = KeypointSmoother()
        first = _static_frame()

        smoother.smooth(first, (100.0, 100.0, 200.0, 200.0), timestamp=0.0)

        moved = first.copy()
        moved[0::3] = 350.0  # ta sama morda, przesunięta o 200 px w prawo
        result = smoother.smooth(moved, (300.0, 100.0, 200.0, 200.0), timestamp=1.0 / FPS)

        assert result[0] > 330.0, "punkt ma podążyć za mordą, a nie zostać w tyle"

    def test_pierwsza_klatka_wraca_bez_zmian(self) -> None:
        """Brak historii = nie ma czego wygładzać."""
        smoother = KeypointSmoother()
        frame = _static_frame()

        result = smoother.smooth(frame, FACE_BOX, timestamp=0.0)

        assert np.allclose(result, frame)

    def test_powtorzony_znacznik_czasu_nie_psuje_stanu(self) -> None:
        """Dwie klatki z tym samym czasem: brak dzielenia przez zero."""
        smoother = KeypointSmoother()
        frame = _static_frame()

        smoother.smooth(frame, FACE_BOX, timestamp=0.0)
        result = smoother.smooth(frame, FACE_BOX, timestamp=0.0)

        assert np.all(np.isfinite(result))
        assert np.allclose(result[0::3], 150.0)

    def test_zdegenerowany_boks_zwraca_wejscie(self) -> None:
        """Boks o zerowej szerokości: nie ma jak znormalizować, zwracamy wejście."""
        smoother = KeypointSmoother()
        frame = _static_frame()

        result = smoother.smooth(frame, (100.0, 100.0, 0.0, 200.0), timestamp=0.0)

        assert np.allclose(result, frame)

    def test_bledna_liczba_wartosci_konczy_wyjatkiem(self) -> None:
        smoother = KeypointSmoother()

        with pytest.raises(ValueError, match="138"):
            smoother.smooth(np.zeros(100, dtype=float), FACE_BOX, timestamp=0.0)

    def test_zly_uklad_o_poprawnej_liczbie_wartosci_konczy_wyjatkiem(self) -> None:
        """Tablica (3, 46) ma 138 wartości, ale to inny układ — nie wolno jej milcząco przyjąć."""
        smoother = KeypointSmoother()

        with pytest.raises(ValueError, match="kształcie"):
            smoother.smooth(np.zeros((3, NUM_KEYPOINTS), dtype=float), FACE_BOX, timestamp=0.0)

    def test_cofniety_czas_restartuje_filtr(self) -> None:
        """
        Regresja: czas cofnięty (sklejone segmenty, reset numeracji klatek) nie może
        zamrozić filtra aż do przekroczenia poprzedniego maksimum czasu.
        """
        smoother = KeypointSmoother()
        frame = _static_frame()
        for i in range(6):
            smoother.smooth(frame, FACE_BOX, timestamp=i / FPS)

        moved = frame.copy()
        moved[0::3] = 250.0
        result = smoother.smooth(moved, FACE_BOX, timestamp=0.0)

        assert np.allclose(result[0::3], 250.0), "po restarcie pierwsza klatka wraca bez zmian"

    def test_kanaly_nie_przeciekaja(self) -> None:
        """Skok w jednej współrzędnej nie może poruszyć pozostałych."""
        smoother = KeypointSmoother()
        frame = _static_frame()
        for i in range(6):
            smoother.smooth(frame, FACE_BOX, timestamp=i / FPS)

        jumped = frame.copy()
        jumped[0] = 400.0
        result = smoother.smooth(jumped, FACE_BOX, timestamp=6 / FPS)

        assert result[0] > 300.0
        assert np.allclose(result[3::3], 150.0)

    def test_tlumi_szum_takze_przy_pelnym_fps(self) -> None:
        """Kalibracja domyślna jest robiona przy 5 fps — przy 25 fps też ma tłumić."""
        rng = np.random.default_rng(1)
        series = _noisy_series(rng, steps=NOISE_STEPS, noise=NOISE_PX)
        smoother = KeypointSmoother()

        smoothed = [
            smoother.smooth(frame, FACE_BOX, timestamp=i / VIDEO_FPS)
            for i, frame in enumerate(series)
        ]

        raw_std = _mean_coordinate_std(series)
        smooth_std = _mean_coordinate_std(smoothed[WARMUP_FRAMES:])

        assert smooth_std < raw_std / 2


class TestSmootherParameters:
    """Walidacja parametrów filtra."""

    @pytest.mark.parametrize(
        "kwargs",
        [{"min_cutoff": 0.0}, {"min_cutoff": -1.0}, {"d_cutoff": 0.0}, {"beta": -5.0}],
    )
    def test_niepoprawne_parametry_koncza_wyjatkiem(self, kwargs: dict) -> None:
        """
        Bez walidacji `min_cutoff=0` dawało ciche `RuntimeWarning: divide by zero`,
        `d_cutoff=0` gołe `ZeroDivisionError`, a ujemna beta niestabilność
        (skok 150→250 przestrzeliwał do 296).
        """
        with pytest.raises(ValueError):
            KeypointSmoother(**kwargs)
