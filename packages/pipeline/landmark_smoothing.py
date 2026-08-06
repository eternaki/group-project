"""
Wygładzanie keypoints w obrębie treku filtrem One Euro.

Zmierzony szum ratio AU (sigma 0.35-0.76) przewyższa próg aktywacji (0.15) kilkukrotnie,
więc AU zapalają się od drgania punktów, nie od mimiki. Filtr 1€ (Casiez, Roussel, Vogel,
CHI 2012, używany w MediaPipe) jest adaptacyjny: przy wolnym ruchu mocno tłumi, przy
szybkim przepuszcza — dzięki temu nie zaciera szczytu mimiki, którego szukamy.

Wygładzanie działa we współrzędnych względem boksu mordy: gdyby liczyć w pikselach
kadru, ruch psa po kadrze rozmyłby punkty.

Wartości domyślne odbiegają od przykładów z publikacji z dwóch NIEZALEŻNYCH powodów:

- `beta` (5.0 zamiast 0.3) — bo ma jednostkę 1/(jednostka sygnału), a my podajemy
  sygnał znormalizowany rozmiarem mordy, nie w pikselach. Prędkość wychodzi w
  "szerokościach mordy na sekundę", więc mnożnik musi być odpowiednio większy.
  MediaPipe robi tak samo (skaluje wejście rozmiarem obiektu i używa beta 10-80).
- `min_cutoff` (0.05 zamiast 1.0) — to częstotliwość w Hz i NIE zależy od skali
  współrzędnych (alfa = 1/(1 + tau*rate) nie zawiera amplitudy sygnału). Powodem
  jest niskie próbkowanie: przy 5 fps i min_cutoff 1.0 wychodzi alfa 0.557, czyli
  filtr jest praktycznie przezroczysty.

Jedna instancja = jeden trek — nie wolno współdzielić filtra między psami.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from packages.data.schemas import NUM_KEYPOINTS

# Odcięcie przy braku ruchu [Hz] — im mniejsze, tym mocniejsze tłumienie drgań
DEFAULT_MIN_CUTOFF: float = 0.05
# Wpływ prędkości na odcięcie — im większy, tym szybsza reakcja na mimikę
DEFAULT_BETA: float = 5.0
# Odcięcie dla estymaty prędkości [Hz]
DEFAULT_D_CUTOFF: float = 1.0

# Próg ochronny przed dzieleniem przez zero przy normalizacji boksem mordy.
# Uwaga: to NIE jest kryterium sensownego rozmiaru mordy — boks kilkupikselowy
# przejdzie i da poprawny, choć mało użyteczny wynik. Filtrowaniem za małych
# mord zajmuje się próg jakości treku, nie ten moduł.
MIN_FACE_SIZE: float = 1e-6

_VALUES_PER_KEYPOINT: int = 3
_COORDS_PER_KEYPOINT: int = 2


def _alpha(rate: float, cutoff: float | np.ndarray) -> float | np.ndarray:
    """Współczynnik wygładzania wykładniczego dla odcięcia i częstotliwości próbkowania."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * rate)


def _exponential_smoothing(
    alpha: float | np.ndarray, value: np.ndarray, previous: np.ndarray
) -> np.ndarray:
    """Wygładzanie wykładnicze: alpha * nowa wartość + (1 - alpha) * poprzednia."""
    return alpha * value + (1.0 - alpha) * previous


@dataclass(eq=False)
class _FilterState:
    """Stan filtra po ostatniej przetworzonej próbce."""

    smoothed: np.ndarray
    derivative: np.ndarray
    timestamp: float


class _OneEuroFilter:
    """
    Filtr 1€ liczony niezależnie dla każdego kanału wektora.

    Prędkość liczona jest z poprzedniej wartości WYGŁADZONEJ, zgodnie z publikacją
    (CHI 2012, Appendix A, Alg. 1, linia 5: `dx <- (x - xfilt.hatxprev()) * rate`,
    gdzie daszek oznacza wartość przefiltrowaną) i z implementacjami referencyjnymi
    autorów, które w 2023 zostały jawnie poprawione z wartości surowej na
    przefiltrowaną. Prędkość jest filtrowana dolnoprzepustowo o stałym odcięciu
    `d_cutoff`, a dopiero jej moduł podnosi odcięcie sygnału głównego.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        """
        Inicjalizuje filtr.

        Raises:
            ValueError: Gdy odcięcia są niedodatnie albo beta jest ujemna
        """
        if min_cutoff <= 0.0 or d_cutoff <= 0.0:
            raise ValueError(
                f"Odcięcia muszą być dodatnie, otrzymano min_cutoff={min_cutoff}, "
                f"d_cutoff={d_cutoff}"
            )
        if beta < 0.0:
            raise ValueError(f"beta nie może być ujemna, otrzymano {beta}")

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._state: _FilterState | None = None

    def filter(self, values: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Zwraca wygładzony wektor dla podanej chwili.

        Args:
            values: Wektor sygnałów — każdy kanał filtrowany osobno
            timestamp: Czas próbki w sekundach

        Returns:
            Wygładzony wektor o kształcie wejścia
        """
        state = self._state
        if state is None:
            return self._start(values, timestamp)

        elapsed = timestamp - state.timestamp
        if elapsed < 0.0:
            # Czas cofnięty — to inny odcinek nagrania, nie ruch. Zaczynamy od nowa,
            # inaczej filtr zamarłby aż do przekroczenia poprzedniego maksimum czasu.
            return self._start(values, timestamp)
        if elapsed == 0.0:
            # Duplikat znacznika czasu nie niesie informacji o ruchu
            return np.array(state.smoothed)

        rate = 1.0 / elapsed
        derivative = _exponential_smoothing(
            _alpha(rate, self.d_cutoff), (values - state.smoothed) * rate, state.derivative
        )
        cutoff = self.min_cutoff + self.beta * np.abs(derivative)
        smoothed = _exponential_smoothing(_alpha(rate, cutoff), values, state.smoothed)

        self._state = _FilterState(smoothed.copy(), derivative, timestamp)
        return smoothed

    def _start(self, values: np.ndarray, timestamp: float) -> np.ndarray:
        """Inicjuje stan pierwszą próbką i zwraca ją bez zmian."""
        self._state = _FilterState(
            smoothed=values.copy(),
            derivative=np.zeros_like(values),
            timestamp=timestamp,
        )
        return np.array(values)


@dataclass(eq=False)
class KeypointSmoother:
    """
    Wygładza 46 keypoints jednego treku w czasie.

    Attributes:
        min_cutoff: Częstotliwość odcięcia przy braku ruchu (mniejsza = mocniejsze tłumienie)
        beta: Wpływ prędkości na odcięcie (większa = szybsza reakcja na ruch)
        d_cutoff: Odcięcie dla estymaty prędkości
    """

    min_cutoff: float = DEFAULT_MIN_CUTOFF
    beta: float = DEFAULT_BETA
    d_cutoff: float = DEFAULT_D_CUTOFF

    _filter: _OneEuroFilter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Tworzy filtr wspólny dla wszystkich współrzędnych tego treku."""
        self._filter = _OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)

    def smooth(
        self,
        keypoints_flat: np.ndarray,
        face_box: tuple[float, float, float, float],
        timestamp: float,
    ) -> np.ndarray:
        """
        Wygładza keypoints jednej klatki treku.

        Args:
            keypoints_flat: Keypoints [x0, y0, v0, ...] w układzie obrazu (138 wartości)
            face_box: Boks mordy (x, y, w, h) w układzie obrazu
            timestamp: Czas klatki w sekundach (od początku wideo)

        Returns:
            Wygładzone keypoints w układzie obrazu (138 wartości); widoczność bez zmian

        Raises:
            ValueError: Gdy kształt tablicy keypoints jest nieprawidłowy
        """
        result = _as_flat_keypoints(keypoints_flat)

        face_x, face_y, face_w, face_h = face_box
        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
            # Bez sensownego boksu nie ma jak znormalizować — oddajemy wejście nietknięte
            return result

        offset = np.array([face_x, face_y], dtype=float)
        scale = np.array([face_w, face_h], dtype=float)

        keypoints = result.reshape(NUM_KEYPOINTS, _VALUES_PER_KEYPOINT)
        local = (keypoints[:, :_COORDS_PER_KEYPOINT] - offset) / scale
        smoothed = self._filter.filter(local.ravel(), timestamp)
        keypoints[:, :_COORDS_PER_KEYPOINT] = (
            smoothed.reshape(NUM_KEYPOINTS, _COORDS_PER_KEYPOINT) * scale + offset
        )
        return result


def _as_flat_keypoints(keypoints_flat: np.ndarray) -> np.ndarray:
    """
    Sprowadza wejście do płaskiej tablicy 138 wartości, pilnując układu.

    Sam rozmiar nie wystarczy: tablica (3, 46) — układ "wiersz x / wiersz y /
    wiersz v" — ma tyle samo elementów i po spłaszczeniu zostałaby po cichu
    zinterpretowana jako przeplot [x, y, v, ...].

    Raises:
        ValueError: Gdy kształt nie jest (138,) ani (46, 3)
    """
    values = np.array(keypoints_flat, dtype=float)
    expected_flat = (NUM_KEYPOINTS * _VALUES_PER_KEYPOINT,)
    expected_pairs = (NUM_KEYPOINTS, _VALUES_PER_KEYPOINT)

    if values.shape not in (expected_flat, expected_pairs):
        raise ValueError(
            f"Oczekiwano keypoints o kształcie {expected_flat} lub {expected_pairs}, "
            f"otrzymano {values.shape}"
        )
    return values.ravel()
