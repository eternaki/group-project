"""
Przetwarzanie jednego treku psa: próg godności, pomiar szumu AU i wynik treku.

Każdy pies ma własny układ odniesienia. Wcześniej cała sekwencja miała jedną klatkę
neutralną, więc na wideo z wieloma psami AU liczyły się względem neutralnej innego psa.

Trek to lista klatek JEDNEGO psa. Numeracja w obrębie treku (`position`) jest inna niż
numeracja klatek wideo (`frame_idx`) — trek zwykle nie zaczyna się od klatki 0 i ma luki
(pies wychodzi z kadru, morda bywa niewykryta). Detektor klatki neutralnej i selektor
peaków pracują na pozycjach, a do zbioru trafiają numery klatek wideo — zamianę robi
`build_track_result`, żeby pomyłka nie rozlała się po miejscach wywołania.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from packages.models.delta_action_units import DeltaActionUnit
from packages.models.head_pose import HeadPose

# Minimalna liczba klatek z wykrytą mordą — z krótszego treku nie da się ani wybrać
# wiarygodnej klatki neutralnej, ani zmierzyć szumu AU
MIN_TRACK_FRAMES: int = 3
# Minimalny krótszy bok boksu mordy; poniżej keypoints są interpolacją, nie pomiarem
MIN_FACE_SIZE_PX: float = 64.0
# Minimalna średnia pewność keypoints w klatce
MIN_KEYPOINT_CONF: float = 0.4

# Minimalna liczba poprawnych pomiarów, z których wolno policzyć odchylenie.
# Z jednej próbki odchylenie nie istnieje — wpisanie tam 0.0 dałoby najgorszemu
# pomiarowi najwyższą wagę wiarygodności.
MIN_NOISE_SAMPLES: int = 2

# Znacznik "trek nie ma klatki neutralnej" (trek odrzucony)
NO_NEUTRAL_FRAME: int = -1

# Układ tablicy keypoints: [x0, y0, v0, x1, ...]
_VISIBILITY_STRIDE: int = 3
_VISIBILITY_OFFSET: int = 2

# Indeksy boksu mordy (x, y, w, h)
_FACE_BOX_WIDTH: int = 2
_FACE_BOX_HEIGHT: int = 3


@dataclass(frozen=True)
class TrackQuality:
    """
    Progi godności treku.

    Trzymane w jednym obiekcie, żeby dało się je przekazać z konfiguracji pipeline'u
    (`process_video_for_dataset`) bez rozdymania list parametrów. Domyślne wartości
    są równe stałym modułowym.

    Attributes:
        min_frames: Minimalna liczba klatek z wykrytą mordą
        min_face_px: Minimalny krótszy bok boksu mordy (mediana po klatkach)
        min_conf: Minimalna średnia pewność keypoints (mediana po klatkach)
    """

    min_frames: int = MIN_TRACK_FRAMES
    min_face_px: float = MIN_FACE_SIZE_PX
    min_conf: float = MIN_KEYPOINT_CONF


DEFAULT_TRACK_QUALITY: TrackQuality = TrackQuality()


@dataclass
class TrackFrame:
    """
    Jedna klatka w obrębie treku.

    Attributes:
        frame_idx: Numer klatki wideo (nie pozycja w treku)
        keypoints: 138 wartości [x0, y0, v0, ...] po wygładzeniu, w układzie obrazu
        face_box: Boks mordy (x, y, w, h) w układzie obrazu
        head_pose: Poza głowy policzona z wygładzonych keypoints
        delta_aus: Delta AU względem klatki neutralnej TEGO treku
    """

    frame_idx: int
    keypoints: np.ndarray
    face_box: tuple[float, float, float, float]
    head_pose: HeadPose
    delta_aus: dict[str, DeltaActionUnit] = field(default_factory=dict)


@dataclass
class TrackResult:
    """
    Wynik przetworzenia jednego treku.

    Attributes:
        track_id: Identyfikator treku z `DogTracker`
        neutral_frame_idx: Numer klatki wideo użytej jako baza AU
            (`NO_NEUTRAL_FRAME` dla treku odrzuconego)
        frames: Klatki treku
        peak_indices: Numery klatek wideo ze szczytem mimiki, w kolejności od
            najsilniejszej (nie pozycje w `frames`)
        au_noise: Odchylenie standardowe ratio na trek, osobno dla każdego AU.
            AU o mniej niż `MIN_NOISE_SAMPLES` poprawnych pomiarach nie ma tu klucza.
        au_sample_count: Liczba poprawnych pomiarów ratio, z których policzono szum.
            Bez niej nie da się skorygować obciążenia krótkich treków (patrz
            `compute_au_noise`) ani odróżnić braku pomiaru od pomiaru stabilnego.
        rejected_reason: None gdy trek przyjęty, inaczej powód odrzucenia
    """

    track_id: int
    neutral_frame_idx: int
    frames: list[TrackFrame]
    peak_indices: list[int]
    au_noise: dict[str, float]
    au_sample_count: dict[str, int] = field(default_factory=dict)
    rejected_reason: Optional[str] = None


def evaluate_track_quality(
    frames: list[TrackFrame],
    *,
    quality: TrackQuality = DEFAULT_TRACK_QUALITY,
) -> Optional[str]:
    """
    Sprawdza, czy trek nadaje się do zbioru.

    Progi liczone są z mediany po klatkach, więc pojedynczy zły kadr (rozmycie,
    chwilowe przysłonięcie) nie przekreśla całego treku.

    Args:
        frames: Klatki treku z wykrytą mordą
        quality: Progi godności (domyślnie stałe modułowe)

    Returns:
        None gdy trek przyjęty, w przeciwnym razie powód odrzucenia po polsku
    """
    if len(frames) < quality.min_frames:
        return f"za mało klatek z mordą: {len(frames)} < {quality.min_frames}"

    median_face = float(np.median([_face_size(frame) for frame in frames]))
    if median_face < quality.min_face_px:
        return f"za mała morda: {median_face:.0f} px < {quality.min_face_px:.0f} px"

    median_conf = float(np.median([_mean_visibility(frame) for frame in frames]))
    if median_conf < quality.min_conf:
        return (
            f"za niska pewność keypoints: {median_conf:.2f} < {quality.min_conf:.2f}"
        )

    return None


def compute_au_noise(frames: list[TrackFrame]) -> dict[str, float]:
    """
    Liczy odchylenie standardowe ratio każdego AU w obrębie treku.

    Wartość trafia do anotacji jako waga wiarygodności: klatka z rozdygotanego treku
    nie powinna ważyć w treningu tyle samo, co ze stabilnego. Mierzymy ratio po
    wygładzeniu (to ono trafia do zbioru), więc miara opisuje szum, który faktycznie
    zostanie w danych — zmierzone przed wygładzaniem sigma 0.35-0.76 wielokrotnie
    przekraczało próg aktywacji 0.15.

    Odchylenie jest nieobciążone (ddof=1) i liczone po klatkach osobno dla każdego AU.
    Wariant populacyjny (ddof=0) zaniża wynik tym mocniej, im krótszy trek (przy 3
    klatkach średnio o 28% wobec 11% dla ddof=1), więc premiowałby treki krótkie —
    dokładnie odwrotnie do celu tej miary. Reszty obciążenia nie da się usunąć samą
    estymatą, dlatego liczba próbek jedzie obok w `au_sample_count`.

    Pomijamy pomiary niepoliczalne (NaN/inf) — trafiłyby do JSON-a jako literał `NaN`
    (niepoprawny JSON) i zatruły każdą statystykę liczoną później.

    Args:
        frames: Klatki treku z policzonymi delta AU

    Returns:
        Słownik nazwa AU → odchylenie standardowe ratio. AU z mniej niż
        `MIN_NOISE_SAMPLES` poprawnymi pomiarami są pominięte (brak klucza).
    """
    return {
        name: float(np.std(values, ddof=1))
        for name, values in _collect_finite_ratios(frames).items()
        if len(values) >= MIN_NOISE_SAMPLES
    }


def count_au_samples(frames: list[TrackFrame]) -> dict[str, int]:
    """
    Liczy poprawne (skończone) pomiary ratio każdego AU w obrębie treku.

    Odpowiada na pytanie „z ilu klatek policzono ten szum" — bez tego nie da się
    odróżnić AU stabilnego przez cały trek od AU zmierzonego raz.

    Args:
        frames: Klatki treku z policzonymi delta AU

    Returns:
        Słownik nazwa AU → liczba poprawnych pomiarów (0 gdy wszystkie były NaN)
    """
    return {name: len(values) for name, values in _collect_finite_ratios(frames).items()}


def build_track_result(
    track_id: int,
    frames: list[TrackFrame],
    neutral_position: int,
    peak_positions: Sequence[int],
    *,
    quality: TrackQuality = DEFAULT_TRACK_QUALITY,
) -> TrackResult:
    """
    Składa wynik treku, zamieniając pozycje w treku na numery klatek wideo.

    `NeutralFrameDetector` i `PeakFrameSelector` dostają listy zbudowane z klatek treku,
    więc zwracają pozycje w tych listach. Trek zaczyna się zwykle w środku wideo i ma
    luki, więc pozycja 2 nie oznacza klatki 2 — bez tej zamiany anotacje wskazywałyby
    obce klatki.

    Próg godności sprawdzany jest tutaj, a nie u wywołującego: trek 1-klatkowy
    dostawał szum 0.0, czyli maksymalną wagę wiarygodności, gdyby wywołujący pominął
    kontrolę albo wykonał ją w złej kolejności.

    Args:
        track_id: Identyfikator treku z `DogTracker`
        frames: Klatki treku (z policzonymi delta AU)
        neutral_position: Pozycja klatki neutralnej w `frames`
        peak_positions: Pozycje klatek peak w `frames`, w kolejności od najsilniejszej
        quality: Progi godności treku (domyślnie stałe modułowe)

    Returns:
        TrackResult z numerami klatek wideo i zmierzonym szumem AU, albo wynik
        odrzucony z wypełnionym `rejected_reason`, gdy trek nie przechodzi progu

    Raises:
        ValueError: Gdy ta sama pozycja peak podana jest więcej niż raz
        IndexError: Gdy pozycja wykracza poza trek
    """
    reason = evaluate_track_quality(frames, quality=quality)
    if reason is not None:
        return rejected_track(track_id, frames, reason)

    return TrackResult(
        track_id=track_id,
        neutral_frame_idx=_video_index(frames, neutral_position, "klatki neutralnej"),
        frames=frames,
        peak_indices=_peak_video_indices(frames, peak_positions),
        au_noise=compute_au_noise(frames),
        au_sample_count=count_au_samples(frames),
    )


def rejected_track(
    track_id: int,
    frames: list[TrackFrame],
    reason: str,
) -> TrackResult:
    """
    Składa wynik odrzuconego treku z zapisanym powodem.

    Audyt lejka danych opiera się na powodach odrzuceń — odrzucenie bez powodu
    znaczy tyle, co ciche zgubienie klatek.

    Args:
        track_id: Identyfikator treku z `DogTracker`
        frames: Klatki treku zebrane do momentu odrzucenia (mogą być puste)
        reason: Powód odrzucenia po polsku (niepusty)

    Returns:
        TrackResult bez klatki neutralnej i bez peaków, z wypełnionym `rejected_reason`

    Raises:
        ValueError: Gdy powód jest pusty
    """
    if not reason.strip():
        raise ValueError(f"Trek {track_id} odrzucony bez powodu — powód jest wymagany")

    return TrackResult(
        track_id=track_id,
        neutral_frame_idx=NO_NEUTRAL_FRAME,
        frames=frames,
        peak_indices=[],
        au_noise={},
        au_sample_count={},
        rejected_reason=reason,
    )


# =============================================================================
# Funkcje pomocnicze (prywatne)
# =============================================================================

def _collect_finite_ratios(frames: list[TrackFrame]) -> dict[str, list[float]]:
    """
    Zbiera po klatkach treku wartości ratio każdego AU, pomijając NaN/inf.

    AU, którego wszystkie pomiary były niepoliczalne, zostaje w słowniku z pustą
    listą — dzięki temu `count_au_samples` może zaraportować zero prób.

    Args:
        frames: Klatki treku z policzonymi delta AU

    Returns:
        Słownik nazwa AU → lista poprawnych wartości ratio
    """
    ratios: dict[str, list[float]] = {}
    for frame in frames:
        for name, au in frame.delta_aus.items():
            samples = ratios.setdefault(name, [])
            ratio = float(au.ratio)
            if math.isfinite(ratio):
                samples.append(ratio)
    return ratios


def _face_size(frame: TrackFrame) -> float:
    """Krótszy bok boksu mordy — to on ogranicza rozdzielczość keypoints."""
    return float(min(frame.face_box[_FACE_BOX_WIDTH], frame.face_box[_FACE_BOX_HEIGHT]))


def _mean_visibility(frame: TrackFrame) -> float:
    """Średnia pewność wszystkich keypoints klatki."""
    return float(np.mean(frame.keypoints[_VISIBILITY_OFFSET::_VISIBILITY_STRIDE]))


def _peak_video_indices(
    frames: list[TrackFrame],
    peak_positions: Sequence[int],
) -> list[int]:
    """
    Zamienia pozycje peaków na numery klatek wideo, pilnując braku powtórzeń.

    Powtórzona pozycja oznacza błąd wywołującego (`PeakFrameSelector` zwraca pozycje
    unikalne), a po cichu wpisałaby tę samą klatkę do zbioru kilka razy — czyli
    zawyżyła jej wagę w treningu. Odrzucamy zamiast deduplikować, bo deduplikacja
    zwróciłaby mniej peaków, niż zamówił wywołujący, i ukryła usterkę.

    Args:
        frames: Klatki treku
        peak_positions: Pozycje peaków w `frames`

    Returns:
        Numery klatek wideo w kolejności wejściowej

    Raises:
        ValueError: Gdy któraś pozycja powtarza się
        IndexError: Gdy pozycja wykracza poza trek
    """
    repeated = sorted(
        position for position, count in Counter(peak_positions).items() if count > 1
    )
    if repeated:
        raise ValueError(f"Powtórzone pozycje peaków w treku: {repeated}")
    return [_video_index(frames, position, "peak") for position in peak_positions]


def _video_index(frames: list[TrackFrame], position: int, label: str) -> int:
    """
    Zamienia pozycję w treku na numer klatki wideo.

    UWAGA: kontrola zakresu wyłapuje pomyłkę „numer klatki zamiast pozycji" tylko
    wtedy, gdy podany numer wypada poza trekiem. W długim treku numer klatki bywa
    poprawną pozycją i takiej pomyłki nie da się wykryć — jedyną obroną jest
    trzymanie konwersji wyłącznie tutaj.

    Args:
        frames: Klatki treku
        position: Pozycja w `frames` (0..len-1), NIE numer klatki wideo
        label: Nazwa pozycji do komunikatu błędu

    Returns:
        Numer klatki wideo

    Raises:
        IndexError: Gdy pozycja wykracza poza trek
    """
    if not 0 <= position < len(frames):
        raise IndexError(
            f"Pozycja {label} poza trekiem: {position} spoza zakresu "
            f"0..{len(frames) - 1}"
        )
    return frames[position].frame_idx
