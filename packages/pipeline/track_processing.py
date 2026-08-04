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

# Znacznik "trek nie ma klatki neutralnej" (trek odrzucony)
NO_NEUTRAL_FRAME: int = -1

# Układ tablicy keypoints: [x0, y0, v0, x1, ...]
_VISIBILITY_STRIDE: int = 3
_VISIBILITY_OFFSET: int = 2

# Indeksy boksu mordy (x, y, w, h)
_FACE_BOX_WIDTH: int = 2
_FACE_BOX_HEIGHT: int = 3


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
        au_noise: Odchylenie standardowe ratio na trek, osobno dla każdego AU
        rejected_reason: None gdy trek przyjęty, inaczej powód odrzucenia
    """

    track_id: int
    neutral_frame_idx: int
    frames: list[TrackFrame]
    peak_indices: list[int]
    au_noise: dict[str, float]
    rejected_reason: Optional[str] = None


def evaluate_track_quality(frames: list[TrackFrame]) -> Optional[str]:
    """
    Sprawdza, czy trek nadaje się do zbioru.

    Progi liczone są z mediany po klatkach, więc pojedynczy zły kadr (rozmycie,
    chwilowe przysłonięcie) nie przekreśla całego treku.

    Args:
        frames: Klatki treku z wykrytą mordą

    Returns:
        None gdy trek przyjęty, w przeciwnym razie powód odrzucenia po polsku
    """
    if len(frames) < MIN_TRACK_FRAMES:
        return f"za mało klatek z mordą: {len(frames)} < {MIN_TRACK_FRAMES}"

    median_face = float(np.median([_face_size(frame) for frame in frames]))
    if median_face < MIN_FACE_SIZE_PX:
        return f"za mała morda: {median_face:.0f} px < {MIN_FACE_SIZE_PX:.0f} px"

    median_conf = float(np.median([_mean_visibility(frame) for frame in frames]))
    if median_conf < MIN_KEYPOINT_CONF:
        return (
            f"za niska pewność keypoints: {median_conf:.2f} < {MIN_KEYPOINT_CONF:.2f}"
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

    Odchylenie liczone jest po klatkach osobno dla każdego AU (populacyjne, ddof=0 —
    tak samo jak w pomiarze, na którym oparto próg aktywacji).

    Args:
        frames: Klatki treku z policzonymi delta AU

    Returns:
        Słownik nazwa AU → odchylenie standardowe ratio (pusty dla treku bez klatek)
    """
    ratios: dict[str, list[float]] = {}
    for frame in frames:
        for name, au in frame.delta_aus.items():
            ratios.setdefault(name, []).append(float(au.ratio))
    return {name: float(np.std(values)) for name, values in ratios.items()}


def build_track_result(
    track_id: int,
    frames: list[TrackFrame],
    neutral_position: int,
    peak_positions: Sequence[int],
) -> TrackResult:
    """
    Składa wynik przyjętego treku, zamieniając pozycje w treku na numery klatek wideo.

    `NeutralFrameDetector` i `PeakFrameSelector` dostają listy zbudowane z klatek treku,
    więc zwracają pozycje w tych listach. Trek zaczyna się zwykle w środku wideo i ma
    luki, więc pozycja 2 nie oznacza klatki 2 — bez tej zamiany anotacje wskazywałyby
    obce klatki.

    Args:
        track_id: Identyfikator treku z `DogTracker`
        frames: Klatki treku (z policzonymi delta AU)
        neutral_position: Pozycja klatki neutralnej w `frames`
        peak_positions: Pozycje klatek peak w `frames`, w kolejności od najsilniejszej

    Returns:
        TrackResult z numerami klatek wideo i zmierzonym szumem AU

    Raises:
        ValueError: Gdy trek jest pusty
        IndexError: Gdy pozycja wykracza poza trek (typowo: podano numer klatki wideo)
    """
    if not frames:
        raise ValueError(f"Trek {track_id} jest pusty — nie ma z czego złożyć wyniku")

    return TrackResult(
        track_id=track_id,
        neutral_frame_idx=_video_index(frames, neutral_position, "klatki neutralnej"),
        frames=frames,
        peak_indices=[
            _video_index(frames, position, "peak") for position in peak_positions
        ],
        au_noise=compute_au_noise(frames),
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
        rejected_reason=reason,
    )


# =============================================================================
# Funkcje pomocnicze (prywatne)
# =============================================================================

def _face_size(frame: TrackFrame) -> float:
    """Krótszy bok boksu mordy — to on ogranicza rozdzielczość keypoints."""
    return float(min(frame.face_box[_FACE_BOX_WIDTH], frame.face_box[_FACE_BOX_HEIGHT]))


def _mean_visibility(frame: TrackFrame) -> float:
    """Średnia pewność wszystkich keypoints klatki."""
    return float(np.mean(frame.keypoints[_VISIBILITY_OFFSET::_VISIBILITY_STRIDE]))


def _video_index(frames: list[TrackFrame], position: int, label: str) -> int:
    """
    Zamienia pozycję w treku na numer klatki wideo.

    Args:
        frames: Klatki treku
        position: Pozycja w `frames`
        label: Nazwa pozycji do komunikatu błędu

    Returns:
        Numer klatki wideo

    Raises:
        IndexError: Gdy pozycja wykracza poza trek
    """
    if not 0 <= position < len(frames):
        raise IndexError(
            f"Pozycja {label} poza trekiem: {position} spoza zakresu "
            f"0..{len(frames) - 1} (podano numer klatki wideo zamiast pozycji?)"
        )
    return frames[position].frame_idx
