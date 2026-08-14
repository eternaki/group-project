"""
SessionStore — przechowywanie i zarządzanie sesjami anotacji.

Każda sesja zawiera anotacje wszystkich klatek w rozszerzonym formacie COCO,
wzbogaconym o dane DogFACS: 46 keypoints, 21 AU, 9 emocji.

Struktura na dysku:
    sessions/{session_id}/annotations.json
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from packages.models.delta_action_units import DeltaActionUnit

# Katalog sesji relatywny do pliku (niezależny od working directory)
_BACKEND_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = _BACKEND_DIR / "sessions"


# Statusy anotacji. `verified` znaczy "człowiek to sprawdził" i tylko taka anotacja
# ma prawo trafić do zbioru jako `human_verified` — bez tego rozróżnienia Sprint 16
# nie odsieje pre-etykiet od danych zweryfikowanych.
ANNOTATION_STATUS_AUTO: str = "auto"
ANNOTATION_STATUS_REVIEWED: str = "reviewed"
ANNOTATION_STATUS_VERIFIED: str = "verified"


# Werdykt człowieka o pojedynczym AU. Trzymany OSOBNO od `aus` (pomiaru reguł),
# bo audyt pokazał, że pomiar bywa odwrotnością tego, co widać: dyszący pies
# z otwartym pyskiem dostawał `neutral`, a siedzący spaniel `surprise`.
# Gdyby werdykt startował od wartości reguły, anotator zatwierdzałby błąd
# jednym kliknięciem — a to jest dokładnie to, czego zbiór nie przetrwa.
AU_VERDICT_ACTIVE: str = "active"
AU_VERDICT_INACTIVE: str = "inactive"

# `not_observable` NIE ZNACZY „nieaktywny". Ucho schowane za głową to brak
# wiedzy, a nie brak ruchu — wpisanie tu zera nauczyłoby sieć, że niewidoczne
# znaczy spoczynkowe. To rozróżnienie jest powodem, dla którego pole ma trzy
# stany zamiast być zwykłym `bool`.
AU_VERDICT_NOT_OBSERVABLE: str = "not_observable"

AU_VERDICTS: frozenset[str] = frozenset(
    {AU_VERDICT_ACTIVE, AU_VERDICT_INACTIVE, AU_VERDICT_NOT_OBSERVABLE}
)


@dataclass
class FrameAnnotation:
    """
    Anotacja jednej klatki wideo dla JEDNEGO psa.

    Klucz anotacji to para (`frame_idx`, `track_id`), a nie sam numer klatki:
    w kadrze bywa kilka psów i każdy ma własne keypoints, AU i emocję.

    Attributes:
        frame_idx: Indeks klatki w oryginalnym wideo
        track_id: Identyfikator psa (treku), którego dotyczy anotacja.
            None w sesjach sprzed obsługi wielu psów — wtedy klatka opisuje
            jedynego psa, jakiego wtedy zapisywano.
        image_url: URL do podglądu klatki (np. /static/{session_id}/frame_0010.jpg)
        annotation_status: Status anotacji — auto | reviewed | verified
        source: Źródło anotacji — ai | manual
        bbox: Bounding box [x, y, w, h]
        keypoints: Flat array 46×3=138 wartości [x0, y0, v0, ...]
        aus: Action Units {name: {ratio, delta, is_active, confidence}}
        emotion: Rozpoznana emocja (9 klas DogFACS)
        emotion_confidence: Pewność klasyfikacji emocji (0-1)
        emotion_rule_applied: Nazwa zastosowanej reguły
        breed: Rasa psa
        breed_confidence: Pewność klasyfikacji rasy (0-1)
        tfm_score: Temporal Feature Map score klatki
        au_verdicts: Werdykty człowieka o poszczególnych AU
            {nazwa AU: active | inactive | not_observable}. BRAK klucza znaczy
            „jeszcze nieoceniony" — dlatego domyślnie słownik jest pusty, a nie
            wypełniony wartościami z reguł.
        frame_role: `neutral` albo `peak` — która klatka pary
        quality: Miary bramki jakości {asymmetry, weak_keypoint_ratio, face_width_px}
        review_order: Pozycja pary w kolejce weryfikacji
    """

    frame_idx: int
    image_url: str
    track_id: Optional[int] = None
    annotation_status: str = ANNOTATION_STATUS_AUTO
    source: str = "ai"
    bbox: Optional[list[float]] = None
    keypoints: Optional[list[float]] = None
    aus: dict = field(default_factory=dict)
    emotion: Optional[str] = None
    emotion_confidence: float = 0.0
    emotion_rule_applied: Optional[str] = None
    breed: Optional[str] = None
    breed_confidence: float = 0.0
    tfm_score: float = 0.0
    au_verdicts: dict = field(default_factory=dict)
    frame_role: Optional[str] = None
    quality: dict = field(default_factory=dict)
    review_order: Optional[int] = None
    # Czy człowiek uznał kadr za nadający się do kodowania AU. Miary geometryczne
    # tego nie rozstrzygają: przegląd kadrów pokazał, że nawet przy najniższej
    # asymetrii połowa ujęć to psy z głową w dół albo odwrócone, bo asymetria
    # jest miarą lewo-prawo i na pochylenie głowy jest z definicji ślepa.
    # Odrzucenie ręczne zajmuje sekundę i jest samo w sobie etykietą.
    usable: bool = True


@dataclass
class DogTrack:
    """
    Jeden pies (trek) w sesji — z WŁASNĄ klatką neutralną.

    Klatka neutralna jest per pies, bo delta AU liczy się względem niej. Jedna
    klatka neutralna na całe wideo znaczyłaby, że AU drugiego psa mierzone są
    względem mimiki pierwszego.

    Attributes:
        track_id: Identyfikator psa (z `DogTracker`)
        neutral_frame_idx: Numer klatki wideo użytej jako baza AU tego psa
        neutral_keypoints: Keypoints klatki neutralnej (138 wartości)
        neutral_source: `auto` (detektor) albo `manual` (wskazana ręcznie)
        peak_frame_indices: Numery klatek szczytowych tego psa
        au_noise: Odchylenie ratio każdego AU w treku — waga wiarygodności
        au_sample_count: Liczba pomiarów, z których policzono szum. Bez niej sigma
            z 3 klatek jest nieodróżnialna od sigmy z 20, a obciążenie krótkich
            treków (~11% przy 3 próbkach) nie do skorygowania w treningu.
    """

    track_id: int
    neutral_frame_idx: int
    neutral_keypoints: Optional[list[float]] = None
    neutral_source: str = "auto"
    peak_frame_indices: list[int] = field(default_factory=list)
    au_noise: dict = field(default_factory=dict)
    au_sample_count: dict = field(default_factory=dict)


@dataclass
class RejectedTrack:
    """
    Pies odrzucony przez próg godności treku — razem z powodem.

    Powód jest częścią wyniku, a nie śmieciem do wyrzucenia: bez niego anotator
    widzi tylko, że pies z nagrania zniknął, i nie wie, czy to usterka, czy
    świadomy odsiew (za mała morda, za krótki trek, za niska pewność).

    Attributes:
        track_id: Identyfikator psa (z `DogTracker`)
        reason: Powód odrzucenia po polsku
        frames: Ile klatek trek zdążył zebrać
        frames_dropped_low_conf: Ile klatek odsiał próg pewności keypoints
    """

    track_id: int
    reason: str
    frames: int = 0
    frames_dropped_low_conf: int = 0


@dataclass
class SessionData:
    """
    Dane całej sesji anotacji.

    Attributes:
        session_id: Unikalny identyfikator sesji (8 znaków hex)
        video_filename: Nazwa oryginalnego pliku wideo
        created_at: Timestamp utworzenia sesji (ISO format)
        total_frames: Całkowita liczba klatek w wideo
        neutral_frame_idx: Klatka neutralna PIERWSZEGO psa. Pole zgodności ze
            starymi sesjami — przy wielu psach właściwym źródłem jest `dogs`.
        neutral_keypoints: Keypoints klatki neutralnej pierwszego psa
        frames: Lista anotacji klatek (posortowana po frame_idx i track_id)
        dogs: Psy przyjęte do anotacji, każdy z własną klatką neutralną
        rejected_tracks: Psy odrzucone wraz z powodem (do pokazania anotatorowi)
    """

    session_id: str
    video_filename: str
    created_at: str
    total_frames: int
    neutral_frame_idx: int
    neutral_keypoints: Optional[list[float]]
    frames: list[FrameAnnotation] = field(default_factory=list)
    dogs: list[DogTrack] = field(default_factory=list)
    rejected_tracks: list[RejectedTrack] = field(default_factory=list)

    def neutral_keypoints_for(self, track_id: Optional[int]) -> Optional[list[float]]:
        """
        Zwraca keypoints klatki neutralnej WŁAŚCIWEGO psa.

        Args:
            track_id: Identyfikator psa albo None (sesja sprzed obsługi wielu psów)

        Returns:
            Keypoints klatki neutralnej tego psa; gdy psa nie ma w `dogs`,
            wartość sesyjna (zgodność wstecz), a gdy i jej brak — None
        """
        for dog in self.dogs:
            if dog.track_id == track_id and dog.neutral_keypoints is not None:
                return dog.neutral_keypoints
        return self.neutral_keypoints


class SessionNotFoundError(KeyError):
    """Wyjątek gdy sesja nie istnieje."""


class FrameNotFoundError(KeyError):
    """Wyjątek gdy klatka nie istnieje w sesji."""


class AmbiguousFrameError(KeyError):
    """
    Wyjątek gdy numer klatki wskazuje kilku psów naraz.

    Bez niego edycja klatki z dwoma psami trafiałaby po cichu w pierwszego
    z brzegu — czyli nadpisywałaby anotację niewłaściwego psa.
    """


def delta_aus_to_dict(delta_aus: dict[str, DeltaActionUnit]) -> dict:
    """
    Konwertuje AU do postaci zapisywanej w sesji i wysyłanej do frontendu.

    Samo `ratio` nie odróżnia realnej aktywacji od pomiaru klamrowanego, więc
    zapisujemy komplet: ratio, delta, is_active i pewność.

    Args:
        delta_aus: AU policzone przez `DeltaActionUnitsExtractor`

    Returns:
        Słownik nazwa AU → {ratio, delta, is_active, confidence} z typami Pythona
        (numpy.bool_ nie serializuje się do JSON)
    """
    return {
        name: {
            "ratio": float(au.ratio),
            "delta": float(au.delta),
            "is_active": bool(au.is_active),
            "confidence": float(au.confidence),
        }
        for name, au in delta_aus.items()
    }


# Zastępnik `track_id` przy sortowaniu: sesje sprzed obsługi wielu psów mają
# tam None, a None nie da się porównać z liczbą
_NO_TRACK_SORT_KEY: int = -1


def _sort_key(frame: FrameAnnotation) -> tuple[int, int]:
    """Porządek klatek w sesji: po numerze klatki, potem po psie."""
    return (
        frame.frame_idx,
        _NO_TRACK_SORT_KEY if frame.track_id is None else frame.track_id,
    )


def _same_annotation(
    frame: FrameAnnotation,
    frame_idx: int,
    track_id: Optional[int],
) -> bool:
    """
    Czy anotacja dotyczy tej klatki i tego psa.

    `track_id=None` w zapytaniu znaczy „dowolny pies" (zgodność ze starymi
    sesjami), a nie „pies bez identyfikatora".

    Args:
        frame: Anotacja zapisana w sesji
        frame_idx: Szukany numer klatki
        track_id: Szukany pies albo None

    Returns:
        True, gdy anotacja pasuje do zapytania
    """
    if frame.frame_idx != frame_idx:
        return False
    return track_id is None or frame.track_id == track_id


def _matching_frames(
    frames: list[FrameAnnotation],
    frame_idx: int,
    track_id: Optional[int],
) -> list[FrameAnnotation]:
    """Zwraca anotacje pasujące do pary (klatka, pies)."""
    return [frame for frame in frames if _same_annotation(frame, frame_idx, track_id)]


class SessionStore:
    """
    Zarządza sesjami anotacji — zapis i odczyt z dysku.

    Sesje przechowywane są jako pliki JSON w katalogu sessions_dir.
    Każda sesja = jeden plik annotations.json.

    Przykład:
        >>> store = SessionStore()
        >>> store.save(session_data)
        >>> session = store.load("abc12345")
        >>> frame = store.get_frame("abc12345", 10)
    """

    def __init__(self, sessions_dir: Path = SESSIONS_DIR) -> None:
        """
        Inicjalizuje store.

        Args:
            sessions_dir: Katalog przechowywania sesji
        """
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session: SessionData) -> None:
        """
        Zapisuje sesję na dysk.

        Args:
            session: Dane sesji do zapisania
        """
        session_dir = self.sessions_dir / session.session_id
        session_dir.mkdir(exist_ok=True)
        path = session_dir / "annotations.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(session), f, ensure_ascii=False, indent=2)

    def load(self, session_id: str) -> SessionData:
        """
        Wczytuje sesję z dysku.

        Args:
            session_id: ID sesji

        Returns:
            SessionData z wszystkimi anotacjami

        Raises:
            SessionNotFoundError: Gdy sesja nie istnieje
        """
        path = self.sessions_dir / session_id / "annotations.json"
        if not path.exists():
            raise SessionNotFoundError(f"Sesja {session_id!r} nie istnieje")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Sesje sprzed obsługi wielu psów nie mają `dogs` ani `rejected_tracks`
        # — brak klucza czytamy jako pustą listę, nie jako błąd.
        return SessionData(
            **{
                key: value
                for key, value in data.items()
                if key not in ("frames", "dogs", "rejected_tracks")
            },
            frames=[FrameAnnotation(**frame) for frame in data.get("frames", [])],
            dogs=[DogTrack(**dog) for dog in data.get("dogs", [])],
            rejected_tracks=[
                RejectedTrack(**track) for track in data.get("rejected_tracks", [])
            ],
        )

    def get_frame(
        self,
        session_id: str,
        frame_idx: int,
        track_id: Optional[int] = None,
    ) -> FrameAnnotation:
        """
        Zwraca anotację klatki wskazanego psa.

        Args:
            session_id: ID sesji
            frame_idx: Indeks klatki
            track_id: Identyfikator psa; None znaczy „którykolwiek", ale tylko
                wtedy, gdy klatka opisuje dokładnie jednego psa

        Returns:
            FrameAnnotation

        Raises:
            SessionNotFoundError: Gdy sesja nie istnieje
            FrameNotFoundError: Gdy klatka nie istnieje
            AmbiguousFrameError: Gdy klatka opisuje kilku psów, a `track_id` nie podano
        """
        matches = _matching_frames(self.load(session_id).frames, frame_idx, track_id)
        if not matches:
            raise FrameNotFoundError(
                f"Klatka {frame_idx} nie znaleziona w sesji {session_id!r}"
            )
        if len(matches) > 1:
            raise AmbiguousFrameError(
                f"Klatka {frame_idx} w sesji {session_id!r} opisuje {len(matches)} psów "
                f"(track_id: {[f.track_id for f in matches]}) — wskaż, którego dotyczy"
            )
        return matches[0]

    def update_frame(self, session_id: str, frame: FrameAnnotation) -> None:
        """
        Aktualizuje anotację klatki i zapisuje sesję na dysk.

        Klatkę wskazuje para (`frame_idx`, `track_id`) niesiona przez `frame`.

        Args:
            session_id: ID sesji
            frame: Zaktualizowana anotacja klatki

        Raises:
            SessionNotFoundError: Gdy sesja nie istnieje
            FrameNotFoundError: Gdy klatka nie istnieje
            AmbiguousFrameError: Gdy klatka bez `track_id` pasuje do kilku psów
        """
        session = self.load(session_id)
        positions = [
            i
            for i, stored in enumerate(session.frames)
            if _same_annotation(stored, frame.frame_idx, frame.track_id)
        ]
        if not positions:
            raise FrameNotFoundError(f"Klatka {frame.frame_idx} nie znaleziona")
        if len(positions) > 1:
            raise AmbiguousFrameError(
                f"Klatka {frame.frame_idx} pasuje do {len(positions)} psów — "
                "anotacja musi nieść track_id"
            )
        session.frames[positions[0]] = frame
        self.save(session)

    def add_frame(self, session_id: str, frame: FrameAnnotation) -> None:
        """
        Dodaje nową klatkę do sesji (posortowaną po frame_idx i track_id).

        Args:
            session_id: ID sesji
            frame: Nowa anotacja klatki

        Raises:
            SessionNotFoundError: Gdy sesja nie istnieje
        """
        session = self.load(session_id)
        session.frames.append(frame)
        session.frames.sort(key=_sort_key)
        self.save(session)

    def exists(self, session_id: str) -> bool:
        """Sprawdza czy sesja istnieje."""
        return (self.sessions_dir / session_id / "annotations.json").exists()
