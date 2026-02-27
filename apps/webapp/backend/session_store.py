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

# Katalog sesji relatywny do pliku (niezależny od working directory)
_BACKEND_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = _BACKEND_DIR / "sessions"


@dataclass
class FrameAnnotation:
    """
    Anotacja jednej klatki wideo.

    Attributes:
        frame_idx: Indeks klatki w oryginalnym wideo
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
    """

    frame_idx: int
    image_url: str
    annotation_status: str = "auto"
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


@dataclass
class SessionData:
    """
    Dane całej sesji anotacji.

    Attributes:
        session_id: Unikalny identyfikator sesji (8 znaków hex)
        video_filename: Nazwa oryginalnego pliku wideo
        created_at: Timestamp utworzenia sesji (ISO format)
        total_frames: Całkowita liczba klatek w wideo
        neutral_frame_idx: Indeks klatki neutralnej
        neutral_keypoints: Keypoints klatki neutralnej (138 wartości)
        frames: Lista anotacji klatek (posortowana po frame_idx)
    """

    session_id: str
    video_filename: str
    created_at: str
    total_frames: int
    neutral_frame_idx: int
    neutral_keypoints: Optional[list[float]]
    frames: list[FrameAnnotation] = field(default_factory=list)


class SessionNotFoundError(KeyError):
    """Wyjątek gdy sesja nie istnieje."""


class FrameNotFoundError(KeyError):
    """Wyjątek gdy klatka nie istnieje w sesji."""


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
        frames = [FrameAnnotation(**frame) for frame in data.pop("frames", [])]
        return SessionData(**data, frames=frames)

    def get_frame(self, session_id: str, frame_idx: int) -> FrameAnnotation:
        """
        Zwraca anotację konkretnej klatki.

        Args:
            session_id: ID sesji
            frame_idx: Indeks klatki

        Returns:
            FrameAnnotation

        Raises:
            SessionNotFoundError: Gdy sesja nie istnieje
            FrameNotFoundError: Gdy klatka nie istnieje
        """
        session = self.load(session_id)
        for frame in session.frames:
            if frame.frame_idx == frame_idx:
                return frame
        raise FrameNotFoundError(
            f"Klatka {frame_idx} nie znaleziona w sesji {session_id!r}"
        )

    def update_frame(self, session_id: str, frame: FrameAnnotation) -> None:
        """
        Aktualizuje anotację klatki i zapisuje sesję na dysk.

        Args:
            session_id: ID sesji
            frame: Zaktualizowana anotacja klatki

        Raises:
            SessionNotFoundError: Gdy sesja nie istnieje
            FrameNotFoundError: Gdy klatka nie istnieje
        """
        session = self.load(session_id)
        for i, f in enumerate(session.frames):
            if f.frame_idx == frame.frame_idx:
                session.frames[i] = frame
                self.save(session)
                return
        raise FrameNotFoundError(f"Klatka {frame.frame_idx} nie znaleziona")

    def add_frame(self, session_id: str, frame: FrameAnnotation) -> None:
        """
        Dodaje nową klatkę do sesji (posortowaną po frame_idx).

        Args:
            session_id: ID sesji
            frame: Nowa anotacja klatki

        Raises:
            SessionNotFoundError: Gdy sesja nie istnieje
        """
        session = self.load(session_id)
        session.frames.append(frame)
        session.frames.sort(key=lambda f: f.frame_idx)
        self.save(session)

    def exists(self, session_id: str) -> bool:
        """Sprawdza czy sesja istnieje."""
        return (self.sessions_dir / session_id / "annotations.json").exists()
