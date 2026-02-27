"""
Testy dla SessionStore i modeli danych sesji (Sprint 9 — DOG-S9-1).

Uruchomienie:
    pytest tests/test_backend/test_session_store.py -v
"""

from pathlib import Path

import pytest
from session_store import (
    FrameAnnotation,
    FrameNotFoundError,
    SessionData,
    SessionNotFoundError,
    SessionStore,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_store(tmp_path: Path) -> SessionStore:
    """SessionStore z tymczasowym katalogiem."""
    return SessionStore(sessions_dir=tmp_path / "sessions")


@pytest.fixture
def sample_frame() -> FrameAnnotation:
    """Przykładowa anotacja klatki."""
    return FrameAnnotation(
        frame_idx=10,
        image_url="/static/abc12345/frame_0010.jpg",
        emotion="happy",
        emotion_confidence=0.85,
        tfm_score=0.75,
        aus={
            "AU101": {"ratio": 1.2, "delta": 0.2, "is_active": True, "confidence": 0.9},
        },
        keypoints=list(range(138)),
    )


@pytest.fixture
def sample_session(sample_frame: FrameAnnotation) -> SessionData:
    """Przykładowa sesja z jedną klatką."""
    return SessionData(
        session_id="abc12345",
        video_filename="dog_video.mp4",
        created_at="2026-02-27T12:00:00",
        total_frames=100,
        neutral_frame_idx=5,
        neutral_keypoints=list(range(138)),
        frames=[sample_frame],
    )


# =============================================================================
# Testy FrameAnnotation
# =============================================================================


class TestFrameAnnotation:
    """Testy modelu danych klatki."""

    def test_default_values(self) -> None:
        """Test domyślnych wartości FrameAnnotation."""
        frame = FrameAnnotation(frame_idx=0, image_url="/static/test/frame_0000.jpg")

        assert frame.annotation_status == "auto"
        assert frame.source == "ai"
        assert frame.bbox is None
        assert frame.keypoints is None
        assert frame.aus == {}
        assert frame.emotion is None
        assert frame.emotion_confidence == 0.0

    def test_custom_values(self) -> None:
        """Test ustawiania niestandardowych wartości."""
        frame = FrameAnnotation(
            frame_idx=5,
            image_url="/static/s1/frame_0005.jpg",
            annotation_status="reviewed",
            source="manual",
            emotion="happy",
            emotion_confidence=0.9,
            breed="labrador",
        )

        assert frame.annotation_status == "reviewed"
        assert frame.source == "manual"
        assert frame.emotion == "happy"
        assert frame.breed == "labrador"


# =============================================================================
# Testy SessionStore
# =============================================================================


class TestSessionStore:
    """Testy klasy SessionStore."""

    def test_save_creates_file(
        self, temp_store: SessionStore, sample_session: SessionData
    ) -> None:
        """Test: save() tworzy plik annotations.json."""
        temp_store.save(sample_session)

        path = temp_store.sessions_dir / "abc12345" / "annotations.json"
        assert path.exists()

    def test_save_and_load_roundtrip(
        self, temp_store: SessionStore, sample_session: SessionData
    ) -> None:
        """Test: zapisana sesja jest identyczna po wczytaniu."""
        temp_store.save(sample_session)
        loaded = temp_store.load("abc12345")

        assert loaded.session_id == "abc12345"
        assert loaded.video_filename == "dog_video.mp4"
        assert loaded.total_frames == 100
        assert loaded.neutral_frame_idx == 5
        assert len(loaded.frames) == 1
        assert loaded.frames[0].frame_idx == 10
        assert loaded.frames[0].emotion == "happy"

    def test_load_nonexistent_raises(self, temp_store: SessionStore) -> None:
        """Test: load() rzuca SessionNotFoundError dla nieznanej sesji."""
        with pytest.raises(SessionNotFoundError):
            temp_store.load("nonexistent")

    def test_exists_returns_false_for_missing(self, temp_store: SessionStore) -> None:
        """Test: exists() zwraca False dla nieistniejącej sesji."""
        assert temp_store.exists("missing") is False

    def test_exists_returns_true_after_save(
        self, temp_store: SessionStore, sample_session: SessionData
    ) -> None:
        """Test: exists() zwraca True po zapisaniu sesji."""
        temp_store.save(sample_session)
        assert temp_store.exists("abc12345") is True

    def test_get_frame_returns_correct_frame(
        self, temp_store: SessionStore, sample_session: SessionData
    ) -> None:
        """Test: get_frame() zwraca właściwą klatkę."""
        temp_store.save(sample_session)
        frame = temp_store.get_frame("abc12345", 10)

        assert frame.frame_idx == 10
        assert frame.emotion == "happy"

    def test_get_frame_nonexistent_raises(
        self, temp_store: SessionStore, sample_session: SessionData
    ) -> None:
        """Test: get_frame() rzuca FrameNotFoundError dla nieistniejącej klatki."""
        temp_store.save(sample_session)
        with pytest.raises(FrameNotFoundError):
            temp_store.get_frame("abc12345", 999)

    def test_update_frame_saves_changes(
        self, temp_store: SessionStore, sample_session: SessionData
    ) -> None:
        """Test: update_frame() zapisuje zmiany do pliku."""
        temp_store.save(sample_session)

        frame = temp_store.get_frame("abc12345", 10)
        frame.emotion = "sad"
        frame.annotation_status = "reviewed"
        temp_store.update_frame("abc12345", frame)

        updated = temp_store.get_frame("abc12345", 10)
        assert updated.emotion == "sad"
        assert updated.annotation_status == "reviewed"

    def test_update_nonexistent_frame_raises(
        self, temp_store: SessionStore, sample_session: SessionData
    ) -> None:
        """Test: update_frame() rzuca FrameNotFoundError dla nieznanej klatki."""
        temp_store.save(sample_session)
        ghost_frame = FrameAnnotation(frame_idx=999, image_url="/static/x/frame_0999.jpg")
        with pytest.raises(FrameNotFoundError):
            temp_store.update_frame("abc12345", ghost_frame)

    def test_add_frame_inserts_sorted(
        self, temp_store: SessionStore, sample_session: SessionData
    ) -> None:
        """Test: add_frame() wstawia klatkę posortowaną po frame_idx."""
        temp_store.save(sample_session)

        new_frame = FrameAnnotation(frame_idx=5, image_url="/static/abc12345/frame_0005.jpg")
        temp_store.add_frame("abc12345", new_frame)

        session = temp_store.load("abc12345")
        indices = [f.frame_idx for f in session.frames]
        assert indices == sorted(indices)
        assert 5 in indices

    def test_neutral_keypoints_preserved(
        self, temp_store: SessionStore, sample_session: SessionData
    ) -> None:
        """Test: neutral_keypoints są poprawnie zapisane i wczytane."""
        temp_store.save(sample_session)
        loaded = temp_store.load("abc12345")

        assert loaded.neutral_keypoints == list(range(138))

    def test_aus_preserved_in_roundtrip(
        self, temp_store: SessionStore, sample_session: SessionData
    ) -> None:
        """Test: AU dict jest poprawnie zachowany po roundtrip."""
        temp_store.save(sample_session)
        frame = temp_store.get_frame("abc12345", 10)

        assert "AU101" in frame.aus
        assert frame.aus["AU101"]["ratio"] == pytest.approx(1.2)
        assert frame.aus["AU101"]["is_active"] is True
