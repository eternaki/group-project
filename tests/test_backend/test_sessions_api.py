"""
Testy API dla endpointów sesji anotacji (Sprint 9 — DOG-S9-2..10).

Uruchomienie:
    pytest tests/test_backend/test_sessions_api.py -v
"""

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from session_store import FrameAnnotation, SessionData, SessionStore

from packages.data.schemas import NUM_KEYPOINTS

# =============================================================================
# Fixtures
# =============================================================================


SESSION_ID = "test1234"


@pytest.fixture
def temp_store(tmp_path: Path) -> SessionStore:
    """Izolowany SessionStore z tymczasowym katalogiem."""
    return SessionStore(sessions_dir=tmp_path / "sessions")


@pytest.fixture
def populated_store(temp_store: SessionStore) -> SessionStore:
    """SessionStore z gotową sesją testową."""
    frame = FrameAnnotation(
        frame_idx=10,
        image_url=f"/static/{SESSION_ID}/frame_0010.jpg",
        emotion="happy",
        emotion_confidence=0.85,
        tfm_score=0.75,
        aus={
            "AU101": {"ratio": 1.2, "delta": 0.2, "is_active": True, "confidence": 0.9},
            "AU143": {"ratio": 0.7, "delta": -0.3, "is_active": True, "confidence": 0.8},
        },
        keypoints=[float(i % 3) for i in range(NUM_KEYPOINTS * 3)],
    )
    session = SessionData(
        session_id=SESSION_ID,
        video_filename="test_video.mp4",
        created_at="2026-02-27T12:00:00",
        total_frames=50,
        neutral_frame_idx=5,
        neutral_keypoints=[0.5] * (NUM_KEYPOINTS * 3),
        frames=[frame],
    )
    temp_store.save(session)
    return temp_store


@pytest.fixture
def app(populated_store: SessionStore, monkeypatch) -> FastAPI:
    """
    FastAPI app z routerem sesji i podmienioną zależnością SessionStore.
    """
    import routers.sessions as sessions_module

    monkeypatch.setattr(sessions_module, "_store", populated_store)

    test_app = FastAPI()
    from routers.sessions import router

    test_app.include_router(router)
    return test_app


@pytest.fixture
async def client(app: FastAPI):
    """Asynchroniczny klient HTTP (httpx + ASGITransport)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# =============================================================================
# Testy GET endpoints — DOG-S9-8
# =============================================================================


@pytest.mark.anyio
class TestGetSession:
    """Testy GET /api/sessions/{session_id}."""

    async def test_get_existing_session(self, client) -> None:
        """Test pobierania istniejącej sesji."""
        resp = await client.get(f"/api/sessions/{SESSION_ID}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == SESSION_ID
        assert data["video_filename"] == "test_video.mp4"
        assert len(data["frames"]) == 1

    async def test_get_nonexistent_session_returns_404(self, client) -> None:
        """Test: nieistniejąca sesja → 404."""
        resp = await client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404

    async def test_list_frames(self, client) -> None:
        """Test GET /frames zwraca listę klatek."""
        resp = await client.get(f"/api/sessions/{SESSION_ID}/frames")

        assert resp.status_code == 200
        data = resp.json()
        assert "frames" in data
        assert len(data["frames"]) == 1
        assert data["frames"][0]["frame_idx"] == 10


# =============================================================================
# Testy PATCH endpoints — DOG-S9-2..5
# =============================================================================


@pytest.mark.anyio
class TestUpdateKeypoints:
    """Testy PATCH /frames/{idx}/keypoints (DOG-S9-2)."""

    async def test_update_valid_keypoints(self, client, populated_store) -> None:
        """Test aktualizacji poprawnych keypoints."""
        new_kp = [1.0] * (NUM_KEYPOINTS * 3)
        resp = await client.patch(
            f"/api/sessions/{SESSION_ID}/frames/10/keypoints",
            json={"keypoints": new_kp},
        )

        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        frame = populated_store.get_frame(SESSION_ID, 10)
        assert frame.keypoints == new_kp
        assert frame.annotation_status == "reviewed"

    async def test_wrong_keypoints_length_returns_422(self, client) -> None:
        """Test: zła liczba keypoints → 422."""
        resp = await client.patch(
            f"/api/sessions/{SESSION_ID}/frames/10/keypoints",
            json={"keypoints": [1.0, 2.0, 3.0]},
        )
        assert resp.status_code == 422

    async def test_nonexistent_frame_returns_404(self, client) -> None:
        """Test: nieistniejąca klatka → 404."""
        resp = await client.patch(
            f"/api/sessions/{SESSION_ID}/frames/999/keypoints",
            json={"keypoints": [0.0] * (NUM_KEYPOINTS * 3)},
        )
        assert resp.status_code == 404


@pytest.mark.anyio
class TestUpdateAUs:
    """Testy PATCH /frames/{idx}/aus (DOG-S9-3)."""

    async def test_update_aus(self, client, populated_store) -> None:
        """Test aktualizacji Action Units klatki."""
        new_aus = {
            "AU101": {"ratio": 1.5, "delta": 0.5, "is_active": True, "confidence": 0.95}
        }
        resp = await client.patch(
            f"/api/sessions/{SESSION_ID}/frames/10/aus",
            json={"aus": new_aus},
        )

        assert resp.status_code == 200
        frame = populated_store.get_frame(SESSION_ID, 10)
        assert frame.aus["AU101"]["ratio"] == pytest.approx(1.5)
        assert frame.annotation_status == "reviewed"


@pytest.mark.anyio
class TestUpdateEmotion:
    """Testy PATCH /frames/{idx}/emotion (DOG-S9-4)."""

    async def test_update_valid_emotion(self, client, populated_store) -> None:
        """Test aktualizacji poprawnej emocji."""
        resp = await client.patch(
            f"/api/sessions/{SESSION_ID}/frames/10/emotion",
            json={"emotion": "sad", "emotion_confidence": 0.75},
        )

        assert resp.status_code == 200
        frame = populated_store.get_frame(SESSION_ID, 10)
        assert frame.emotion == "sad"
        assert frame.emotion_confidence == pytest.approx(0.75)

    async def test_invalid_emotion_returns_422(self, client) -> None:
        """Test: nieznana emocja → 422."""
        resp = await client.patch(
            f"/api/sessions/{SESSION_ID}/frames/10/emotion",
            json={"emotion": "nonexistent_emotion"},
        )
        assert resp.status_code == 422


@pytest.mark.anyio
class TestUpdateBreed:
    """Testy PATCH /frames/{idx}/breed (DOG-S9-5)."""

    async def test_update_breed(self, client, populated_store) -> None:
        """Test aktualizacji rasy psa."""
        resp = await client.patch(
            f"/api/sessions/{SESSION_ID}/frames/10/breed",
            json={"breed": "labrador", "breed_confidence": 0.9},
        )

        assert resp.status_code == 200
        frame = populated_store.get_frame(SESSION_ID, 10)
        assert frame.breed == "labrador"
        assert frame.breed_confidence == pytest.approx(0.9)
        assert frame.annotation_status == "reviewed"


# =============================================================================
# Testy POST recompute endpoints — DOG-S9-6..7
# =============================================================================


@pytest.mark.anyio
class TestRecomputeAUs:
    """Testy POST /frames/{idx}/recompute_aus (DOG-S9-6)."""

    async def test_recompute_aus_from_keypoints(self, client, populated_store) -> None:
        """Test przeliczania AU z keypoints (geometrycznie)."""
        resp = await client.post(
            f"/api/sessions/{SESSION_ID}/frames/10/recompute_aus"
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "aus" in data
        assert len(data["aus"]) > 0

        # Sprawdź że AU zostały zapisane w store
        frame = populated_store.get_frame(SESSION_ID, 10)
        assert len(frame.aus) > 0

    async def test_recompute_aus_no_keypoints_returns_422(
        self, client, populated_store
    ) -> None:
        """Test: brak keypoints w klatce → 422."""
        frame = populated_store.get_frame(SESSION_ID, 10)
        frame.keypoints = None
        populated_store.update_frame(SESSION_ID, frame)

        resp = await client.post(
            f"/api/sessions/{SESSION_ID}/frames/10/recompute_aus"
        )
        assert resp.status_code == 422


@pytest.mark.anyio
class TestRecomputeEmotion:
    """Testy POST /frames/{idx}/recompute_emotion (DOG-S9-7)."""

    async def test_recompute_emotion_from_aus(self, client) -> None:
        """Test przeliczania emocji z AU (rule-based)."""
        resp = await client.post(
            f"/api/sessions/{SESSION_ID}/frames/10/recompute_emotion"
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        valid_emotions = [
            "happy", "sad", "angry", "fearful", "relaxed",
            "neutral", "surprise", "pain", "submission",
        ]
        assert data["emotion"] in valid_emotions

    async def test_recompute_emotion_no_aus_returns_422(
        self, client, populated_store
    ) -> None:
        """Test: brak AU w klatce → 422."""
        frame = populated_store.get_frame(SESSION_ID, 10)
        frame.aus = {}
        populated_store.update_frame(SESSION_ID, frame)

        resp = await client.post(
            f"/api/sessions/{SESSION_ID}/frames/10/recompute_emotion"
        )
        assert resp.status_code == 422


# =============================================================================
# Testy add_frame — DOG-S9-10
# =============================================================================


@pytest.mark.anyio
class TestAddFrame:
    """Testy POST /add_frame (DOG-S9-10)."""

    async def test_add_manual_frame(self, client, populated_store) -> None:
        """Test dodania klatki manualnie."""
        resp = await client.post(
            f"/api/sessions/{SESSION_ID}/add_frame",
            json={"frame_idx": 42, "image_url": f"/static/{SESSION_ID}/frame_0042.jpg"},
        )

        assert resp.status_code == 200
        assert resp.json()["frame_idx"] == 42

        session = populated_store.load(SESSION_ID)
        indices = [f.frame_idx for f in session.frames]
        assert 42 in indices

    async def test_add_frame_sorted_by_idx(self, client, populated_store) -> None:
        """Test: klatki są posortowane po frame_idx."""
        await client.post(
            f"/api/sessions/{SESSION_ID}/add_frame",
            json={"frame_idx": 3, "image_url": f"/static/{SESSION_ID}/frame_0003.jpg"},
        )

        session = populated_store.load(SESSION_ID)
        indices = [f.frame_idx for f in session.frames]
        assert indices == sorted(indices)

    async def test_add_frame_to_nonexistent_session_returns_404(self, client) -> None:
        """Test: nieistniejąca sesja → 404."""
        resp = await client.post(
            "/api/sessions/nonexistent/add_frame",
            json={"frame_idx": 1, "image_url": "/static/x/frame_0001.jpg"},
        )
        assert resp.status_code == 404


# =============================================================================
# Testy export_coco — DOG-S9-9
# =============================================================================


@pytest.mark.anyio
class TestExportCoco:
    """Testy POST /export_coco (DOG-S9-9)."""

    async def test_export_returns_json(self, client) -> None:
        """Test: eksport zwraca JSON."""
        resp = await client.post(f"/api/sessions/{SESSION_ID}/export_coco")

        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]

    async def test_export_coco_valid_structure(self, client) -> None:
        """Test: wyeksportowany JSON ma strukturę COCO."""
        resp = await client.post(f"/api/sessions/{SESSION_ID}/export_coco")
        data = resp.json()

        assert "info" in data
        assert "categories" in data
        assert "images" in data
        assert "annotations" in data

        # Jedna klatka → jeden obraz i jedna anotacja
        assert len(data["images"]) == 1
        assert len(data["annotations"]) == 1

    async def test_export_coco_annotation_has_dogfacs_fields(self, client) -> None:
        """Test: anotacja zawiera pola DogFACS."""
        resp = await client.post(f"/api/sessions/{SESSION_ID}/export_coco")
        ann = resp.json()["annotations"][0]

        assert "emotion" in ann
        assert "aus" in ann
        assert "annotation_status" in ann
        assert "neutral_frame_id" in ann

    async def test_export_nonexistent_session_returns_404(self, client) -> None:
        """Test: nieistniejąca sesja → 404."""
        resp = await client.post("/api/sessions/nonexistent/export_coco")
        assert resp.status_code == 404
