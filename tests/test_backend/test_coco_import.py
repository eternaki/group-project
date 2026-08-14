"""
Testy importu zbioru COCO do sesji anotacji i werdyktów człowieka o AU.

Najważniejsze, czego pilnują te testy, to rozdział dwóch rzeczy, które łatwo
zlać w jedno: POMIARU reguł (`aus`) i WERDYKTU człowieka (`au_verdicts`).
Gdyby werdykt startował od wartości reguły, anotator zatwierdzałby błąd jednym
kliknięciem, a zbiór odziedziczyłby dokładnie te usterki, przed którymi ma
chronić weryfikacja.

Uruchomienie:
    pytest tests/test_backend/test_coco_import.py -v
"""

import json
from pathlib import Path

import httpx
import pytest
from coco_import import CocoImportError, build_session, load_coco
from fastapi import FastAPI
from session_store import (
    ANNOTATION_STATUS_AUTO,
    ANNOTATION_STATUS_VERIFIED,
    AU_VERDICT_ACTIVE,
    AU_VERDICT_NOT_OBSERVABLE,
    SessionStore,
)

from packages.data.coco import (
    FRAME_ROLE_NEUTRAL,
    FRAME_ROLE_PEAK,
    LABEL_SOURCE_HUMAN_VERIFIED,
)
from packages.data.schemas import NUM_KEYPOINTS

_KEYPOINTS: list[float] = [float(i % 5) for i in range(NUM_KEYPOINTS * 3)]


def _annotation(
    annotation_id: int,
    image_id: int,
    role: str,
    review_order: int,
    neutral_frame_id: int,
) -> dict:
    """Buduje jedną anotację COCO po kuracji."""
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "track_id": 0,
        "frame_role": role,
        "review_order": review_order,
        "neutral_frame_id": neutral_frame_id,
        "neutral_source": "auto",
        "bbox": [10.0, 20.0, 100.0, 120.0],
        "keypoints": _KEYPOINTS,
        "emotion": "neutral",
        "emotion_rule_applied": "brak_aktywnych_au",
        "breed": "Beagle",
        "confidence": {"emotion": 0.8, "breed": 0.33},
        "tfm_score": 3.5,
        "au_analysis": {"AU25": {"ratio": 1.4, "is_active": True, "confidence": 0.7}},
        "au_noise": {"AU25": 0.2},
        "au_sample_count": {"AU25": 6},
        "quality": {"asymmetry": 0.11, "weak_keypoint_ratio": 0.02, "face_width_px": 120.0},
    }


def make_curated_coco(pair_count: int = 2) -> dict:
    """
    Buduje zbiór COCO po kuracji z zadaną liczbą par.

    Args:
        pair_count: Ile par (neutralna, szczytowa) ma zawierać zbiór

    Returns:
        Słownik COCO
    """
    images: list[dict] = []
    annotations: list[dict] = []
    for order in range(pair_count):
        neutral_id, peak_id = 2 * order + 1, 2 * order + 2
        for image_id, role in ((neutral_id, FRAME_ROLE_NEUTRAL), (peak_id, FRAME_ROLE_PEAK)):
            images.append(
                {"id": image_id, "file_name": f"VIDEO_{order}/frame_{image_id:04d}.jpg"}
            )
            annotations.append(_annotation(image_id, image_id, role, order, neutral_id))
    return {"info": {}, "licenses": [], "categories": [], "images": images, "annotations": annotations}


class TestBuildSession:
    """Składanie sesji ze zbioru po kuracji."""

    def test_kazda_para_dostaje_wlasny_track(self) -> None:
        session = build_session(make_curated_coco(3), "abc12345", "curated.json")
        assert len(session.dogs) == 3
        assert [dog.track_id for dog in session.dogs] == [0, 1, 2]

    def test_para_wnosi_dwie_klatki(self) -> None:
        session = build_session(make_curated_coco(3), "abc12345", "curated.json")
        assert len(session.frames) == 6

    def test_klatka_neutralna_jest_baza_swojej_pary(self) -> None:
        """Bez własnej klatki neutralnej para nie ma względem czego mierzyć AU."""
        session = build_session(make_curated_coco(2), "abc12345", "curated.json")
        for dog in session.dogs:
            neutrals = [
                frame
                for frame in session.frames
                if frame.track_id == dog.track_id
                and frame.frame_role == FRAME_ROLE_NEUTRAL
            ]
            assert len(neutrals) == 1
            assert neutrals[0].frame_idx == dog.neutral_frame_idx

    def test_werdykty_startuja_puste(self) -> None:
        """Autometki NIE wypełniają odpowiedzi człowieka — to sedno weryfikacji."""
        session = build_session(make_curated_coco(2), "abc12345", "curated.json")
        assert all(frame.au_verdicts == {} for frame in session.frames)

    def test_pomiar_regul_jedzie_jako_podpowiedz(self) -> None:
        session = build_session(make_curated_coco(1), "abc12345", "curated.json")
        assert session.frames[0].aus["AU25"]["ratio"] == pytest.approx(1.4)

    def test_status_startuje_jako_auto(self) -> None:
        session = build_session(make_curated_coco(1), "abc12345", "curated.json")
        assert all(f.annotation_status == ANNOTATION_STATUS_AUTO for f in session.frames)

    def test_miary_jakosci_trafiaja_do_sesji(self) -> None:
        session = build_session(make_curated_coco(1), "abc12345", "curated.json")
        assert session.frames[0].quality["asymmetry"] == pytest.approx(0.11)

    def test_szum_au_ladowany_razem_z_liczba_prob(self) -> None:
        """Sigma bez liczby pomiarów jest nie do zważenia w treningu."""
        session = build_session(make_curated_coco(1), "abc12345", "curated.json")
        dog = session.dogs[0]
        assert dog.au_noise == {"AU25": 0.2}
        assert dog.au_sample_count == {"AU25": 6}

    def test_limit_przycina_liczbe_par(self) -> None:
        session = build_session(make_curated_coco(5), "abc12345", "curated.json", limit=2)
        assert len(session.dogs) == 2

    def test_url_klatki_wskazuje_na_zbior(self) -> None:
        session = build_session(make_curated_coco(1), "abc12345", "curated.json")
        assert session.frames[0].image_url.startswith("/dataset/")

    def test_zbior_bez_kuracji_odrzucony(self) -> None:
        coco = make_curated_coco(1)
        for annotation in coco["annotations"]:
            del annotation["review_order"]
        with pytest.raises(CocoImportError, match="review_order"):
            build_session(coco, "abc12345", "curated.json")

    def test_niepelna_para_odrzucona(self) -> None:
        coco = make_curated_coco(1)
        coco["annotations"] = [
            a for a in coco["annotations"] if a["frame_role"] == FRAME_ROLE_PEAK
        ]
        with pytest.raises(CocoImportError, match="pary"):
            build_session(coco, "abc12345", "curated.json")


class TestLoadCoco:
    """Wczytywanie zbioru z dysku."""

    def test_brak_pliku_daje_czytelny_blad(self, tmp_path: Path) -> None:
        with pytest.raises(CocoImportError, match="Nie znaleziono"):
            load_coco(tmp_path / "nie_ma.json")

    def test_zly_json_daje_czytelny_blad(self, tmp_path: Path) -> None:
        path = tmp_path / "zly.json"
        path.write_text("{nie json", encoding="utf-8")
        with pytest.raises(CocoImportError, match="JSON"):
            load_coco(path)


# =============================================================================
# Testy API — import i werdykty
# =============================================================================


@pytest.fixture
def temp_store(tmp_path: Path) -> SessionStore:
    """Izolowany SessionStore."""
    return SessionStore(sessions_dir=tmp_path / "sessions")


@pytest.fixture
def app(temp_store: SessionStore, monkeypatch) -> FastAPI:
    """FastAPI z routerem sesji i podmienionym magazynem."""
    import routers.sessions as sessions_module

    monkeypatch.setattr(sessions_module, "_store", temp_store)
    test_app = FastAPI()
    test_app.include_router(sessions_module.router)
    return test_app


@pytest.fixture
async def client(app: FastAPI):
    """Asynchroniczny klient HTTP."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


@pytest.fixture
def curated_path(tmp_path: Path) -> Path:
    """Plik zbioru po kuracji na dysku."""
    path = tmp_path / "curated.json"
    path.write_text(json.dumps(make_curated_coco(2)), encoding="utf-8")
    return path


@pytest.mark.anyio
class TestImportCocoEndpoint:
    """POST /api/sessions/import_coco."""

    async def test_import_tworzy_sesje(self, client, curated_path: Path) -> None:
        resp = await client.post(
            "/api/sessions/import_coco", json={"path": str(curated_path)}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pairs"] == 2
        assert data["frames"] == 4

    async def test_zaimportowana_sesja_da_sie_odczytac(self, client, curated_path: Path) -> None:
        created = await client.post(
            "/api/sessions/import_coco", json={"path": str(curated_path)}
        )
        session_id = created.json()["session_id"]
        resp = await client.get(f"/api/sessions/{session_id}/frames")
        assert resp.status_code == 200
        assert len(resp.json()["frames"]) == 4

    async def test_brak_pliku_daje_400(self, client, tmp_path: Path) -> None:
        resp = await client.post(
            "/api/sessions/import_coco", json={"path": str(tmp_path / "brak.json")}
        )
        assert resp.status_code == 400

    async def test_limit_respektowany(self, client, curated_path: Path) -> None:
        resp = await client.post(
            "/api/sessions/import_coco", json={"path": str(curated_path), "limit": 1}
        )
        assert resp.json()["pairs"] == 1


@pytest.mark.anyio
class TestAUVerdictsEndpoint:
    """PATCH /api/sessions/{id}/frames/{idx}/au_verdicts."""

    async def _import(self, client, curated_path: Path) -> str:
        resp = await client.post(
            "/api/sessions/import_coco", json={"path": str(curated_path)}
        )
        return resp.json()["session_id"]

    async def test_zapisuje_werdykt(self, client, curated_path: Path) -> None:
        session_id = await self._import(client, curated_path)
        resp = await client.patch(
            f"/api/sessions/{session_id}/frames/2/au_verdicts?track_id=0",
            json={"verdicts": {"AU25": AU_VERDICT_ACTIVE}},
        )
        assert resp.status_code == 200
        assert resp.json()["au_verdicts"]["AU25"] == AU_VERDICT_ACTIVE

    async def test_werdykty_sie_dopisuja_a_nie_zastepuja(
        self, client, curated_path: Path
    ) -> None:
        """Anotator ocenia AU po kolei i nie może tracić wcześniejszych decyzji."""
        session_id = await self._import(client, curated_path)
        url = f"/api/sessions/{session_id}/frames/2/au_verdicts?track_id=0"
        await client.patch(url, json={"verdicts": {"AU25": AU_VERDICT_ACTIVE}})
        resp = await client.patch(
            url, json={"verdicts": {"EAD103": AU_VERDICT_NOT_OBSERVABLE}}
        )
        verdicts = resp.json()["au_verdicts"]
        assert verdicts["AU25"] == AU_VERDICT_ACTIVE
        assert verdicts["EAD103"] == AU_VERDICT_NOT_OBSERVABLE

    async def test_nieznany_werdykt_odrzucony(self, client, curated_path: Path) -> None:
        session_id = await self._import(client, curated_path)
        resp = await client.patch(
            f"/api/sessions/{session_id}/frames/2/au_verdicts?track_id=0",
            json={"verdicts": {"AU25": "moze_tak_moze_nie"}},
        )
        assert resp.status_code == 422

    async def test_oznaczenie_jako_zweryfikowane(self, client, curated_path: Path) -> None:
        session_id = await self._import(client, curated_path)
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/au_verdicts?track_id=0",
            json={"verdicts": {"AU25": AU_VERDICT_ACTIVE}, "mark_verified": True},
        )
        session = await client.get(f"/api/sessions/{session_id}")
        peak = [f for f in session.json()["frames"] if f["frame_idx"] == 2][0]
        assert peak["annotation_status"] == ANNOTATION_STATUS_VERIFIED

    async def test_eksport_niesie_werdykty_i_zrodlo_etykiety(
        self, client, curated_path: Path
    ) -> None:
        """Bez `human_verified` Sprint 16 nie odsieje etykiet ludzkich od reguł."""
        session_id = await self._import(client, curated_path)
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/au_verdicts?track_id=0",
            json={"verdicts": {"AU25": AU_VERDICT_ACTIVE}, "mark_verified": True},
        )
        resp = await client.post(f"/api/sessions/{session_id}/export_coco")
        assert resp.status_code == 200
        coco = json.loads(resp.content)
        verified = [
            a for a in coco["annotations"] if a["au_verdicts"].get("AU25") == AU_VERDICT_ACTIVE
        ]
        assert len(verified) == 1
        assert verified[0]["label_source"] == LABEL_SOURCE_HUMAN_VERIFIED
