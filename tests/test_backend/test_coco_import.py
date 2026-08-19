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
import shutil
from pathlib import Path

import httpx
import pytest
from coco_import import (
    IMPORT_ROOT_ENV,
    CocoImportError,
    build_session,
    load_coco,
    resolve_import_path,
)
from fastapi import FastAPI
from label_store import ANNOTATOR_ENV, LABELS_ROOT_ENV
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

    def test_blad_nie_wynosi_sciezki_do_klienta(self, tmp_path: Path) -> None:
        """Komunikat jedzie wprost do przeglądarki — nie może nieść układu dysku."""
        secret = tmp_path / "tajny_katalog" / "zly.json"
        secret.parent.mkdir()
        secret.write_text("{nie json", encoding="utf-8")
        with pytest.raises(CocoImportError) as caught:
            load_coco(secret)
        assert "tajny_katalog" not in str(caught.value)


class TestResolveImportPath:
    """
    Przycięcie ścieżki od klienta do katalogu danych.

    Ścieżka przychodzi z przeglądarki, więc bez tego endpoint importu czytałby
    dowolny plik z dysku serwera.
    """

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch) -> Path:
        """Katalog danych z jednym poprawnym zbiorem w środku."""
        data_root = tmp_path / "data"
        (data_root / "dataset_v2").mkdir(parents=True)
        (data_root / "dataset_v2" / "curated.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv(IMPORT_ROOT_ENV, str(data_root))
        return data_root

    def test_sciezka_wzgledna_pod_korzeniem_przechodzi(self, root: Path) -> None:
        resolved = resolve_import_path("dataset_v2/curated.json")
        assert resolved == (root / "dataset_v2" / "curated.json").resolve()

    def test_wyjscie_w_gore_odrzucone(self, root: Path, tmp_path: Path) -> None:
        (tmp_path / "sekret.json").write_text("{}", encoding="utf-8")
        with pytest.raises(CocoImportError, match="wychodzi poza"):
            resolve_import_path("../sekret.json")

    def test_sciezka_bezwzgledna_poza_korzeniem_odrzucona(
        self, root: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "sekret.json"
        outside.write_text("{}", encoding="utf-8")
        with pytest.raises(CocoImportError, match="wychodzi poza"):
            resolve_import_path(str(outside))

    def test_sciezka_bezwzgledna_pod_korzeniem_przechodzi(self, root: Path) -> None:
        inside = root / "dataset_v2" / "curated.json"
        assert resolve_import_path(str(inside)) == inside.resolve()

    def test_inne_rozszerzenie_odrzucone(self, root: Path) -> None:
        (root / "klucz.txt").write_text("sekret", encoding="utf-8")
        with pytest.raises(CocoImportError, match="json"):
            resolve_import_path("klucz.txt")

    def test_nieistniejacy_plik_odrzucony(self, root: Path) -> None:
        with pytest.raises(CocoImportError, match="Nie znaleziono"):
            resolve_import_path("dataset_v2/nie_ma.json")


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
def curated_path(tmp_path: Path, monkeypatch) -> Path:
    """
    Plik zbioru po kuracji, leżący pod korzeniem importu.

    Katalog etykiet też idzie do tmp: bez tego testy dopisywałyby werdykty
    do prawdziwego `data/labels/` w repozytorium i zaśmiecały pracę zespołu.
    """
    monkeypatch.setenv(IMPORT_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(LABELS_ROOT_ENV, str(tmp_path / "labels"))
    monkeypatch.setenv(ANNOTATOR_ENV, "test")
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

    async def test_brak_pliku_daje_400(self, client, curated_path: Path) -> None:
        resp = await client.post(
            "/api/sessions/import_coco", json={"path": "brak.json"}
        )
        assert resp.status_code == 400

    async def test_wyjscie_poza_katalog_danych_daje_400(
        self, client, curated_path: Path, tmp_path: Path
    ) -> None:
        """Ścieżka od klienta nie może sięgnąć poza katalog danych."""
        outside = tmp_path.parent / "sekret.json"
        outside.write_text("{}", encoding="utf-8")
        resp = await client.post(
            "/api/sessions/import_coco", json={"path": "../sekret.json"}
        )
        assert resp.status_code == 400

    async def test_limit_respektowany(self, client, curated_path: Path) -> None:
        resp = await client.post(
            "/api/sessions/import_coco", json={"path": str(curated_path), "limit": 1}
        )
        assert resp.json()["pairs"] == 1


@pytest.mark.anyio
class TestReviewEndpoint:
    """
    Jeden zapis obejmuje AU, keypoints, rasę i emocję.

    Osobne przejścia po zbiorze dla każdego pola znaczyłyby, że 518 par trzeba
    obejrzeć cztery razy. Gorzej: zły pomiar keypoints unieważnia etykiety AU
    tej klatki, więc bez oceny punktów w TYM SAMYM przejściu dowiadujemy się
    o problemie dopiero przy kolejnym obiegu całego materiału.
    """

    async def _import(self, client, curated_path: Path) -> str:
        resp = await client.post("/api/sessions/import_coco", json={"path": "curated.json"})
        return resp.json()["session_id"]

    async def test_zapisuje_wszystkie_pola_naraz(self, client, curated_path: Path) -> None:
        session_id = await self._import(client, curated_path)
        resp = await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={
                "verdicts": {"AU25": AU_VERDICT_ACTIVE},
                "usable": True,
                "keypoints_ok": True,
                "breed": "Beagle",
                "emotion": "happy",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["au_verdicts"]["AU25"] == AU_VERDICT_ACTIVE
        assert body["keypoints_ok"] is True
        assert body["breed"] == "Beagle"
        assert body["emotion"] == "happy"

    async def test_zle_keypoints_zapisuja_sie_jako_ostrzezenie(
        self, client, curated_path: Path
    ) -> None:
        """Bez tego pola nie da się odsiać etykiet AU zmierzonych na złych punktach."""
        session_id = await self._import(client, curated_path)
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {}, "usable": True, "keypoints_ok": False},
        )
        coco = json.loads((await client.post(f"/api/sessions/{session_id}/export_coco")).content)
        broken = [a for a in coco["annotations"] if a.get("keypoints_ok") is False]
        assert len(broken) == 1

    async def test_nieoceniony_stan_punktow_zostaje_pusty(
        self, client, curated_path: Path
    ) -> None:
        """None to brak wiedzy, nie „punkty dobre"."""
        session_id = await self._import(client, curated_path)
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {}, "usable": True},
        )
        session = await client.get(f"/api/sessions/{session_id}")
        peak = [f for f in session.json()["frames"] if f["frame_idx"] == 2][0]
        assert peak["keypoints_ok"] is None

    async def test_nieznana_emocja_odrzucona(self, client, curated_path: Path) -> None:
        session_id = await self._import(client, curated_path)
        resp = await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {}, "emotion": "wesolutki"},
        )
        assert resp.status_code == 422

    async def test_pominiete_pola_nie_kasuja_wartosci(self, client, curated_path: Path) -> None:
        """Brak rasy w żądaniu znaczy „zostaw", a nie „wyczyść"."""
        session_id = await self._import(client, curated_path)
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {}, "breed": "Beagle"},
        )
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {"AU26": AU_VERDICT_ACTIVE}},
        )
        session = await client.get(f"/api/sessions/{session_id}")
        peak = [f for f in session.json()["frames"] if f["frame_idx"] == 2][0]
        assert peak["breed"] == "Beagle"


@pytest.mark.anyio
class TestSessionReuse:
    """
    Ten sam zbiór to zawsze ta sama sesja.

    Losowy identyfikator znaczyłby, że każde otwarcie narzędzia zakłada nową
    sesję, a wczorajsza praca zostaje w osieroconej. Anotator wracałby do
    pierwszej pary i nie miał jak odzyskać swoich werdyktów — czyli dzień
    pracy grupy znikałby po zamknięciu przeglądarki.
    """

    async def test_drugi_import_podejmuje_te_sama_sesje(
        self, client, curated_path: Path
    ) -> None:
        first = await client.post("/api/sessions/import_coco", json={"path": "curated.json"})
        second = await client.post("/api/sessions/import_coco", json={"path": "curated.json"})
        assert first.json()["session_id"] == second.json()["session_id"]
        assert second.json()["resumed"] is True

    async def test_werdykty_przezywaja_ponowne_otwarcie(
        self, client, curated_path: Path
    ) -> None:
        """Sedno: praca ma się kumulować, a nie zaczynać od zera."""
        session_id = (
            await client.post("/api/sessions/import_coco", json={"path": "curated.json"})
        ).json()["session_id"]
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {"AU25": AU_VERDICT_ACTIVE}, "mark_verified": True},
        )

        again = await client.post("/api/sessions/import_coco", json={"path": "curated.json"})
        assert again.json()["verified"] == 1  # etykieta wczytana z pliku, nie z sesji

        frames = (await client.get(f"/api/sessions/{session_id}/frames")).json()["frames"]
        peak = [f for f in frames if f["frame_idx"] == 2][0]
        assert peak["au_verdicts"]["AU25"] == AU_VERDICT_ACTIVE

    async def test_praca_przezywa_skasowanie_sesji(self, client, curated_path: Path) -> None:
        """
        Sesja jest kasowalnym cache'em — źródłem prawdy są pliki etykiet.

        To dzięki temu `git pull` z werdyktami kolegi wystarcza, żeby zobaczyć
        jego postęp: sesja na dysku zna wyłącznie własną maszynę.
        """
        import routers.sessions as sessions_module

        session_id = (
            await client.post("/api/sessions/import_coco", json={"path": "curated.json"})
        ).json()["session_id"]
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {"AU25": AU_VERDICT_ACTIVE}, "keypoints_ok": True},
        )

        # Kasujemy sesję tak, jak zrobiłby to nowy klon repozytorium
        shutil.rmtree(sessions_module._store.sessions_dir / session_id)

        restored = await client.post("/api/sessions/import_coco", json={"path": "curated.json"})
        assert restored.json()["verified"] == 1
        frames = (await client.get(f"/api/sessions/{session_id}/frames")).json()["frames"]
        peak = [f for f in frames if f["frame_idx"] == 2][0]
        assert peak["au_verdicts"]["AU25"] == AU_VERDICT_ACTIVE
        assert peak["keypoints_ok"] is True


@pytest.mark.anyio
class TestDatasetListing:
    """Wyliczanie zbiorów gotowych do weryfikacji — anotator nie wpisuje ścieżek."""

    async def test_znajduje_zbior_po_kuracji(self, client, curated_path: Path) -> None:
        resp = await client.get("/api/sessions/datasets/available")
        assert resp.status_code == 200
        names = [d["path"] for d in resp.json()["datasets"]]
        assert "curated.json" in names

    async def test_podaje_liczbe_par(self, client, curated_path: Path) -> None:
        datasets = (await client.get("/api/sessions/datasets/available")).json()["datasets"]
        entry = [d for d in datasets if d["path"] == "curated.json"][0]
        assert entry["pairs"] == 2

    async def test_jeden_zbior_to_jedna_pozycja(self, client, curated_path: Path) -> None:
        """
        Materiał surowy i paczka robocza to TEN SAM zbiór — lista pokazuje jedną pozycję.

        U autora zbioru leżą oba: surowy `<zbior>/curated.json` na jego dysku
        i paczka `<zbior>/work/curated.json` z repozytorium. Dwie pozycje o tej
        samej nazwie zmuszały do zgadywania, którą otworzyć.
        """
        work = curated_path.parent / "work"
        (work / "frames").mkdir(parents=True)
        shutil.copy(curated_path, work / "curated.json")

        datasets = (await client.get("/api/sessions/datasets/available")).json()["datasets"]

        assert len(datasets) == 1
        assert datasets[0]["variant"] == "work", (
            "wybrana ma byc paczka — to ja widzi reszta zespolu"
        )

    async def test_pokazuje_postep_pracy(self, client, curated_path: Path) -> None:
        session_id = (
            await client.post("/api/sessions/import_coco", json={"path": "curated.json"})
        ).json()["session_id"]
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {"AU25": AU_VERDICT_ACTIVE}, "mark_verified": True},
        )
        datasets = (await client.get("/api/sessions/datasets/available")).json()["datasets"]
        entry = [d for d in datasets if d["path"] == "curated.json"][0]
        assert entry["verified"] == 1


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
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {"AU25": AU_VERDICT_ACTIVE}},
        )
        assert resp.status_code == 200
        assert resp.json()["au_verdicts"]["AU25"] == AU_VERDICT_ACTIVE

    async def test_werdykty_sie_dopisuja_a_nie_zastepuja(
        self, client, curated_path: Path
    ) -> None:
        """Anotator ocenia AU po kolei i nie może tracić wcześniejszych decyzji."""
        session_id = await self._import(client, curated_path)
        url = f"/api/sessions/{session_id}/frames/2/review?track_id=0"
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
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {"AU25": "moze_tak_moze_nie"}},
        )
        assert resp.status_code == 422

    async def test_oznaczenie_jako_zweryfikowane(self, client, curated_path: Path) -> None:
        session_id = await self._import(client, curated_path)
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {"AU25": AU_VERDICT_ACTIVE}, "mark_verified": True},
        )
        session = await client.get(f"/api/sessions/{session_id}")
        peak = [f for f in session.json()["frames"] if f["frame_idx"] == 2][0]
        assert peak["annotation_status"] == ANNOTATION_STATUS_VERIFIED

    async def test_odrzucenie_kadru_zapisuje_sie_jako_etykieta(
        self, client, curated_path: Path
    ) -> None:
        """
        Kadr odrzucony przez człowieka ZOSTAJE w zbiorze z flagą, a nie znika.

        Przegląd materiału pokazał, że żadna miara geometryczna nie odróżnia
        kadru nadającego się do kodowania AU od psa z głową w dół — asymetria
        jest miarą lewo-prawo i na pochylenie jest ślepa. Decyzja człowieka jest
        więc jedynym wiarygodnym filtrem, a zapisana staje się etykietą uczącą.
        """
        session_id = await self._import(client, curated_path)
        resp = await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={
                "verdicts": {"AU25": AU_VERDICT_NOT_OBSERVABLE},
                                "usable": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["usable"] is False

        coco = json.loads((await client.post(f"/api/sessions/{session_id}/export_coco")).content)
        rejected = [a for a in coco["annotations"] if a["usable"] is False]
        assert len(rejected) == 1
        assert rejected[0]["label_source"] == LABEL_SOURCE_HUMAN_VERIFIED

    async def test_kadr_domyslnie_jest_uznany_za_nadajacy_sie(
        self, client, curated_path: Path
    ) -> None:
        session_id = await self._import(client, curated_path)
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
            json={"verdicts": {"AU25": AU_VERDICT_ACTIVE}, "mark_verified": True},
        )
        session = await client.get(f"/api/sessions/{session_id}")
        peak = [f for f in session.json()["frames"] if f["frame_idx"] == 2][0]
        assert peak["usable"] is True

    async def test_eksport_niesie_werdykty_i_zrodlo_etykiety(
        self, client, curated_path: Path
    ) -> None:
        """Bez `human_verified` Sprint 16 nie odsieje etykiet ludzkich od reguł."""
        session_id = await self._import(client, curated_path)
        await client.patch(
            f"/api/sessions/{session_id}/frames/2/review?track_id=0",
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
