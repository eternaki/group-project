"""
Router dla zarządzania sesjami anotacji (Sprint 9 — DOG-S9-1..10).

Endpoints:
    GET  /api/sessions/{id}                              — pobierz sesję
    GET  /api/sessions/{id}/frames                       — lista klatek
    PATCH /api/sessions/{id}/frames/{idx}/keypoints      — edytuj keypoints
    PATCH /api/sessions/{id}/frames/{idx}/aus            — edytuj AU
    PATCH /api/sessions/{id}/frames/{idx}/emotion        — edytuj emocję
    PATCH /api/sessions/{id}/frames/{idx}/breed          — edytuj rasę
    POST  /api/sessions/{id}/frames/{idx}/recompute_aus  — przelicz AU z keypoints
    POST  /api/sessions/{id}/frames/{idx}/recompute_emotion — przelicz emocję z AU
    POST  /api/sessions/{id}/export_coco                 — eksport do COCO JSON
    POST  /api/sessions/{id}/add_frame                   — dodaj klatkę manualnie
"""

import json
import tempfile
import uuid
from dataclasses import asdict
from typing import Optional

import numpy as np
from coco_import import (
    SESSION_ID_LENGTH,
    CocoImportError,
    build_session,
    load_coco,
    resolve_import_path,
)
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from session_store import (
    ANNOTATION_STATUS_REVIEWED,
    ANNOTATION_STATUS_VERIFIED,
    AU_VERDICTS,
    AmbiguousFrameError,
    DogTrack,
    FrameAnnotation,
    FrameNotFoundError,
    SessionData,
    SessionNotFoundError,
    SessionStore,
    delta_aus_to_dict,
)

from packages.data.coco import (
    FRAME_ROLE_NEUTRAL,
    FRAME_ROLE_PEAK,
    LABEL_SOURCE_AUTO_RULES,
    LABEL_SOURCE_HUMAN_VERIFIED,
)
from packages.data.schemas import (
    EMOTION_CLASSES,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    SKELETON_CONNECTIONS,
)
from packages.models.delta_action_units import DeltaActionUnit, DeltaActionUnitsExtractor
from packages.models.emotion import classify_emotion_from_delta_aus

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
_store = SessionStore()


# =============================================================================
# Pydantic modele requestów
# =============================================================================


class UpdateKeypointsRequest(BaseModel):
    """Request dla PATCH keypoints."""

    keypoints: list[float]  # 138 wartości (46 × [x, y, visibility])


class AUData(BaseModel):
    """Dane jednego Action Unit."""

    ratio: float
    delta: float
    is_active: bool
    confidence: float


class UpdateAUsRequest(BaseModel):
    """Request dla PATCH aus."""

    aus: dict[str, AUData]


class UpdateEmotionRequest(BaseModel):
    """Request dla PATCH emotion."""

    emotion: str
    emotion_confidence: float = 0.0
    emotion_rule_applied: Optional[str] = None


class UpdateBreedRequest(BaseModel):
    """Request dla PATCH breed."""

    breed: str
    breed_confidence: float = 0.0


class AddFrameRequest(BaseModel):
    """Request dla POST add_frame."""

    frame_idx: int
    image_url: str
    source: str = "manual"
    track_id: Optional[int] = None


class ImportCocoRequest(BaseModel):
    """
    Request dla POST import_coco.

    Attributes:
        path: Ścieżka do zbioru po kuracji (`curate_for_review.py`)
        limit: Najwyżej tyle par; None znaczy wszystkie
    """

    path: str
    limit: Optional[int] = None


class UpdateAUVerdictsRequest(BaseModel):
    """
    Request dla PATCH au_verdicts — werdykty człowieka o AU.

    Attributes:
        verdicts: {nazwa AU: active | inactive | not_observable}. Klucze
            pominięte zostają nieocenione, a nie wyzerowane.
        mark_verified: Czy oznaczyć klatkę jako sprawdzoną przez człowieka
        usable: Czy kadr w ogóle nadaje się do kodowania AU. False znaczy, że
            człowiek odrzucił kadr — i to też jest etykieta, a nie brak danych.
    """

    verdicts: dict[str, str]
    mark_verified: bool = False
    usable: bool = True


# =============================================================================
# Funkcje pomocnicze
# =============================================================================


def _load_session_or_404(session_id: str):
    """Wczytuje sesję lub rzuca HTTP 404."""
    try:
        return _store.load(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Sesja {session_id!r} nie istnieje")


def _get_frame_or_404(
    session_id: str,
    frame_idx: int,
    track_id: Optional[int] = None,
) -> FrameAnnotation:
    """
    Wczytuje anotację klatki wskazanego psa lub zwraca błąd HTTP.

    Args:
        session_id: ID sesji
        frame_idx: Numer klatki
        track_id: Który pies w klatce (wymagany, gdy jest ich kilku)

    Returns:
        FrameAnnotation

    Raises:
        HTTPException: 404 gdy nie ma sesji/klatki, 409 gdy klatka opisuje
            kilku psów, a `track_id` nie wskazano
    """
    try:
        return _store.get_frame(session_id, frame_idx, track_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Sesja {session_id!r} nie istnieje")
    except AmbiguousFrameError as error:
        # 409, nie 404: klatka istnieje, ale żądanie nie mówi, którego psa dotyczy.
        # Ciche wybranie pierwszego nadpisałoby anotację niewłaściwego psa.
        raise HTTPException(status_code=409, detail=str(error.args[0]))
    except FrameNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Klatka {frame_idx} nie istnieje w sesji {session_id!r}"
        )


def _dict_to_delta_aus(aus_dict: dict) -> dict[str, DeltaActionUnit]:
    """Rekonstruuje DeltaActionUnit ze słownika zapisanego w JSON."""
    return {
        name: DeltaActionUnit(
            name=name,
            ratio=data["ratio"],
            delta=data["delta"],
            is_active=data["is_active"],
            confidence=data["confidence"],
        )
        for name, data in aus_dict.items()
    }


# =============================================================================
# POST import_coco — wprowadzenie zbioru wsadowego pod ręce anotatora
# =============================================================================


@router.post("/import_coco")
async def import_coco(request: ImportCocoRequest):
    """
    Tworzy sesję weryfikacji ze zbioru COCO po kuracji.

    Bez tego wejścia weryfikacja zbioru policzonego wsadowo wymagałaby
    ponownego przepuszczenia tych samych nagrań przez pipeline.

    Ścieżka przychodzi z przeglądarki, więc `resolve_import_path` przycina ją
    do katalogu danych — inaczej endpoint czytałby dowolny plik z dysku serwera.

    Args:
        request: Ścieżka do zbioru (względem katalogu danych) i limit par

    Returns:
        `{session_id, pairs, frames, source}`

    Raises:
        HTTPException: 400, gdy ścieżka wychodzi poza katalog danych, zbiór nie
            istnieje albo nie przeszedł kuracji
    """
    try:
        path = resolve_import_path(request.path)
        coco = load_coco(path)
        session = build_session(
            coco=coco,
            session_id=uuid.uuid4().hex[:SESSION_ID_LENGTH],
            source_name=path.name,
            limit=request.limit,
        )
    except CocoImportError as error:
        raise HTTPException(status_code=400, detail=str(error))

    _store.save(session)
    return {
        "session_id": session.session_id,
        "pairs": len(session.dogs),
        "frames": len(session.frames),
        "source": session.video_filename,
    }


# =============================================================================
# GET endpoints — DOG-S9-8
# =============================================================================


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Zwraca pełne dane sesji z wszystkimi anotacjami."""
    session = _load_session_or_404(session_id)
    return asdict(session)


@router.get("/{session_id}/frames")
async def list_frames(session_id: str, track_id: Optional[int] = None):
    """
    Zwraca klatki z anotacjami — wszystkie albo tylko jednego psa.

    Args:
        session_id: ID sesji
        track_id: Gdy podany, zwraca klatki wyłącznie tego psa

    Returns:
        `{"frames": [...], "dogs": [...], "rejected_tracks": [...]}` — odrzucone
        treki jadą razem z klatkami, żeby anotator wiedział, dlaczego pies zniknął
    """
    session = _load_session_or_404(session_id)
    frames = [f for f in session.frames if track_id is None or f.track_id == track_id]
    return {
        "frames": [asdict(f) for f in frames],
        "dogs": [asdict(dog) for dog in session.dogs],
        "rejected_tracks": [asdict(track) for track in session.rejected_tracks],
    }


# =============================================================================
# PATCH endpoints — DOG-S9-2..5
# =============================================================================


@router.patch("/{session_id}/frames/{frame_idx}/keypoints")
async def update_keypoints(
    session_id: str,
    frame_idx: int,
    request: UpdateKeypointsRequest,
    track_id: Optional[int] = None,
):
    """Aktualizuje keypoints klatki wskazanego psa (DOG-S9-2)."""
    expected = NUM_KEYPOINTS * 3
    if len(request.keypoints) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Oczekiwano {expected} wartości keypoints, otrzymano {len(request.keypoints)}",
        )
    frame = _get_frame_or_404(session_id, frame_idx, track_id)
    frame.keypoints = request.keypoints
    frame.annotation_status = "reviewed"
    _store.update_frame(session_id, frame)
    return {"ok": True}


@router.patch("/{session_id}/frames/{frame_idx}/au_verdicts")
async def update_au_verdicts(
    session_id: str,
    frame_idx: int,
    request: UpdateAUVerdictsRequest,
    track_id: Optional[int] = None,
):
    """
    Zapisuje werdykty człowieka o AU tej klatki.

    Werdykty dopisują się do już zapisanych, a nie zastępują ich w całości:
    anotator ocenia AU po kolei i nie powinien tracić wcześniejszych decyzji,
    gdy zapisze kolejną.

    Args:
        session_id: ID sesji
        frame_idx: Numer klatki
        request: Werdykty i flaga oznaczenia klatki jako sprawdzonej
        track_id: Który pies w klatce

    Returns:
        `{"ok": True, "au_verdicts": {...}}`

    Raises:
        HTTPException: 422 przy nieznanym werdykcie
    """
    unknown = sorted(set(request.verdicts.values()) - AU_VERDICTS)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Nieznany werdykt: {unknown}. Dozwolone: {sorted(AU_VERDICTS)}",
        )

    frame = _get_frame_or_404(session_id, frame_idx, track_id)
    frame.au_verdicts = {**frame.au_verdicts, **request.verdicts}
    frame.usable = request.usable
    frame.annotation_status = (
        ANNOTATION_STATUS_VERIFIED if request.mark_verified else ANNOTATION_STATUS_REVIEWED
    )
    frame.source = "manual"
    _store.update_frame(session_id, frame)
    return {"ok": True, "au_verdicts": frame.au_verdicts, "usable": frame.usable}


@router.patch("/{session_id}/frames/{frame_idx}/aus")
async def update_aus(
    session_id: str,
    frame_idx: int,
    request: UpdateAUsRequest,
    track_id: Optional[int] = None,
):
    """Aktualizuje Action Units klatki wskazanego psa (DOG-S9-3)."""
    frame = _get_frame_or_404(session_id, frame_idx, track_id)
    frame.aus = {name: au.model_dump() for name, au in request.aus.items()}
    frame.annotation_status = "reviewed"
    _store.update_frame(session_id, frame)
    return {"ok": True}


@router.patch("/{session_id}/frames/{frame_idx}/emotion")
async def update_emotion(
    session_id: str,
    frame_idx: int,
    request: UpdateEmotionRequest,
    track_id: Optional[int] = None,
):
    """Aktualizuje emocję klatki wskazanego psa (DOG-S9-4)."""
    if request.emotion not in EMOTION_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"Nieznana emocja {request.emotion!r}. Dozwolone: {EMOTION_CLASSES}",
        )
    frame = _get_frame_or_404(session_id, frame_idx, track_id)
    frame.emotion = request.emotion
    frame.emotion_confidence = request.emotion_confidence
    frame.emotion_rule_applied = request.emotion_rule_applied
    frame.annotation_status = "reviewed"
    _store.update_frame(session_id, frame)
    return {"ok": True}


@router.patch("/{session_id}/frames/{frame_idx}/breed")
async def update_breed(
    session_id: str,
    frame_idx: int,
    request: UpdateBreedRequest,
    track_id: Optional[int] = None,
):
    """Aktualizuje rasę wskazanego psa na klatce (DOG-S9-5)."""
    frame = _get_frame_or_404(session_id, frame_idx, track_id)
    frame.breed = request.breed
    frame.breed_confidence = request.breed_confidence
    frame.annotation_status = "reviewed"
    _store.update_frame(session_id, frame)
    return {"ok": True}


# =============================================================================
# POST recompute endpoints — DOG-S9-6..7
# =============================================================================


@router.post("/{session_id}/frames/{frame_idx}/recompute_aus")
async def recompute_aus(session_id: str, frame_idx: int, track_id: Optional[int] = None):
    """
    Przelicza AU z keypoints klatki i klatki neutralnej TEGO psa (DOG-S9-6).

    Baza AU jest per pies — liczenie delty względem klatki neutralnej innego psa
    dałoby aktywacje wynikające z różnicy budowy pysków, nie z mimiki.
    """
    session = _load_session_or_404(session_id)

    frame = _get_frame_or_404(session_id, frame_idx, track_id)
    if frame.keypoints is None:
        raise HTTPException(status_code=422, detail="Klatka nie ma keypoints")

    neutral = session.neutral_keypoints_for(frame.track_id)
    if neutral is None:
        raise HTTPException(status_code=422, detail="Sesja nie ma neutral keypoints")

    keypoints = np.array(frame.keypoints, dtype=np.float32)
    neutral_kp = np.array(neutral, dtype=np.float32)

    extractor = DeltaActionUnitsExtractor(neutral_kp)
    delta_aus = extractor.extract(keypoints)

    frame.aus = delta_aus_to_dict(delta_aus)
    _store.update_frame(session_id, frame)
    return {"ok": True, "aus": frame.aus}


@router.post("/{session_id}/frames/{frame_idx}/recompute_emotion")
async def recompute_emotion(session_id: str, frame_idx: int, track_id: Optional[int] = None):
    """Przelicza emocję z AU klatki wskazanego psa (DOG-S9-7)."""
    frame = _get_frame_or_404(session_id, frame_idx, track_id)
    if not frame.aus:
        raise HTTPException(status_code=422, detail="Klatka nie ma AU")

    delta_aus = _dict_to_delta_aus(frame.aus)
    prediction = classify_emotion_from_delta_aus(delta_aus)

    frame.emotion = prediction.emotion
    frame.emotion_confidence = float(prediction.confidence)
    frame.emotion_rule_applied = prediction.rule_applied
    _store.update_frame(session_id, frame)

    return {
        "ok": True,
        "emotion": frame.emotion,
        "emotion_confidence": frame.emotion_confidence,
        "emotion_rule_applied": frame.emotion_rule_applied,
    }


# =============================================================================
# POST add_frame — DOG-S9-10
# =============================================================================


@router.post("/{session_id}/add_frame")
async def add_frame(session_id: str, request: AddFrameRequest):
    """Dodaje klatkę manualnie do sesji (DOG-S9-10)."""
    _load_session_or_404(session_id)  # sprawdź czy sesja istnieje
    frame = FrameAnnotation(
        frame_idx=request.frame_idx,
        image_url=request.image_url,
        track_id=request.track_id,
        source=request.source,
        annotation_status="auto",
    )
    _store.add_frame(session_id, frame)
    return {"ok": True, "frame_idx": request.frame_idx, "track_id": request.track_id}


# =============================================================================
# POST export_coco — DOG-S9-9
# =============================================================================


def _count_visible_keypoints(keypoints: list[float]) -> int:
    """Liczy widoczne keypoints (visibility > 0.3)."""
    return sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0.3)


def _neutral_frame_id(session, track_id: Optional[int]) -> int:
    """
    Zwraca numer klatki neutralnej psa, którego dotyczy anotacja.

    Args:
        session: Dane sesji (`SessionData`)
        track_id: Identyfikator psa albo None (sesja sprzed obsługi wielu psów)

    Returns:
        Numer klatki neutralnej tego psa; gdy psa nie ma w `dogs` — wartość sesyjna
    """
    for dog in session.dogs:
        if dog.track_id == track_id:
            return dog.neutral_frame_idx
    return session.neutral_frame_idx


def _dog_for(session: SessionData, track_id: Optional[int]) -> Optional[DogTrack]:
    """Zwraca trek o podanym identyfikatorze albo None (sesja sprzed wielu psów)."""
    for dog in session.dogs:
        if dog.track_id == track_id:
            return dog
    return None


def _track_fields_for_export(
    frame: FrameAnnotation,
    dog: Optional[DogTrack],
) -> dict:
    """
    Pola treku, bez których zbiór z webappu nie daje się scalić ze zbiorem z batcha.

    Sprint 15 (weryfikacja ręczna) produkuje anotacje, które Sprint 16 ma połączyć
    z pre-etykietami z batch annotation. Bez `label_source` nie da się odsiać jednych
    od drugich, bez `au_noise`/`au_sample_count` nie da się ich zważyć, a bez
    `frame_role` nie wiadomo, która klatka jest bazą AU swojego treku.

    Args:
        frame: Anotacja klatki z sesji
        dog: Trek psa, którego dotyczy klatka (None dla sesji sprzed obsługi wielu psów)

    Returns:
        Słownik pól treku gotowy do dopisania w anotacji COCO
    """
    fields: dict = {
        "label_source": (
            LABEL_SOURCE_HUMAN_VERIFIED
            if frame.annotation_status == ANNOTATION_STATUS_VERIFIED
            else LABEL_SOURCE_AUTO_RULES
        ),
    }
    if dog is None:
        return fields

    # Rola zapisana wprost (sesje z importu COCO) jest wiarygodniejsza niż
    # wywnioskowana z numeru klatki — ten sam kadr neutralny bywa bazą kilku par.
    fields["frame_role"] = frame.frame_role or (
        FRAME_ROLE_NEUTRAL if frame.frame_idx == dog.neutral_frame_idx else FRAME_ROLE_PEAK
    )
    fields["neutral_source"] = dog.neutral_source
    fields["au_noise"] = dict(dog.au_noise)
    fields["au_sample_count"] = dict(dog.au_sample_count)
    return fields


def _build_coco_annotation(
    frame: FrameAnnotation,
    annotation_id: int,
    image_id: int,
    neutral_frame_id: int,
    dog: Optional[DogTrack] = None,
) -> dict:
    """Buduje obiekt anotacji COCO dla jednej klatki."""
    ann: dict = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        # Bez track_id anotacje dwóch psów z tej samej klatki są nieodróżnialne
        "track_id": frame.track_id,
        "annotation_status": frame.annotation_status,
        "source": frame.source,
        "neutral_frame_id": neutral_frame_id,
        "emotion": frame.emotion,
        "emotion_confidence": frame.emotion_confidence,
        "emotion_rule_applied": frame.emotion_rule_applied,
        "breed": frame.breed,
        "breed_confidence": frame.breed_confidence,
        "tfm_score": frame.tfm_score,
        # Ten sam format co w batch annotation (packages.data.coco): ratio nie
        # odróżnia realnej aktywacji od klamrowanego pomiaru, więc zapisujemy komplet.
        "au_analysis": {
            name: {
                "ratio": au.get("ratio", 0.0),
                "is_active": au.get("is_active", False),
                "confidence": au.get("confidence", 0.0),
            }
            for name, au in (frame.aus or {}).items()
        },
        # Werdykty człowieka OSOBNO od pomiaru reguł. Scalenie ich w jedno pole
        # zatarłoby różnicę między „człowiek orzekł, że nieaktywne" a „reguła
        # nie wykryła aktywacji" — a tylko pierwsze jest etykietą uczącą.
        "au_verdicts": dict(frame.au_verdicts or {}),
        # Kadr odrzucony przez człowieka zostaje w zbiorze z tą flagą, a nie
        # znika: „człowiek uznał to za nienadające się" jest etykietą uczącą
        # dla przyszłego filtra jakości, a milczenie nią nie jest.
        "usable": bool(frame.usable),
    }
    if frame.quality:
        ann["quality"] = dict(frame.quality)
    if frame.bbox is not None:
        ann["bbox"] = frame.bbox
        ann["area"] = frame.bbox[2] * frame.bbox[3]
        ann["iscrowd"] = 0
    if frame.keypoints is not None:
        ann["keypoints"] = frame.keypoints
        ann["num_keypoints"] = _count_visible_keypoints(frame.keypoints)
    ann.update(_track_fields_for_export(frame, dog))
    return ann


@router.post("/{session_id}/export_coco")
async def export_coco(session_id: str):
    """Eksportuje sesję do formatu COCO JSON (DOG-S9-9)."""
    from datetime import datetime

    session = _load_session_or_404(session_id)

    coco: dict = {
        "info": {
            "description": "DogFACS Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Politechnika Gdańska WETI",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "categories": [
            {
                "id": 1,
                "name": "dog",
                "supercategory": "animal",
                "keypoints": KEYPOINT_NAMES,
                "skeleton": [list(conn) for conn in SKELETON_CONNECTIONS],
            }
        ],
        "images": [],
        "annotations": [],
    }

    for idx, frame in enumerate(session.frames):
        image_id = idx + 1
        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"frame_{frame.frame_idx:04d}.jpg",
                "frame_idx": frame.frame_idx,
                "video": session.video_filename,
            }
        )
        ann = _build_coco_annotation(
            frame=frame,
            annotation_id=image_id,
            image_id=image_id,
            # Klatka neutralna psa, którego dotyczy anotacja — nie sesyjna
            neutral_frame_id=_neutral_frame_id(session, frame.track_id),
            dog=_dog_for(session, frame.track_id),
        )
        coco["annotations"].append(ann)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json", encoding="utf-8"
    ) as tmp:
        json.dump(coco, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name

    filename = f"dogfacs_{session_id}_{session.video_filename}.json"
    return FileResponse(tmp_path, media_type="application/json", filename=filename)
