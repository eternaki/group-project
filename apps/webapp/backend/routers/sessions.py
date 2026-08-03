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
from dataclasses import asdict
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from session_store import (
    FrameAnnotation,
    FrameNotFoundError,
    SessionNotFoundError,
    SessionStore,
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


# =============================================================================
# Funkcje pomocnicze
# =============================================================================


def _load_session_or_404(session_id: str):
    """Wczytuje sesję lub rzuca HTTP 404."""
    try:
        return _store.load(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Sesja {session_id!r} nie istnieje")


def _get_frame_or_404(session_id: str, frame_idx: int) -> FrameAnnotation:
    """Wczytuje klatkę lub rzuca HTTP 404."""
    try:
        return _store.get_frame(session_id, frame_idx)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Sesja {session_id!r} nie istnieje")
    except FrameNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Klatka {frame_idx} nie istnieje w sesji {session_id!r}"
        )


def _delta_aus_to_dict(delta_aus: dict[str, DeltaActionUnit]) -> dict:
    """Konwertuje DeltaActionUnit do słownika do serializacji JSON."""
    return {
        name: {
            "ratio": float(au.ratio),
            "delta": float(au.delta),
            "is_active": bool(au.is_active),
            "confidence": float(au.confidence),
        }
        for name, au in delta_aus.items()
    }


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
# GET endpoints — DOG-S9-8
# =============================================================================


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Zwraca pełne dane sesji z wszystkimi anotacjami."""
    session = _load_session_or_404(session_id)
    return asdict(session)


@router.get("/{session_id}/frames")
async def list_frames(session_id: str):
    """Zwraca listę klatek z anotacjami dla danej sesji."""
    session = _load_session_or_404(session_id)
    return {"frames": [asdict(f) for f in session.frames]}


# =============================================================================
# PATCH endpoints — DOG-S9-2..5
# =============================================================================


@router.patch("/{session_id}/frames/{frame_idx}/keypoints")
async def update_keypoints(
    session_id: str,
    frame_idx: int,
    request: UpdateKeypointsRequest,
):
    """Aktualizuje keypoints klatki (DOG-S9-2)."""
    expected = NUM_KEYPOINTS * 3
    if len(request.keypoints) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Oczekiwano {expected} wartości keypoints, otrzymano {len(request.keypoints)}",
        )
    frame = _get_frame_or_404(session_id, frame_idx)
    frame.keypoints = request.keypoints
    frame.annotation_status = "reviewed"
    _store.update_frame(session_id, frame)
    return {"ok": True}


@router.patch("/{session_id}/frames/{frame_idx}/aus")
async def update_aus(
    session_id: str,
    frame_idx: int,
    request: UpdateAUsRequest,
):
    """Aktualizuje Action Units klatki (DOG-S9-3)."""
    frame = _get_frame_or_404(session_id, frame_idx)
    frame.aus = {name: au.model_dump() for name, au in request.aus.items()}
    frame.annotation_status = "reviewed"
    _store.update_frame(session_id, frame)
    return {"ok": True}


@router.patch("/{session_id}/frames/{frame_idx}/emotion")
async def update_emotion(
    session_id: str,
    frame_idx: int,
    request: UpdateEmotionRequest,
):
    """Aktualizuje emocję klatki (DOG-S9-4)."""
    if request.emotion not in EMOTION_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"Nieznana emocja {request.emotion!r}. Dozwolone: {EMOTION_CLASSES}",
        )
    frame = _get_frame_or_404(session_id, frame_idx)
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
):
    """Aktualizuje rasę psa na klatce (DOG-S9-5)."""
    frame = _get_frame_or_404(session_id, frame_idx)
    frame.breed = request.breed
    frame.breed_confidence = request.breed_confidence
    frame.annotation_status = "reviewed"
    _store.update_frame(session_id, frame)
    return {"ok": True}


# =============================================================================
# POST recompute endpoints — DOG-S9-6..7
# =============================================================================


@router.post("/{session_id}/frames/{frame_idx}/recompute_aus")
async def recompute_aus(session_id: str, frame_idx: int):
    """Przelicza AU z keypoints klatki i klatki neutralnej (DOG-S9-6)."""
    session = _load_session_or_404(session_id)

    frame = _get_frame_or_404(session_id, frame_idx)
    if frame.keypoints is None:
        raise HTTPException(status_code=422, detail="Klatka nie ma keypoints")
    if session.neutral_keypoints is None:
        raise HTTPException(status_code=422, detail="Sesja nie ma neutral keypoints")

    keypoints = np.array(frame.keypoints, dtype=np.float32)
    neutral_kp = np.array(session.neutral_keypoints, dtype=np.float32)

    extractor = DeltaActionUnitsExtractor(neutral_kp)
    delta_aus = extractor.extract(keypoints)

    frame.aus = _delta_aus_to_dict(delta_aus)
    _store.update_frame(session_id, frame)
    return {"ok": True, "aus": frame.aus}


@router.post("/{session_id}/frames/{frame_idx}/recompute_emotion")
async def recompute_emotion(session_id: str, frame_idx: int):
    """Przelicza emocję z AU klatki (DOG-S9-7)."""
    frame = _get_frame_or_404(session_id, frame_idx)
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
        source=request.source,
        annotation_status="auto",
    )
    _store.add_frame(session_id, frame)
    return {"ok": True, "frame_idx": request.frame_idx}


# =============================================================================
# POST export_coco — DOG-S9-9
# =============================================================================


def _count_visible_keypoints(keypoints: list[float]) -> int:
    """Liczy widoczne keypoints (visibility > 0.3)."""
    return sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0.3)


def _build_coco_annotation(
    frame: FrameAnnotation,
    annotation_id: int,
    image_id: int,
    neutral_frame_id: int,
) -> dict:
    """Buduje obiekt anotacji COCO dla jednej klatki."""
    ann: dict = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
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
    }
    if frame.bbox is not None:
        ann["bbox"] = frame.bbox
        ann["area"] = frame.bbox[2] * frame.bbox[3]
        ann["iscrowd"] = 0
    if frame.keypoints is not None:
        ann["keypoints"] = frame.keypoints
        ann["num_keypoints"] = _count_visible_keypoints(frame.keypoints)
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
            neutral_frame_id=session.neutral_frame_idx,
        )
        coco["annotations"].append(ann)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json", encoding="utf-8"
    ) as tmp:
        json.dump(coco, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name

    filename = f"dogfacs_{session_id}_{session.video_filename}.json"
    return FileResponse(tmp_path, media_type="application/json", filename=filename)
