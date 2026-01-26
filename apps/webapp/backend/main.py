"""
FastAPI backend dla DogFACS Dataset Generator.

Endpoints:
- POST /api/process_video - Przetwarza wideo i zwraca peak frames
- POST /api/export_coco - Eksportuje dataset do formatu COCO
- GET /api/health - Health check
"""

import tempfile
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from packages.pipeline import InferencePipeline, PipelineConfig


# =============================================================================
# FastAPI App Configuration
# =============================================================================

app = FastAPI(
    title="DogFACS Dataset Generator API",
    description="API dla generowania datasetu z wideo psów",
    version="1.0.0",
)

# CORS dla React frontend (Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files dla zapisanych klatek
STATIC_DIR = Path("apps/webapp/backend/static/frames")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global pipeline instance
pipeline: Optional[InferencePipeline] = None


# =============================================================================
# Pydantic Models
# =============================================================================

class ProcessVideoRequest(BaseModel):
    """Request dla przetwarzania wideo."""
    num_peaks: int = 10
    neutral_idx: Optional[int] = None
    min_separation_frames: int = 30


class ExportCOCORequest(BaseModel):
    """Request dla eksportu COCO."""
    peak_frames: list[dict]
    neutral_frame_idx: int
    video_filename: str


# =============================================================================
# Startup & Health
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Załaduj pipeline przy starcie aplikacji."""
    global pipeline

    print("\n" + "=" * 60)
    print("ŁADOWANIE DOGFACS DATASET GENERATOR")
    print("=" * 60)

    config = PipelineConfig(
        bbox_weights=Path("models/yolov8m.pt"),
        breed_weights=Path("models/breed.pt"),
        keypoints_weights=Path("models/keypoints_dogflw.pt"),
        device="cpu",  # Użyj CPU dla stabilności
        use_rule_based_emotion=True,  # Rule-based emotion
    )

    pipeline = InferencePipeline(config)
    pipeline.load()

    print("\n✅ Pipeline załadowany - API gotowe!")
    print("=" * 60 + "\n")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None and pipeline.is_loaded,
    }


# =============================================================================
# Video Processing
# =============================================================================

def extract_frames_from_video(video_path: Path, max_frames: int = 600) -> list[np.ndarray]:
    """
    Ekstrahuj klatki z wideo.

    Args:
        video_path: Ścieżka do pliku wideo
        max_frames: Maksymalna liczba klatek (30s @ 30fps = 900 frames)

    Returns:
        Lista klatek jako numpy arrays (BGR)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Nie można otworzyć wideo: {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_count += 1

        if frame_count >= max_frames:
            break

    cap.release()

    return frames


def save_frame_to_disk(frame: np.ndarray, frame_idx: int, session_id: str) -> Path:
    """
    Zapisz klatkę do dysku.

    Args:
        frame: Klatka jako numpy array (BGR)
        frame_idx: Indeks klatki
        session_id: ID sesji

    Returns:
        Ścieżka do zapisanej klatki
    """
    session_dir = STATIC_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    filename = f"frame_{frame_idx:04d}.jpg"
    filepath = session_dir / filename

    cv2.imwrite(str(filepath), frame)

    return filepath


@app.post("/api/process_video")
async def process_video(
    file: UploadFile = File(...),
    num_peaks: int = 10,
    neutral_idx: Optional[int] = None,
    min_separation_frames: int = 30,
):
    """
    Przetwarza uploaded wideo i zwraca peak frames.

    Pipeline:
    1. Ekstrahuj klatki z wideo
    2. Wykryj keypoints dla wszystkich klatek
    3. Auto-detekcja neutral frame (lub użyj podanego)
    4. Oblicz delta AU dla wszystkich klatek
    5. Wybierz peak frames (TFM-based)
    6. Klasyfikuj emocje dla peak frames

    Returns:
        JSON z neutral_frame_idx, peak_frames, total_frames
    """
    if pipeline is None or not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Pipeline nie załadowany")

    # Zapisz uploaded file do temp
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        video_path = Path(tmp.name)

    try:
        # Ekstrahuj klatki
        print(f"\n📹 Przetwarzanie wideo: {file.filename}")
        frames_list = extract_frames_from_video(video_path, max_frames=600)
        print(f"  → Wyekstrahowano {len(frames_list)} klatek")

        # Przetwórz wideo przez pipeline
        result = pipeline.process_video_for_dataset(
            frames_list=frames_list,
            num_peaks=num_peaks,
            neutral_idx=neutral_idx,
            min_separation_frames=min_separation_frames,
        )

        # Generuj session ID
        session_id = str(uuid.uuid4())[:8]

        # Zapisz peak frames do dysku
        peak_frames_data = []
        for peak_data in result["peak_frames"]:
            frame_idx = peak_data["frame_idx"]
            frame = peak_data["frame"]
            delta_aus = peak_data["delta_aus"]
            emotion = peak_data["emotion"]
            tfm_score = peak_data["tfm_score"]

            # Zapisz klatkę
            frame_path = save_frame_to_disk(frame, frame_idx, session_id)

            # URL dla frontend
            image_url = f"/static/{session_id}/frame_{frame_idx:04d}.jpg"

            # Przekonwertuj delta AUs do dict
            aus_dict = {
                au.name: {
                    "ratio": float(au.ratio),
                    "delta": float(au.delta),
                    "is_active": au.is_active,
                    "confidence": float(au.confidence),
                }
                for au in delta_aus.values()
            }

            peak_frames_data.append({
                "frame_idx": frame_idx,
                "image_url": image_url,
                "aus": aus_dict,
                "emotion": emotion.emotion,
                "emotion_confidence": float(emotion.confidence),
                "emotion_rule_applied": emotion.rule_applied,
                "tfm_score": float(tfm_score),
            })

        # Zapisz neutral frame
        neutral_idx = result["neutral_frame_idx"]
        neutral_frame = frames_list[neutral_idx]
        neutral_path = save_frame_to_disk(neutral_frame, neutral_idx, session_id)
        neutral_url = f"/static/{session_id}/frame_{neutral_idx:04d}.jpg"

        return JSONResponse({
            "session_id": session_id,
            "video_filename": file.filename,
            "neutral_frame_idx": neutral_idx,
            "neutral_frame_url": neutral_url,
            "peak_frames": peak_frames_data,
            "total_frames": len(frames_list),
        })

    except Exception as e:
        print(f"❌ Błąd przetwarzania: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Usuń temp file
        video_path.unlink(missing_ok=True)


# =============================================================================
# COCO Export
# =============================================================================

@app.post("/api/export_coco")
async def export_coco(request: ExportCOCORequest):
    """
    Eksportuje dataset do formatu COCO JSON.

    Args:
        request: ExportCOCORequest z peak_frames, neutral_frame_idx

    Returns:
        COCO JSON jako FileResponse
    """
    from packages.data.coco import COCODataset

    # TODO: Implementuj generowanie COCO z peak frames
    # Na razie zwróć placeholder

    coco_dataset = {
        "info": {
            "description": "DogFACS Dataset",
            "version": "1.0",
            "year": 2026,
        },
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "dog",
                "supercategory": "animal",
            }
        ],
    }

    # Dodaj images i annotations z peak frames
    for i, peak_frame in enumerate(request.peak_frames):
        image_id = i + 1
        annotation_id = i + 1

        # Image
        coco_dataset["images"].append({
            "id": image_id,
            "file_name": f"frame_{peak_frame['frame_idx']:04d}.jpg",
            "width": 640,  # TODO: Get from actual frame
            "height": 480,
        })

        # Annotation
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "emotion": peak_frame["emotion"],
            "emotion_confidence": peak_frame["emotion_confidence"],
            "emotion_rule_applied": peak_frame["emotion_rule_applied"],
            "neutral_frame_id": request.neutral_frame_idx,
            "au_analysis": {
                au_name: au_data["delta"]
                for au_name, au_data in peak_frame["aus"].items()
            },
            "tfm_score": peak_frame["tfm_score"],
        }

        coco_dataset["annotations"].append(annotation)

    # Zapisz do temp file
    import json
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        json.dump(coco_dataset, tmp, indent=2)
        tmp_path = tmp.name

    return FileResponse(
        tmp_path,
        media_type="application/json",
        filename=f"dogfacs_dataset_{request.video_filename}.json",
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
