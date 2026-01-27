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

    # Ścieżka do project root (4 poziomy w górę od apps/webapp/backend/main.py)
    # apps/webapp/backend/main.py -> apps/webapp/backend -> apps/webapp -> apps -> project_root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    models_dir = project_root / "models"

    print(f"\nProject root: {project_root}")
    print(f"Models directory: {models_dir}")

    try:
        config = PipelineConfig(
            bbox_weights=models_dir / "yolov8m.pt",
            breed_weights=models_dir / "breed.pt",
            keypoints_weights=models_dir / "keypoints_dogflw.pt",
            breeds_json=project_root / "packages" / "models" / "breeds.json",
            device="cpu",  # Użyj CPU dla stabilności
            use_rule_based_emotion=True,  # Rule-based emotion
        )

        pipeline = InferencePipeline(config)
        pipeline.load()

        print("\n✅ Pipeline załadowany - API gotowe!")
        print("=" * 60 + "\n")
    except FileNotFoundError as e:
        print(f"\n⚠️  BRAK MODELI: {e}")
        print("\nAby pobrać modele, uruchom:")
        print("  python scripts/download/download_models.py")
        print("\nBackend działa w STUB MODE - API dostępne, ale bez modeli.")
        print("=" * 60 + "\n")
        pipeline = None


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

def extract_frames_from_video(
    video_path: Path,
    fps_sample: float = 30.0,
    max_duration: int = 60
) -> list[np.ndarray]:
    """
    Ekstrahuj klatki z wideo z kontrolą FPS.

    Args:
        video_path: Ścieżka do pliku wideo
        fps_sample: FPS dla ekstrakcji (1-30 fps)
        max_duration: Maksymalna długość wideo (sekundy)

    Returns:
        Lista klatek jako numpy arrays (BGR)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Nie można otworzyć wideo: {video_path}")

    # Pobierz FPS wideo
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # Default

    # Oblicz co którą klatkę brać
    frame_skip = max(1, int(video_fps / fps_sample))

    frames = []
    frame_count = 0
    max_frames = int(max_duration * fps_sample)

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Bierz tylko co N-tą klatkę
        if frame_count % frame_skip == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    return frames


def draw_keypoints_on_frame(
    frame: np.ndarray,
    keypoints: Optional[np.ndarray],
    draw_skeleton: bool = True,
) -> np.ndarray:
    """
    Rysuj keypoints na klatce.

    Args:
        frame: Klatka jako numpy array (BGR)
        keypoints: Keypoints jako flat array [x0,y0,v0,...] lub None
        draw_skeleton: Czy rysować połączenia między punktami

    Returns:
        Klatka z narysowanymi keypoints
    """
    if keypoints is None:
        return frame

    from packages.data.schemas import (
        NUM_KEYPOINTS,
        SKELETON_CONNECTIONS,
        get_keypoint_color,
    )

    frame = frame.copy()
    kp = keypoints.reshape(NUM_KEYPOINTS, 3)

    # Rysuj skeleton (połączenia) najpierw
    if draw_skeleton:
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            x1, y1, v1 = kp[start_idx]
            x2, y2, v2 = kp[end_idx]

            # Rysuj tylko jeśli oba punkty są widoczne
            if v1 > 0.3 and v2 > 0.3:
                cv2.line(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (200, 200, 200),  # Szara linia
                    1,
                    cv2.LINE_AA,
                )

    # Rysuj punkty
    for i in range(NUM_KEYPOINTS):
        x, y, visibility = kp[i]

        if visibility > 0.3:  # Rysuj tylko widoczne punkty
            color = get_keypoint_color(i)
            # BGR dla OpenCV
            color_bgr = (color[2], color[1], color[0])

            # Większy punkt dla ważnych części (oczy, nos, usta)
            radius = 4 if i in [0, 1, 2, 7, 8] else 3

            cv2.circle(frame, (int(x), int(y)), radius, color_bgr, -1)
            cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 255), 1)

    return frame


def save_frame_to_disk(
    frame: np.ndarray,
    frame_idx: int,
    session_id: str,
    keypoints: Optional[np.ndarray] = None,
    draw_keypoints: bool = True,
) -> Path:
    """
    Zapisz klatkę do dysku z opcjonalnymi keypoints.

    Args:
        frame: Klatka jako numpy array (BGR)
        frame_idx: Indeks klatki
        session_id: ID sesji
        keypoints: Opcjonalne keypoints do narysowania
        draw_keypoints: Czy rysować keypoints na klatce

    Returns:
        Ścieżka do zapisanej klatki
    """
    session_dir = STATIC_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Rysuj keypoints jeśli podane
    if draw_keypoints and keypoints is not None:
        frame = draw_keypoints_on_frame(frame, keypoints)

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
    fps_sample: float = 1.0,
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
        raise HTTPException(
            status_code=503,
            detail="Pipeline nie załadowany. Uruchom: python scripts/download/download_models.py"
        )

    # Zapisz uploaded file do temp
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        video_path = Path(tmp.name)

    try:
        # Ekstrahuj klatki
        print(f"\n📹 Przetwarzanie wideo: {file.filename}")
        print(f"  → FPS sampling: {fps_sample} fps")
        frames_list = extract_frames_from_video(video_path, fps_sample=fps_sample, max_duration=60)
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

            # Zapisz klatkę z keypoints
            keypoints = peak_data["keypoints"]
            frame_path = save_frame_to_disk(frame, frame_idx, session_id, keypoints=keypoints)

            # URL dla frontend
            image_url = f"/static/{session_id}/frame_{frame_idx:04d}.jpg"

            # Przekonwertuj delta AUs do dict (konwertuj numpy types na Python)
            aus_dict = {
                au.name: {
                    "ratio": float(au.ratio),
                    "delta": float(au.delta),
                    "is_active": bool(au.is_active),  # numpy.bool_ -> Python bool
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

        # Zapisz neutral frame z keypoints
        neutral_idx = result["neutral_frame_idx"]
        neutral_frame = frames_list[neutral_idx]
        neutral_keypoints = result["neutral_keypoints"]
        neutral_path = save_frame_to_disk(
            neutral_frame, neutral_idx, session_id, keypoints=neutral_keypoints
        )
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
