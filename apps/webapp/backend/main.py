"""
FastAPI backend dla DogFACS Dataset Generator.

Endpoints:
- POST /api/process_video - Przetwarza wideo na treki psów i zwraca klatki szczytowe
- POST /api/export_coco - Eksportuje dataset do formatu COCO
- GET /api/health - Health check
"""

import os
import sys
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from packages.models.breed import BreedPrediction
from packages.models.emotion import classify_emotion_from_delta_aus
from packages.pipeline import InferencePipeline, PipelineConfig, VideoDatasetConfig
from packages.pipeline.inference import (
    DATASET_MIN_KEYPOINT_CONF,
    DATASET_PEAK_SEPARATION_S,
)
from packages.pipeline.peak_selector import compute_tfm
from packages.pipeline.track_processing import (
    NO_NEUTRAL_FRAME,
    TrackFrame,
    TrackResult,
)

# Dodaj katalog backend do PYTHONPATH (session_store i routers nie są pakietem)
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from routers.sessions import router as sessions_router  # noqa: E402  # isort: skip
from session_store import (  # noqa: E402  # isort: skip
    DogTrack,
    FrameAnnotation,
    RejectedTrack,
    SessionData,
    SessionStore,
    delta_aus_to_dict,
)

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
STATIC_DIR = _BACKEND_DIR / "static" / "frames"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Klatki zbiorów wsadowych — sesje z importu COCO wskazują na nie przez /dataset.
# Wystawiamy CAŁY katalog danych, a nie katalog jednego zbioru: inaczej dałoby
# się pracować tylko na jednym zbiorze naraz, a mamy ich kilka i anotator ma
# móc przełączać się między nimi bez restartu serwera.
DATASET_FRAMES_ENV: str = "DOGFACS_DATASET_FRAMES"
DEFAULT_DATASET_FRAMES: str = "data"
DATASET_FRAMES_DIR = Path(
    os.environ.get(DATASET_FRAMES_ENV, DEFAULT_DATASET_FRAMES)
).resolve()
if DATASET_FRAMES_DIR.is_dir():
    app.mount(
        "/dataset",
        StaticFiles(directory=str(DATASET_FRAMES_DIR)),
        name="dataset",
    )

# Podłącz router sesji (Sprint 9)
app.include_router(sessions_router)

# Global pipeline instance
pipeline: Optional[InferencePipeline] = None
_session_store = SessionStore()

# Domyślny odstęp między klatkami szczytowymi [klatki próbkowane].
# Wyprowadzony ze wspólnego progu zbierania zbioru, żeby anotator weryfikował
# DOKŁADNIE ten materiał, który wyprodukował batch — przy 30 klatkach i próbkowaniu
# 1 kl./s wychodziło 30 s odstępu wobec 3 s w batchu, czyli inne peaki.
DEFAULT_MIN_SEPARATION_FRAMES: int = int(DATASET_PEAK_SEPARATION_S * 1.0)
# Domyślna liczba klatek szczytowych na psa
DEFAULT_NUM_PEAKS: int = 10
# Domyślne próbkowanie klatek wideo [kl./s]
DEFAULT_FPS_SAMPLE: float = 1.0


# =============================================================================
# Pydantic Models
# =============================================================================

class ProcessVideoRequest(BaseModel):
    """Request dla przetwarzania wideo."""
    num_peaks: int = 10
    neutral_idx: Optional[int] = None
    min_separation_frames: int = DEFAULT_MIN_SEPARATION_FRAMES


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


def frame_url(session_id: str, frame_idx: int) -> str:
    """Adres podglądu klatki w sesji."""
    return f"/static/{session_id}/frame_{frame_idx:04d}.jpg"


def classify_breed_of_dog(
    frame: np.ndarray,
    body_box: tuple[int, int, int, int],
) -> Optional[BreedPrediction]:
    """
    Klasyfikuje rasę z kadru CAŁEGO psa.

    Kadr musi obejmować sylwetkę, nie mordę: EfficientNet-B4 uczono na całych
    psach i na samym pysku deklarowana skuteczność przestaje obowiązywać.

    Args:
        frame: Pełna klatka wideo
        body_box: Boks psa (x, y, w, h) — NIE boks mordy

    Returns:
        Predykcja rasy albo None, gdy model niedostępny lub kadr pusty
    """
    if pipeline is None or pipeline.breed_model is None:
        return None

    x, y, w, h = (int(v) for v in body_box)
    cropped = frame[max(0, y): y + h, max(0, x): x + w]
    if cropped.size == 0:
        return None

    try:
        return pipeline.breed_model.predict(cropped)
    except (ValueError, RuntimeError) as error:
        print(f"  ! Błąd klasyfikacji rasy: {error}")
        return None


def build_peak_annotation(
    session_id: str,
    frame: np.ndarray,
    track_frame: TrackFrame,
    track_id: int,
) -> FrameAnnotation:
    """
    Buduje anotację jednej klatki szczytowej wraz z zapisem jej podglądu.

    Do anotacji trafia `body_box` (boks psa) — to on jest bboxem w COCO i z niego
    liczy się rasa. `face_box` opisuje tylko kadr, w którym mierzono keypoints.

    Args:
        session_id: ID sesji
        frame: Pełna klatka wideo
        track_frame: Klatka treku z keypoints i delta AU
        track_id: Identyfikator psa

    Returns:
        Anotacja klatki gotowa do zapisu w sesji
    """
    save_frame_to_disk(frame, track_frame.frame_idx, session_id)
    emotion = classify_emotion_from_delta_aus(track_frame.delta_aus)
    breed = classify_breed_of_dog(frame, track_frame.body_box)

    return FrameAnnotation(
        frame_idx=track_frame.frame_idx,
        track_id=track_id,
        image_url=frame_url(session_id, track_frame.frame_idx),
        bbox=[int(v) for v in track_frame.body_box],
        keypoints=[float(v) for v in track_frame.keypoints],
        aus=delta_aus_to_dict(track_frame.delta_aus),
        emotion=emotion.emotion,
        emotion_confidence=float(emotion.confidence),
        emotion_rule_applied=emotion.rule_applied,
        breed=breed.class_name if breed is not None else None,
        breed_confidence=float(breed.confidence) if breed is not None else 0.0,
        tfm_score=float(compute_tfm(track_frame.delta_aus)),
    )


def annotate_track(
    session_id: str,
    frames_list: list[np.ndarray],
    track: TrackResult,
) -> tuple[DogTrack, list[FrameAnnotation]]:
    """
    Buduje dane jednego psa: jego klatkę neutralną i jego klatki szczytowe.

    Klatka neutralna jest per pies, bo delta AU liczy się względem niej —
    wspólna klatka neutralna mierzyłaby AU drugiego psa względem mimiki pierwszego.

    Args:
        session_id: ID sesji
        frames_list: Kolejne klatki wideo
        track: Przyjęty trek psa

    Returns:
        (opis psa dla sesji, anotacje jego klatek szczytowych)
    """
    peaks = set(track.peak_indices)
    annotations = [
        build_peak_annotation(session_id, frames_list[tf.frame_idx], tf, track.track_id)
        for tf in track.frames
        if tf.frame_idx in peaks
    ]

    neutral = next(
        (tf for tf in track.frames if tf.frame_idx == track.neutral_frame_idx), None
    )
    if neutral is not None:
        save_frame_to_disk(frames_list[neutral.frame_idx], neutral.frame_idx, session_id)

    dog = DogTrack(
        track_id=track.track_id,
        neutral_frame_idx=track.neutral_frame_idx,
        neutral_keypoints=(
            [float(v) for v in neutral.keypoints] if neutral is not None else None
        ),
        neutral_source=track.neutral_source,
        peak_frame_indices=list(track.peak_indices),
        au_noise={name: float(value) for name, value in track.au_noise.items()},
        au_sample_count={
            name: int(value) for name, value in track.au_sample_count.items()
        },
    )
    return dog, annotations


def build_session(
    session_id: str,
    video_filename: str,
    frames_list: list[np.ndarray],
    result: dict,
) -> SessionData:
    """
    Składa sesję anotacji z wyniku pipeline'u (wszystkie psy z wideo).

    Args:
        session_id: ID sesji
        video_filename: Nazwa wgranego pliku
        frames_list: Kolejne klatki wideo
        result: Wynik `process_video_for_dataset`

    Returns:
        Dane sesji z anotacjami, listą psów i psami odrzuconymi
    """
    dogs: list[DogTrack] = []
    frames: list[FrameAnnotation] = []
    for track in result["tracks"]:
        dog, annotations = annotate_track(session_id, frames_list, track)
        dogs.append(dog)
        frames.extend(annotations)
    frames.sort(key=lambda f: (f.frame_idx, f.track_id or 0))

    return SessionData(
        session_id=session_id,
        video_filename=video_filename,
        created_at=datetime.now().isoformat(),
        total_frames=result["total_frames"],
        # Pola sesyjne to dane PIERWSZEGO psa — zgodność ze starymi sesjami
        neutral_frame_idx=dogs[0].neutral_frame_idx if dogs else NO_NEUTRAL_FRAME,
        neutral_keypoints=dogs[0].neutral_keypoints if dogs else None,
        frames=frames,
        dogs=dogs,
        rejected_tracks=[
            RejectedTrack(
                track_id=track.track_id,
                reason=track.rejected_reason,
                frames=len(track.frames),
                frames_dropped_low_conf=track.frames_dropped_low_conf,
            )
            for track in result["rejected_tracks"]
        ],
    )


def process_video_response(session: SessionData) -> dict:
    """
    Buduje odpowiedź endpointu: lista psów plus pola zgodności ze starym frontem.

    Args:
        session: Zapisana sesja

    Returns:
        Słownik JSON-owalny z psami przyjętymi i odrzuconymi
    """
    dogs = [
        {
            "track_id": dog.track_id,
            "neutral_frame_idx": dog.neutral_frame_idx,
            "neutral_frame_url": frame_url(session.session_id, dog.neutral_frame_idx),
            "neutral_source": dog.neutral_source,
            "au_noise": dog.au_noise,
            "peak_frames": [
                asdict(f) for f in session.frames if f.track_id == dog.track_id
            ],
        }
        for dog in session.dogs
    ]

    return {
        "session_id": session.session_id,
        "video_filename": session.video_filename,
        "dogs": dogs,
        # Odrzucone treki jadą w odpowiedzi, bo to informacja dla anotatora
        # („pies zniknął, bo morda była za mała"), a nie śmieć do wyrzucenia
        "rejected_dogs": [asdict(track) for track in session.rejected_tracks],
        "total_frames": session.total_frames,
        # Pola zgodności: stary front czyta płaską listę peaków i jedną klatkę
        # neutralną. Przy jednym psie znaczą dokładnie to, co znaczyły wcześniej.
        "neutral_frame_idx": session.neutral_frame_idx,
        "neutral_frame_url": frame_url(session.session_id, session.neutral_frame_idx),
        "peak_frames": [asdict(f) for f in session.frames],
    }


async def save_upload_to_temp(file: UploadFile) -> Path:
    """Zapisuje wgrany plik do katalogu tymczasowego i zwraca jego ścieżkę."""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        return Path(tmp.name)


@app.post("/api/process_video")
async def process_video(
    file: UploadFile = File(...),
    num_peaks: int = DEFAULT_NUM_PEAKS,
    neutral_idx: Optional[int] = None,
    min_separation_frames: int = DEFAULT_MIN_SEPARATION_FRAMES,
    fps_sample: float = DEFAULT_FPS_SAMPLE,
):
    """
    Przetwarza wgrane wideo na treki psów i zwraca klatki szczytowe każdego psa.

    Pipeline (`process_video_for_dataset`): trekowanie psów → keypoints i ich
    wygładzanie w obrębie treku → własna klatka neutralna psa → delta AU → peaki.
    Psy, które nie przeszły progu godności treku, wracają w `rejected_dogs`
    z powodem — cichy brak psa w wyniku to dla anotatora brak informacji.

    Args:
        file: Plik wideo
        num_peaks: Ile klatek szczytowych wybrać NA PSA
        neutral_idx: Ręcznie wskazana klatka neutralna (dotyczy psów, które ją mają)
        min_separation_frames: Minimalny odstęp peaków w klatkach próbkowanych
        fps_sample: Próbkowanie klatek wideo [kl./s]

    Returns:
        JSON z `dogs`, `rejected_dogs`, `total_frames` oraz polami zgodności

    Raises:
        HTTPException: 503 gdy pipeline nie załadowany, 500 przy błędzie przetwarzania
    """
    if pipeline is None or not pipeline.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Pipeline nie załadowany. Uruchom: python scripts/download/download_models.py"
        )

    video_path = await save_upload_to_temp(file)
    try:
        print(f"\n📹 Przetwarzanie wideo: {file.filename} ({fps_sample} fps)")
        frames_list = extract_frames_from_video(video_path, fps_sample=fps_sample)
        print(f"  → Wyekstrahowano {len(frames_list)} klatek")

        # Odstęp peaków w API pipeline'u liczy się w sekundach; front podaje go
        # w klatkach próbkowanych, więc przeliczamy przez próbkowanie.
        config = VideoDatasetConfig(
            num_peaks=num_peaks,
            fps=fps_sample,
            neutral_frame_idx=neutral_idx,
            min_peak_separation_s=min_separation_frames / fps_sample,
            min_keypoint_conf=DATASET_MIN_KEYPOINT_CONF,
        )
        result = pipeline.process_video_for_dataset(frames_list, config=config)

        session = build_session(
            str(uuid.uuid4())[:8], file.filename, frames_list, result
        )
        _session_store.save(session)
        print(
            f"  → Sesja {session.session_id}: {len(session.dogs)} psów, "
            f"{len(session.frames)} klatek, "
            f"{len(session.rejected_tracks)} treków odrzuconych"
        )

        return JSONResponse(process_video_response(session))

    except (ValueError, RuntimeError, cv2.error) as e:
        import traceback

        print(f"❌ Błąd przetwarzania: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    finally:
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
            # Format zgodny z batchem: {ratio, is_active, confidence}. Wcześniej szła
            # tu goła `delta` (ratio - 1), a `packages.data.coco.au_ratio` czyta gołego
            # floata jako RATIO — aktywacja +45% (ratio 1.45) zapisywała się jako 0.45
            # i była odczytywana jako 55% SPADEK, czyli z odwróconym kierunkiem.
            "au_analysis": {
                au_name: {
                    "ratio": au_data["ratio"],
                    "is_active": au_data["is_active"],
                    "confidence": au_data["confidence"],
                }
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
