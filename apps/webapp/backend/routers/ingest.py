"""
Router dosypywania nagrań.

Endpoints:
    POST /api/ingest/upload  — przyjmij pliki wideo do wspólnego katalogu
    POST /api/ingest/start   — puść przetwarzanie w tle
    GET  /api/ingest/status  — ile nagrań, ile przerobionych, czy trwa praca
"""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from ingest import (
    IngestError,
    list_videos,
    save_video,
    start_processing,
    status,
)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# Korzeń repozytorium — katalog roboczy procesu przetwarzania
_REPO_ROOT = Path(__file__).resolve().parents[4]


@router.post("/upload")
async def upload_videos(files: list[UploadFile] = File(...)):
    """
    Przyjmuje nagrania do wspólnego katalogu zespołu.

    Katalog jest jeden dla wszystkich — dzieli się przydział, nie materiał,
    a przydział liczy się ze skrótu nazwy nagrania, więc dorzucenie plików
    nie rusza tego, co ludzie już mają w pracy.

    Args:
        files: Pliki wideo z przeglądarki

    Returns:
        `{"saved": [nazwy], "skipped": [{name, reason}], "videos_total": int}`
    """
    saved: list[str] = []
    skipped: list[dict] = []
    for upload in files:
        try:
            path = save_video(upload.filename or "", await upload.read())
            saved.append(path.name)
        except IngestError as error:
            skipped.append({"name": upload.filename, "reason": str(error)})
    return {"saved": saved, "skipped": skipped, "videos_total": len(list_videos())}


@router.post("/start")
async def start_ingest():
    """
    Puszcza przetwarzanie dosypanych nagrań w osobnym procesie.

    Osobny proces, bo jedno nagranie to około siedemdziesięciu sekund — w procesie
    backendu zablokowałoby anotację na cały przebieg, czyli dokładnie to, czemu
    ta funkcja ma zapobiegać.

    Returns:
        `{"ok": True, "pid": int}`

    Raises:
        HTTPException: 409, gdy przetwarzanie już trwa albo nie ma czego przetwarzać
    """
    try:
        pid = start_processing(_REPO_ROOT)
    except IngestError as error:
        raise HTTPException(status_code=409, detail=str(error))
    return {"ok": True, "pid": pid}


@router.get("/status")
async def ingest_status():
    """
    Zwraca stan dosypywania.

    Returns:
        `{videos_total, videos_processed, running, pairs_ready, stage, stage_label}`
    """
    current = status()
    return {
        "videos_total": current.videos_total,
        "videos_processed": current.videos_processed,
        "running": current.running,
        "pairs_ready": current.pairs_ready,
        "stage": current.stage,
        "stage_label": current.stage_label,
    }
