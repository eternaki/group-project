"""
Pakiet pipeline dla projektu Dog FACS.

Zawiera:
- InferencePipeline: Zunifikowany pipeline inference dla wszystkich modeli
- VideoProcessor: Procesor do ekstrakcji klatek z wideo
- YouTubeDownloader: Pobieranie wideo z YouTube i innych źródeł
- TemporalProcessor: Procesor do agregacji czasowej dla wideo
- KeypointSmoother: Wygładzanie keypoints w obrębie treku (filtr One Euro)
- TrackFrame/TrackResult: Przetwarzanie pojedynczego treku psa (próg godności, szum AU)
"""

from .downloader import DownloadResult, YouTubeDownloader
from .inference import (
    DogAnnotation,
    FrameResult,
    InferencePipeline,
    PipelineConfig,
    VideoDatasetConfig,
)
from .landmark_smoothing import KeypointSmoother
from .temporal_processor import (
    TemporalAUBuffer,
    TemporalAUResult,
    TemporalProcessor,
)
from .track_processing import (
    DEFAULT_TRACK_QUALITY,
    MIN_FACE_SIZE_PX,
    MIN_KEYPOINT_CONF,
    MIN_NOISE_SAMPLES,
    MIN_TRACK_FRAMES,
    NO_NEUTRAL_FRAME,
    TrackFrame,
    TrackQuality,
    TrackResult,
    build_track_result,
    compute_au_noise,
    count_au_samples,
    evaluate_track_quality,
    rejected_track,
)
from .video import VideoInfo, VideoProcessor

__all__ = [
    # Inference Pipeline
    "PipelineConfig",
    "VideoDatasetConfig",
    "DogAnnotation",
    "FrameResult",
    "InferencePipeline",
    # Video Processing
    "VideoProcessor",
    "VideoInfo",
    # YouTube Downloader
    "YouTubeDownloader",
    "DownloadResult",
    # Temporal Processing (dla wideo)
    "TemporalAUBuffer",
    "TemporalAUResult",
    "TemporalProcessor",
    # Wygładzanie keypoints (dla treków)
    "KeypointSmoother",
    # Przetwarzanie treku (jeden pies)
    "TrackFrame",
    "TrackResult",
    "TrackQuality",
    "DEFAULT_TRACK_QUALITY",
    "evaluate_track_quality",
    "compute_au_noise",
    "count_au_samples",
    "build_track_result",
    "rejected_track",
    "MIN_TRACK_FRAMES",
    "MIN_FACE_SIZE_PX",
    "MIN_KEYPOINT_CONF",
    "MIN_NOISE_SAMPLES",
    "NO_NEUTRAL_FRAME",
]
