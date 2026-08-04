"""
Pakiet pipeline dla projektu Dog FACS.

Zawiera:
- InferencePipeline: Zunifikowany pipeline inference dla wszystkich modeli
- VideoProcessor: Procesor do ekstrakcji klatek z wideo
- YouTubeDownloader: Pobieranie wideo z YouTube i innych źródeł
- TemporalProcessor: Procesor do agregacji czasowej dla wideo
- KeypointSmoother: Wygładzanie keypoints w obrębie treku (filtr One Euro)
"""

from .downloader import DownloadResult, YouTubeDownloader
from .inference import (
    DogAnnotation,
    FrameResult,
    InferencePipeline,
    PipelineConfig,
)
from .landmark_smoothing import KeypointSmoother
from .temporal_processor import (
    TemporalAUBuffer,
    TemporalAUResult,
    TemporalProcessor,
)
from .video import VideoInfo, VideoProcessor

__all__ = [
    # Inference Pipeline
    "PipelineConfig",
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
]
