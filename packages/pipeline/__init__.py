"""
Pakiet pipeline dla projektu Dog FACS.

Zawiera:
- InferencePipeline: Zunifikowany pipeline inference dla wszystkich modeli
- VideoProcessor: Procesor do ekstrakcji klatek z wideo
- YouTubeDownloader: Pobieranie wideo z YouTube i innych źródeł
- TemporalProcessor: Procesor do agregacji czasowej dla wideo
"""

from .inference import (
    PipelineConfig,
    DogAnnotation,
    FrameResult,
    InferencePipeline,
)
from .video import VideoProcessor, VideoInfo
from .downloader import YouTubeDownloader, DownloadResult
from .temporal_processor import (
    TemporalAUBuffer,
    TemporalAUResult,
    TemporalProcessor,
)

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
]
