"""
Pakiet pipeline dla projektu Dog FACS.

Zawiera:
- InferencePipeline: Zunifikowany pipeline inference dla wszystkich modeli
- VideoProcessor: Procesor do ekstrakcji klatek z wideo
- YouTubeDownloader: Pobieranie wideo z YouTube i innych źródeł
"""

from .inference import (
    PipelineConfig,
    DogAnnotation,
    FrameResult,
    InferencePipeline,
)
from .video import VideoProcessor, VideoInfo
from .downloader import YouTubeDownloader, DownloadResult

__all__ = [
    "PipelineConfig",
    "DogAnnotation",
    "FrameResult",
    "InferencePipeline",
    "VideoProcessor",
    "VideoInfo",
    "YouTubeDownloader",
    "DownloadResult",
]
