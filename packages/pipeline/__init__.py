"""
Pakiet pipeline dla projektu Dog FACS.

Zawiera:
- InferencePipeline: Zunifikowany pipeline inference dla wszystkich modeli
- VideoProcessor: Procesor do ekstrakcji klatek z wideo
"""

from .inference import (
    PipelineConfig,
    DogAnnotation,
    FrameResult,
    InferencePipeline,
)
from .video import VideoProcessor, VideoInfo

__all__ = [
    "PipelineConfig",
    "DogAnnotation",
    "FrameResult",
    "InferencePipeline",
    "VideoProcessor",
    "VideoInfo",
]
