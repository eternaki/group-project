"""
Klasyfikacja emocji dla pobranych wideo TikTok.

Owija istniejący InferencePipeline.process_video_for_dataset (keypoints ->
neutral frame -> delta AU -> reguły DogFACS) i agreguje emocje z kilku
"peak frames" do jednej finalnej etykiety emocji dla całego wideo.
"""

from collections import Counter
from pathlib import Path

from packages.pipeline.inference import InferencePipeline, PipelineConfig
from packages.pipeline.video import VideoProcessor
from scripts.download.tiktok.config import (
    BBOX_WEIGHTS,
    BREED_WEIGHTS_DISABLED,
    BREEDS_JSON,
    EMOTION_FRAME_SAMPLE_COUNT,
    EMOTION_MAX_HEAD_ANGLE,
    EMOTION_MIN_CONFIDENCE,
    EMOTION_MIN_KEYPOINT_CONF,
    EMOTION_MIN_SHARPNESS,
    EMOTION_MIN_VOTE_RATIO,
    EMOTION_NUM_PEAKS,
    FACE_DETECTOR_WEIGHTS,
    KEYPOINTS_WEIGHTS,
)


class VideoEmotionClassifier:
    """
    Klasyfikuje dominującą emocję dla krótkiego klipu wideo.

    Użycie:
        classifier = VideoEmotionClassifier(device="cpu")
        classifier.load()
        emotion = classifier.classify_video(Path("clip.mp4"))  # np. "happy" lub None
    """

    def __init__(self, device: str = "cpu") -> None:
        self._pipeline = InferencePipeline(
            PipelineConfig(
                bbox_weights=BBOX_WEIGHTS,
                breed_weights=BREED_WEIGHTS_DISABLED,
                keypoints_weights=KEYPOINTS_WEIGHTS,
                breeds_json=BREEDS_JSON,
                face_detector_weights=FACE_DETECTOR_WEIGHTS,
                device=device,
                use_rule_based_emotion=True,
            )
        )
        self._video_processor = VideoProcessor()

    def load(self) -> None:
        """Ładuje modele bbox/breed/keypoints wymagane przez pipeline."""
        self._pipeline.load()

    def classify_video(self, video_path: Path) -> str | None:
        """
        Zwraca dominującą emocję wideo lub None, jeśli nie da się jej ustalić.

        Args:
            video_path: Ścieżka do lokalnego pliku wideo

        Returns:
            Nazwa emocji z EMOTION_CLASSES, albo None (za mało pewnych klatek)
        """
        frames = self._sample_frames(video_path)
        if len(frames) < 2:
            return None

        try:
            result = self._pipeline.process_video_for_dataset(
                frames_list=frames,
                num_peaks=EMOTION_NUM_PEAKS,
                max_head_angle=EMOTION_MAX_HEAD_ANGLE,
                min_keypoint_conf=EMOTION_MIN_KEYPOINT_CONF,
                min_sharpness=EMOTION_MIN_SHARPNESS,
            )
        except ValueError:
            return None

        peak_frames = result["peak_frames"]
        if not peak_frames:
            return None

        return self._aggregate_emotion(peak_frames)

    def _sample_frames(self, video_path: Path) -> list:
        """Próbkuje równomiernie rozłożone klatki z wideo."""
        info = self._video_processor.get_video_info(video_path)
        if info.duration <= 0:
            return []

        sample_fps = EMOTION_FRAME_SAMPLE_COUNT / info.duration
        processor = VideoProcessor(fps_sample=sample_fps)
        frames = processor.extract_frames_to_list(
            video_path, max_frames=EMOTION_FRAME_SAMPLE_COUNT
        )
        return [frame for _, frame in frames]

    @staticmethod
    def _aggregate_emotion(peak_frames: list[dict]) -> str | None:
        """
        Głosowanie większościowe po emocjach peak frames (remis: suma confidence).

        Odrzuca (zwraca None) niepewne/niezgodne wyniki zamiast wrzucać je do
        przypadkowej emocji: zwycięska emocja musi mieć >=EMOTION_MIN_VOTE_RATIO
        głosów i średnie confidence >=EMOTION_MIN_CONFIDENCE.
        """
        votes = Counter(pf["emotion"].emotion for pf in peak_frames)
        top_count = max(votes.values())
        tied = [emotion for emotion, count in votes.items() if count == top_count]

        if len(tied) == 1:
            winner = tied[0]
        else:
            confidence_sums = {emotion: 0.0 for emotion in tied}
            for pf in peak_frames:
                emotion = pf["emotion"].emotion
                if emotion in confidence_sums:
                    confidence_sums[emotion] += pf["emotion"].confidence
            winner = max(confidence_sums, key=confidence_sums.get)

        vote_ratio = top_count / len(peak_frames)
        if vote_ratio < EMOTION_MIN_VOTE_RATIO:
            return None

        winner_confidences = [
            pf["emotion"].confidence for pf in peak_frames if pf["emotion"].emotion == winner
        ]
        avg_confidence = sum(winner_confidences) / len(winner_confidences)
        if avg_confidence < EMOTION_MIN_CONFIDENCE:
            return None

        return winner
