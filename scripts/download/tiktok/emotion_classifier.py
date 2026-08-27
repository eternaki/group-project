"""
Klasyfikacja emocji dla pobranych wideo TikTok.

Owija InferencePipeline.process_video_for_dataset (trekowanie psów -> własna
klatka neutralna na trek -> delta AU -> reguły DogFACS) i agreguje emocje
z peak frames wszystkich przyjętych treków do jednej finalnej etykiety.
"""

from collections import Counter
from pathlib import Path

from packages.models.emotion import classify_emotion_from_delta_aus
from packages.pipeline.inference import InferencePipeline, PipelineConfig, VideoDatasetConfig
from packages.pipeline.video import VideoProcessor
from scripts.download.tiktok.config import (
    BBOX_WEIGHTS,
    BREED_WEIGHTS_DISABLED,
    BREEDS_JSON,
    EMOTION_FRAME_SAMPLE_COUNT,
    EMOTION_MAX_ROLL,
    EMOTION_MAX_YAW_ASYMMETRY,
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
            Nazwa emocji z EMOTION_CLASSES, albo None (brak godnych treków/peaków)
        """
        frames, sampling_fps = self._sample_frames(video_path)
        if len(frames) < 2:
            return None

        config = VideoDatasetConfig(
            num_peaks=min(EMOTION_NUM_PEAKS, len(frames)),
            fps=sampling_fps,
            min_keypoint_conf=EMOTION_MIN_KEYPOINT_CONF,
            max_yaw_asymmetry=EMOTION_MAX_YAW_ASYMMETRY,
            max_roll=EMOTION_MAX_ROLL,
            min_sharpness=EMOTION_MIN_SHARPNESS,
        )

        try:
            result = self._pipeline.process_video_for_dataset(frames, config=config)
        except ValueError:
            return None

        peak_predictions = []
        for track in result["tracks"]:
            peak_ids = set(track.peak_indices)
            for track_frame in track.frames:
                if track_frame.frame_idx in peak_ids:
                    peak_predictions.append(classify_emotion_from_delta_aus(track_frame.delta_aus))

        if not peak_predictions:
            return None

        return self._aggregate_emotion(peak_predictions)

    def _sample_frames(self, video_path: Path) -> tuple[list, float]:
        """Próbkuje równomiernie rozłożone klatki z wideo i zwraca (klatki, osiągnięte fps)."""
        info = self._video_processor.get_video_info(video_path)
        if info.duration <= 0:
            return [], 0.0

        sample_fps = EMOTION_FRAME_SAMPLE_COUNT / info.duration
        processor = VideoProcessor(fps_sample=sample_fps)
        frames = processor.extract_frames_to_list(
            video_path, max_frames=EMOTION_FRAME_SAMPLE_COUNT
        )
        if len(frames) < 2:
            return [], 0.0

        # Tempo NAPRAWDĘ osiągnięte (jak w batch_annotate.effective_fps) - od niego
        # zależą przeliczenia sekund na pozycje klatek w process_video_for_dataset.
        achieved_fps = len(frames) / info.duration
        return [frame for _, frame in frames], achieved_fps

    @staticmethod
    def _aggregate_emotion(peak_predictions: list) -> str | None:
        """
        Głosowanie większościowe po emocjach peak frames (remis: suma confidence).

        Odrzuca (zwraca None) niepewne/niezgodne wyniki zamiast wrzucać je do
        przypadkowej emocji: zwycięska emocja musi mieć >=EMOTION_MIN_VOTE_RATIO
        głosów i średnie confidence >=EMOTION_MIN_CONFIDENCE.
        """
        votes = Counter(pred.emotion for pred in peak_predictions)
        top_count = max(votes.values())
        tied = [emotion for emotion, count in votes.items() if count == top_count]

        if len(tied) == 1:
            winner = tied[0]
        else:
            confidence_sums = {emotion: 0.0 for emotion in tied}
            for pred in peak_predictions:
                if pred.emotion in confidence_sums:
                    confidence_sums[pred.emotion] += pred.confidence
            winner = max(confidence_sums, key=confidence_sums.get)

        vote_ratio = top_count / len(peak_predictions)
        if vote_ratio < EMOTION_MIN_VOTE_RATIO:
            return None

        winner_confidences = [pred.confidence for pred in peak_predictions if pred.emotion == winner]
        avg_confidence = sum(winner_confidences) / len(winner_confidences)
        if avg_confidence < EMOTION_MIN_CONFIDENCE:
            return None

        return winner
