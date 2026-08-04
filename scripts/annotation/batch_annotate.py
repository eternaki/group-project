#!/usr/bin/env python3
"""
Skrypt do masowej anotacji wideo przy użyciu pipeline AI.

Funkcje:
- Przetwarzanie wszystkich wideo w katalogu
- Ekstrakcja klatek z konfigurowalnymi FPS
- Zapisywanie anotacji przyrostowo
- Obsługa błędów i wznowienie przetwarzania
- Logowanie postępu

Użycie:
    python scripts/annotation/batch_annotate.py --input-dir data/raw --output-dir data/annotations
    python scripts/annotation/batch_annotate.py --resume  # wznowienie przerwanego przetwarzania
    python scripts/annotation/batch_annotate.py --dry-run  # tylko analiza bez przetwarzania
"""

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import torch

from packages.data.coco import au_analysis_from_delta_aus
from packages.models.breed import BreedPrediction
from packages.models.emotion import EmotionPrediction, classify_emotion_from_delta_aus
from packages.models.head_pose import DEFAULT_MAX_ROLL, DEFAULT_MAX_YAW_ASYMMETRY
from packages.pipeline.inference import VideoDatasetConfig
from packages.pipeline.peak_selector import compute_tfm
from packages.pipeline.track_processing import TrackFrame, TrackResult

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Konfiguracja batch annotation."""

    input_dir: Path = field(default_factory=lambda: Path("data/raw"))
    output_dir: Path = field(default_factory=lambda: Path("data/annotations"))
    frames_dir: Path = field(default_factory=lambda: Path("data/frames"))
    progress_file: Path = field(default_factory=lambda: Path("data/annotations/progress.json"))

    # Parametry ekstrakcji
    fps: float = 1.0  # klatki na sekundę do ekstrakcji
    max_frames_per_video: int = 30  # maksymalna liczba klatek z wideo

    # Parametry generowania datasetu (peak frames + emocje/AU)
    num_peaks: int = 10  # liczba peak frames na wideo do anotacji emocji
    peak_min_separation_s: float = 3.0  # min. odstęp peak frames w SEKUNDACH nagrania
    min_keypoint_conf: float = 0.3  # min. pewność keypoints dla peak frame
    max_yaw_asymmetry: float = DEFAULT_MAX_YAW_ASYMMETRY  # dla peak frame
    max_roll: float = DEFAULT_MAX_ROLL  # dla peak frame

    # Parametry przetwarzania
    batch_size: int = 4  # rozmiar batcha dla GPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_interval: int = 100  # zapisuj co N klatek

    # Filtrowanie jakości
    min_confidence: float = 0.3  # minimalna pewność detekcji
    flag_low_confidence: float = 0.5  # próg dla flagowania niskiej jakości

    # Rozszerzenia wideo
    video_extensions: list[str] = field(
        default_factory=lambda: [".mp4", ".webm", ".mkv", ".avi", ".mov"]
    )


@dataclass
class VideoContext:
    """
    Dane jednego wideo potrzebne przy zapisie anotacji treku.

    Attributes:
        video_id: Identyfikator wideo (nazwa pliku bez rozszerzenia)
        emotion: Etykieta emocji z katalogu źródłowego
        frames: Wyekstrahowane klatki (indeks = numer klatki w treku)
        frame_numbers: Numery klatek w oryginalnym nagraniu
        frames_dir: Katalog na zapisywane klatki tego wideo
    """

    video_id: str
    emotion: str
    frames: list[np.ndarray]
    frame_numbers: list[int]
    frames_dir: Path


@dataclass
class ProcessingProgress:
    """Stan postępu przetwarzania."""

    processed_videos: list[str] = field(default_factory=list)
    total_frames: int = 0
    total_detections: int = 0
    low_confidence_frames: list[str] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    start_time: str = ""
    last_update: str = ""

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "processed_videos": self.processed_videos,
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "low_confidence_count": len(self.low_confidence_frames),
            "error_count": len(self.errors),
            "start_time": self.start_time,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingProgress":
        """Tworzy z słownika."""
        return cls(
            processed_videos=data.get("processed_videos", []),
            total_frames=data.get("total_frames", 0),
            total_detections=data.get("total_detections", 0),
            low_confidence_frames=data.get("low_confidence_frames", []),
            errors=data.get("errors", []),
            start_time=data.get("start_time", ""),
            last_update=data.get("last_update", ""),
        )


class BatchAnnotator:
    """
    Klasa do masowej anotacji wideo.

    Użycie:
        annotator = BatchAnnotator(config)
        annotator.process_all()
    """

    def __init__(self, config: BatchConfig) -> None:
        """
        Inicjalizuje annotator.

        Args:
            config: Konfiguracja batch annotation
        """
        self.config = config
        self.progress = ProcessingProgress()
        self.pipeline = None
        self.coco_dataset = None

        # Utwórz katalogi
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.frames_dir.mkdir(parents=True, exist_ok=True)

        # Wczytaj postęp jeśli istnieje
        self._load_progress()

    def _load_progress(self) -> None:
        """Wczytuje postęp z pliku."""
        if self.config.progress_file.exists():
            try:
                with open(self.config.progress_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.progress = ProcessingProgress.from_dict(data)
                logger.info(f"Wczytano postęp: {self.progress.total_frames} klatek przetworzonych")
            except Exception as e:
                logger.warning(f"Nie można wczytać postępu: {e}")
                self.progress = ProcessingProgress()

    def _save_progress(self) -> None:
        """Zapisuje postęp do pliku."""
        self.progress.last_update = datetime.now().isoformat()

        with open(self.config.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.progress.to_dict(), f, indent=2, ensure_ascii=False)

    def _init_pipeline(self) -> bool:
        """
        Inicjalizuje pipeline AI.

        Returns:
            True jeśli sukces
        """
        try:
            from packages.pipeline import InferencePipeline, PipelineConfig

            pipeline_config = PipelineConfig(device=self.config.device)
            self.pipeline = InferencePipeline(pipeline_config)
            self.pipeline.load()  # Załaduj wagi wszystkich modeli
            logger.info(f"Pipeline zainicjalizowany na urządzeniu: {self.config.device}")
            return True

        except ImportError as e:
            logger.error(f"Nie można zaimportować pipeline: {e}")
            return False
        except Exception as e:
            logger.error(f"Błąd inicjalizacji pipeline: {e}")
            return False

    def _init_coco_dataset(self) -> None:
        """Inicjalizuje lub wczytuje dataset COCO."""
        from packages.data import COCODataset
        from packages.data.coco import COCOInfo

        coco_path = self.config.output_dir / "annotations.json"

        if coco_path.exists():
            self.coco_dataset = COCODataset.load(coco_path)
            logger.info(f"Wczytano istniejący dataset COCO: {self.coco_dataset.num_images} obrazów")
        else:
            self.coco_dataset = COCODataset(
                COCOInfo(description="Dog FACS Dataset - Automatyczne anotacje"),
            )
            logger.info("Utworzono nowy dataset COCO")

    def get_video_files(self) -> list[Path]:
        """
        Znajduje wszystkie pliki wideo.

        Returns:
            Lista ścieżek do plików wideo
        """
        video_files = []

        for ext in self.config.video_extensions:
            video_files.extend(self.config.input_dir.rglob(f"*{ext}"))
            video_files.extend(self.config.input_dir.rglob(f"*{ext.upper()}"))

        # Sortuj dla spójności
        video_files = sorted(set(video_files))

        return video_files

    def extract_frames(
        self,
        video_path: Path,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """
        Ekstrahuje klatki z wideo.

        Args:
            video_path: Ścieżka do pliku wideo

        Yields:
            Tuple (numer_klatki, obraz)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Nie można otworzyć wideo: {video_path}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)

        if video_fps <= 0:
            video_fps = 30.0

        # Oblicz interwał
        frame_interval = int(video_fps / self.config.fps)
        if frame_interval < 1:
            frame_interval = 1

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                if extracted_count >= self.config.max_frames_per_video:
                    break

                yield frame_count, frame
                extracted_count += 1

            frame_count += 1

        cap.release()
        logger.debug(f"Wyekstrahowano {extracted_count} klatek z {video_path.name}")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
    ) -> Optional[dict]:
        """
        Przetwarza pojedynczą klatkę.

        Args:
            frame: Obraz jako numpy array
            frame_id: Identyfikator klatki

        Returns:
            Wynik przetwarzania lub None
        """
        if self.pipeline is None:
            return None

        try:
            result = self.pipeline.process_frame(frame, frame_id)
            return result

        except Exception as e:
            logger.error(f"Błąd przetwarzania klatki {frame_id}: {e}")
            return None

    def process_video(self, video_path: Path) -> dict:
        """
        Przetwarza pojedyncze wideo przez pełny pipeline datasetu.

        Wykorzystuje process_video_for_dataset(): trekowanie psów, własna klatka
        neutralna każdego psa, delta Action Units względem niej i wybór peak frames.
        Do COCO trafiają peak frames każdego przyjętego treku (boks mordy, keypoints,
        rasa, emocja, AU). Treki odrzucone lądują w logu z powodem — bez tego nie
        wiadomo, gdzie lejek danych traci materiał.

        Args:
            video_path: Ścieżka do pliku wideo

        Returns:
            Statystyki przetwarzania
        """
        video_id = video_path.stem
        emotion = video_path.parent.name  # Zakładamy strukturę: emocja/wideo.mp4

        stats = {
            "video_id": video_id,
            "frames_processed": 0,
            "detections": 0,
            "low_confidence": 0,
            "errors": 0,
        }

        logger.info(f"Przetwarzanie: {video_path.name} (emocja: {emotion})")

        frames_video_dir = self.config.frames_dir / emotion / video_id
        frames_video_dir.mkdir(parents=True, exist_ok=True)

        # Zbierz klatki (ekstrakcja wg fps, limit max_frames_per_video)
        frame_items = list(self.extract_frames(video_path))
        if len(frame_items) < 2:
            logger.warning(f"Za mało klatek ({len(frame_items)}) w {video_path.name}, pomijam")
            return stats

        context = VideoContext(
            video_id=video_id,
            emotion=emotion,
            frames=[frame for _, frame in frame_items],
            frame_numbers=[number for number, _ in frame_items],
            frames_dir=frames_video_dir,
        )

        # Konfiguracja powstaje poza try: zły próg to pomyłka wywołującego,
        # a nie powód do cichego pominięcia wideo.
        video_config = self._video_config(len(context.frames))
        try:
            dataset_result = self.pipeline.process_video_for_dataset(
                context.frames, config=video_config
            )
        except ValueError as e:
            logger.warning(f"Pominięto {video_path.name}: {e}")
            return stats

        for track in dataset_result["rejected_tracks"]:
            logger.info(f"  Trek {track.track_id} odrzucony: {track.rejected_reason}")

        for track in dataset_result["tracks"]:
            self._save_track_peaks(track, context, stats)

        self.progress.total_detections += stats["detections"]

        return stats

    def _video_config(self, num_frames: int) -> VideoDatasetConfig:
        """
        Buduje konfigurację przetwarzania wideo z ustawień batcha.

        Args:
            num_frames: Liczba wyekstrahowanych klatek

        Returns:
            Progi dla process_video_for_dataset()
        """
        return VideoDatasetConfig(
            num_peaks=min(self.config.num_peaks, num_frames),
            fps=self.config.fps,
            min_peak_separation_s=self.config.peak_min_separation_s,
            min_keypoint_conf=self.config.min_keypoint_conf,
            max_yaw_asymmetry=self.config.max_yaw_asymmetry,
            max_roll=self.config.max_roll,
        )

    def _save_track_peaks(
        self,
        track: TrackResult,
        context: VideoContext,
        stats: dict,
    ) -> None:
        """
        Zapisuje klatki szczytowe jednego treku (jednego psa) do zbioru.

        Nazwa pliku niesie track_id, bo na wideo z kilkoma psami ta sama klatka
        trafia do zbioru raz na psa — bez tego pliki nadpisywałyby się nawzajem.

        Args:
            track: Przyjęty trek
            context: Dane wideo
            stats: Statystyki przetwarzania (modyfikowane w miejscu)
        """
        peak_indices = set(track.peak_indices)

        for track_frame in track.frames:
            if track_frame.frame_idx not in peak_indices:
                continue

            frame_number = context.frame_numbers[track_frame.frame_idx]
            frame_id = f"{context.video_id}_t{track.track_id}_{frame_number:06d}"
            frame_path = context.frames_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(frame_path), context.frames[track_frame.frame_idx])
            stats["frames_processed"] += 1

            if self.coco_dataset is None:
                continue

            emotion_pred = self._add_track_annotation(track_frame, context, frame_path)
            stats["detections"] += 1

            # Flaguj niską pewność emocji
            if emotion_pred.confidence < self.config.flag_low_confidence:
                stats["low_confidence"] += 1
                self.progress.low_confidence_frames.append(frame_id)

            self.progress.total_frames += 1

            if self.progress.total_frames % self.config.save_interval == 0:
                self._save_intermediate()

    def _add_track_annotation(
        self,
        track_frame: TrackFrame,
        context: VideoContext,
        frame_path: Path,
    ) -> EmotionPrediction:
        """
        Dopisuje anotację jednej klatki szczytowej treku do zbioru COCO.

        Args:
            track_frame: Klatka treku z policzonymi delta AU
            context: Dane wideo
            frame_path: Ścieżka zapisanej klatki

        Returns:
            Predykcja emocji tej klatki
        """
        frame_img = context.frames[track_frame.frame_idx]
        height, width = frame_img.shape[:2]

        image_id = self.coco_dataset.add_image(
            file_name=str(frame_path.relative_to(self.config.frames_dir)),
            width=width,
            height=height,
            source_video=context.video_id,
            frame_number=context.frame_numbers[track_frame.frame_idx],
            emotion_label=context.emotion,
        )

        emotion_pred = classify_emotion_from_delta_aus(track_frame.delta_aus)
        breed_pred = self._classify_breed(frame_img, track_frame.face_box)

        confidence = {"emotion": float(emotion_pred.confidence)}
        breed_id = breed_name = None
        if breed_pred is not None:
            breed_id = int(breed_pred.class_id)
            breed_name = breed_pred.class_name
            confidence["breed"] = float(breed_pred.confidence)

        # Keypoints są już w układzie pełnego obrazu, format [x, y, v, ...]
        keypoints = [float(v) for v in track_frame.keypoints]

        # Zapisujemy komplet (ratio + is_active + confidence), bo samo ratio
        # nie odróżnia realnej aktywacji od klamrowanego, niewiarygodnego pomiaru.
        self.coco_dataset.add_annotation(
            image_id=image_id,
            bbox=[int(v) for v in track_frame.face_box],
            keypoints=keypoints,
            num_keypoints=int(sum(1 for v in keypoints[2::3] if v > 0)),
            breed_id=breed_id,
            breed_name=breed_name,
            emotion_id=int(emotion_pred.emotion_id),
            emotion_name=emotion_pred.emotion,
            confidence=confidence,
            au_analysis=au_analysis_from_delta_aus(track_frame.delta_aus),
            emotion_rule_applied=emotion_pred.rule_applied,
            tfm_score=float(compute_tfm(track_frame.delta_aus)),
        )
        return emotion_pred

    def _classify_breed(
        self,
        frame: np.ndarray,
        face_box: tuple[float, float, float, float],
    ) -> Optional[BreedPrediction]:
        """
        Klasyfikuje rasę psa z kadru mordy.

        UWAGA: to kadr MORDY, nie całego psa — trek nie niesie boksu ciała.
        Rasa z samej mordy jest słabsza niż z sylwetki; docelowo trek powinien
        nieść też boks ciała (patrz raport zadania 6).

        Args:
            frame: Pełna klatka wideo
            face_box: Boks mordy (x, y, w, h)

        Returns:
            Predykcja rasy albo None, gdy model niedostępny lub kadr pusty
        """
        if self.pipeline is None or self.pipeline.breed_model is None:
            return None

        x, y, w, h = (int(v) for v in face_box)
        cropped = frame[max(0, y) : y + h, max(0, x) : x + w]
        if cropped.size == 0:
            return None

        try:
            return self.pipeline.breed_model.predict(cropped)
        except Exception as e:
            logger.warning(f"Błąd klasyfikacji rasy: {e}")
            return None

    def _save_intermediate(self) -> None:
        """Zapisuje pośrednie wyniki."""
        # Zapisz COCO
        if self.coco_dataset is not None:
            coco_path = self.config.output_dir / "annotations.json"
            self.coco_dataset.save(coco_path)

        # Zapisz postęp
        self._save_progress()

        # Zwolnij pamięć
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Zapisano pośrednie wyniki ({self.progress.total_frames} klatek)")

    def process_all(self, dry_run: bool = False) -> dict:
        """
        Przetwarza wszystkie wideo.

        Args:
            dry_run: Tylko analiza bez przetwarzania

        Returns:
            Statystyki całkowite
        """
        # Znajdź wideo
        video_files = self.get_video_files()

        # Filtruj już przetworzone
        remaining = [
            v for v in video_files
            if v.stem not in self.progress.processed_videos
        ]

        logger.info(f"Znaleziono {len(video_files)} wideo, {len(remaining)} do przetworzenia")

        if dry_run:
            return {
                "total_videos": len(video_files),
                "remaining_videos": len(remaining),
                "processed_videos": len(self.progress.processed_videos),
            }

        # Inicjalizuj pipeline
        if not self._init_pipeline():
            return {"error": "Nie można zainicjalizować pipeline"}

        # Inicjalizuj COCO
        self._init_coco_dataset()

        # Ustaw czas startu
        if not self.progress.start_time:
            self.progress.start_time = datetime.now().isoformat()

        # Przetwarzaj
        total_stats = {
            "videos_processed": 0,
            "total_frames": 0,
            "total_detections": 0,
            "total_errors": 0,
            "processing_time_sec": 0,
        }

        start_time = time.time()

        for i, video_path in enumerate(remaining):
            try:
                logger.info(f"[{i+1}/{len(remaining)}] Przetwarzanie: {video_path.name}")

                video_stats = self.process_video(video_path)

                total_stats["videos_processed"] += 1
                total_stats["total_frames"] += video_stats["frames_processed"]
                total_stats["total_detections"] += video_stats["detections"]
                total_stats["total_errors"] += video_stats["errors"]

                # Zapisz jako przetworzone
                self.progress.processed_videos.append(video_path.stem)
                self._save_progress()

            except Exception as e:
                logger.error(f"Błąd przetwarzania {video_path}: {e}")
                self.progress.errors.append({
                    "video": str(video_path),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                total_stats["total_errors"] += 1

        # Finalne zapisanie
        self._save_intermediate()

        total_stats["processing_time_sec"] = round(time.time() - start_time, 2)

        # Oblicz FPS
        if total_stats["processing_time_sec"] > 0:
            total_stats["avg_fps"] = round(
                total_stats["total_frames"] / total_stats["processing_time_sec"], 2
            )

        return total_stats

    def generate_report(self) -> str:
        """
        Generuje raport z przetwarzania.

        Returns:
            Raport jako string
        """
        lines = [
            "=" * 60,
            "RAPORT BATCH ANNOTATION",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            "STATYSTYKI",
            "-" * 40,
            f"Przetworzone wideo:    {len(self.progress.processed_videos)}",
            f"Przetworzone klatki:   {self.progress.total_frames}",
            f"Łączne detekcje:       {self.progress.total_detections}",
            f"Niska pewność:         {len(self.progress.low_confidence_frames)}",
            f"Błędy:                 {len(self.progress.errors)}",
            "",
        ]

        if self.progress.start_time:
            lines.extend([
                "CZAS",
                "-" * 40,
                f"Start:         {self.progress.start_time[:19]}",
                f"Ostatnia akt.: {self.progress.last_update[:19] if self.progress.last_update else 'N/A'}",
                "",
            ])

        if self.progress.errors:
            lines.extend([
                "OSTATNIE BŁĘDY",
                "-" * 40,
            ])
            for error in self.progress.errors[-5:]:
                lines.append(f"  - {Path(error['video']).name}: {error['error'][:50]}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Masowa anotacja wideo dla Dog FACS Dataset"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Katalog z wideo",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Katalog wyjściowy",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/frames"),
        help="Katalog na wyekstrahowane klatki",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Klatki na sekundę do ekstrakcji (domyślnie: 1.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maksymalna liczba klatek z wideo (domyślnie: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Rozmiar batcha GPU (domyślnie: 4)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Zapisuj co N klatek (domyślnie: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Wznów przerwaną sesję",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tylko analiza, bez przetwarzania",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Pokaż raport i wyjdź",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Urządzenie do obliczeń",
    )

    args = parser.parse_args()

    # Konfiguracja
    config = BatchConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frames_dir=args.frames_dir,
        fps=args.fps,
        max_frames_per_video=args.max_frames,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        device=args.device,
    )

    # Utwórz annotator
    annotator = BatchAnnotator(config)

    # Tylko raport
    if args.report:
        print(annotator.generate_report())
        return

    # Dry run
    if args.dry_run:
        stats = annotator.process_all(dry_run=True)
        print("\n=== ANALIZA (DRY RUN) ===")
        print(f"Łącznie wideo:       {stats['total_videos']}")
        print(f"Do przetworzenia:    {stats['remaining_videos']}")
        print(f"Już przetworzone:    {stats['processed_videos']}")
        return

    # Przetwarzanie
    print("\n=== BATCH ANNOTATION ===")
    print(f"Input:  {config.input_dir}")
    print(f"Output: {config.output_dir}")
    print(f"Device: {config.device}")
    print(f"FPS:    {config.fps}")
    print("")

    stats = annotator.process_all()

    if "error" in stats:
        print(f"\nBŁĄD: {stats['error']}")
        return

    print("\n=== PODSUMOWANIE ===")
    print(f"Przetworzone wideo:  {stats['videos_processed']}")
    print(f"Przetworzone klatki: {stats['total_frames']}")
    print(f"Detekcje:            {stats['total_detections']}")
    print(f"Błędy:               {stats['total_errors']}")
    print(f"Czas przetwarzania:  {stats['processing_time_sec']}s")

    if "avg_fps" in stats:
        print(f"Średnie FPS:         {stats['avg_fps']}")

    print("\n" + annotator.generate_report())


if __name__ == "__main__":
    main()
