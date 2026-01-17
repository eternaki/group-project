"""
Procesor wideo dla ekstrakcji klatek.

Wyodrębnia klatki z filmów MP4 z konfigurowalną częstotliwością
i zwraca je jako generator dla efektywnego wykorzystania pamięci.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """
    Informacje o pliku wideo.

    Attributes:
        path: Ścieżka do pliku
        fps: Liczba klatek na sekundę
        frame_count: Całkowita liczba klatek
        width: Szerokość w pikselach
        height: Wysokość w pikselach
        duration: Czas trwania w sekundach
    """

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


class VideoProcessor:
    """
    Procesor do ekstrakcji klatek z plików wideo.

    Użycie:
        processor = VideoProcessor(fps_sample=1.0)

        # Pobierz informacje o wideo
        info = processor.get_video_info(Path("video.mp4"))
        print(f"Duration: {info.duration:.1f}s, Frames: {info.frame_count}")

        # Iteruj przez klatki
        for frame_idx, frame_rgb in processor.extract_frames(Path("video.mp4")):
            # Przetwórz klatkę...
            pass
    """

    def __init__(self, fps_sample: float = 1.0) -> None:
        """
        Inicjalizuje procesor wideo.

        Args:
            fps_sample: Częstotliwość próbkowania klatek (np. 1.0 = 1 klatka/s)
        """
        self.fps_sample = fps_sample

    def get_video_info(self, video_path: Path) -> VideoInfo:
        """
        Pobiera metadane pliku wideo.

        Args:
            video_path: Ścieżka do pliku wideo

        Returns:
            VideoInfo z metadanymi

        Raises:
            ValueError: Gdy nie można otworzyć pliku wideo
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Nie można otworzyć wideo: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            duration = frame_count / fps if fps > 0 else 0.0

            return VideoInfo(
                path=video_path,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                duration=duration,
            )
        finally:
            cap.release()

    def extract_frames(
        self,
        video_path: Path,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """
        Wyodrębnia klatki z wideo z określoną częstotliwością.

        Generator zwraca klatki w formacie RGB dla efektywnego
        wykorzystania pamięci.

        Args:
            video_path: Ścieżka do pliku wideo
            start_time: Czas startu w sekundach (domyślnie 0)
            end_time: Czas końca w sekundach (None = do końca)

        Yields:
            Tuple (indeks_klatki, klatka_rgb)

        Raises:
            ValueError: Gdy nie można otworzyć pliku wideo
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Nie można otworzyć wideo: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Oblicz interwał próbkowania
        if self.fps_sample <= 0:
            frame_interval = 1
        else:
            frame_interval = max(1, int(video_fps / self.fps_sample))

        # Oblicz zakres klatek
        start_frame = int(start_time * video_fps)
        if end_time is not None:
            end_frame = min(int(end_time * video_fps), total_frames)
        else:
            end_frame = total_frames

        # Przeskocz do startu
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        extracted_count = 0

        try:
            while frame_idx < end_frame:
                ret, frame = cap.read()

                if not ret:
                    break

                # Próbkuj zgodnie z interwałem
                if (frame_idx - start_frame) % frame_interval == 0:
                    # Konwertuj BGR -> RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield frame_idx, frame_rgb
                    extracted_count += 1

                frame_idx += 1

        finally:
            cap.release()

    def extract_frames_to_list(
        self,
        video_path: Path,
        max_frames: int | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """
        Wyodrębnia klatki do listy (dla małych wideo).

        Args:
            video_path: Ścieżka do pliku wideo
            max_frames: Maksymalna liczba klatek (None = bez limitu)

        Returns:
            Lista krotek (indeks_klatki, klatka_rgb)
        """
        frames = []
        for frame_idx, frame in self.extract_frames(video_path):
            frames.append((frame_idx, frame))
            if max_frames is not None and len(frames) >= max_frames:
                break
        return frames

    def save_frames(
        self,
        video_path: Path,
        output_dir: Path,
        format: str = "jpg",
        quality: int = 95,
    ) -> list[Path]:
        """
        Zapisuje klatki jako pliki obrazów.

        Args:
            video_path: Ścieżka do pliku wideo
            output_dir: Katalog docelowy
            format: Format obrazu ('jpg', 'png')
            quality: Jakość dla JPEG (1-100)

        Returns:
            Lista ścieżek do zapisanych plików
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        video_name = video_path.stem

        for frame_idx, frame in self.extract_frames(video_path):
            # Nazwa pliku: video_frame_000001.jpg
            filename = f"{video_name}_frame_{frame_idx:06d}.{format}"
            output_path = output_dir / filename

            # Konwertuj RGB -> BGR dla OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if format.lower() == "jpg":
                cv2.imwrite(
                    str(output_path),
                    frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, quality],
                )
            else:
                cv2.imwrite(str(output_path), frame_bgr)

            saved_paths.append(output_path)

        return saved_paths

    def count_estimated_frames(self, video_path: Path) -> int:
        """
        Szacuje liczbę klatek do wyodrębnienia.

        Args:
            video_path: Ścieżka do pliku wideo

        Returns:
            Szacowana liczba klatek
        """
        info = self.get_video_info(video_path)

        if self.fps_sample <= 0:
            return info.frame_count

        frame_interval = max(1, int(info.fps / self.fps_sample))
        return info.frame_count // frame_interval
