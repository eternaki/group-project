#!/usr/bin/env python3
"""
Skrypt do benchmarku wydajności pipeline AI.

Funkcje:
- Testowanie przepustowości (FPS)
- Monitorowanie pamięci GPU
- Porównywanie różnych batch sizes
- Wykrywanie wycieków pamięci

Użycie:
    python scripts/annotation/benchmark.py
    python scripts/annotation/benchmark.py --batch-sizes 1,2,4,8
    python scripts/annotation/benchmark.py --num-iterations 100
"""

import argparse
import gc
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Konfiguracja benchmarku."""

    num_iterations: int = 50
    warmup_iterations: int = 5
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    image_size: tuple[int, int] = (720, 1280)  # H x W
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BenchmarkResult:
    """Wynik pojedynczego benchmarku."""

    batch_size: int
    iterations: int
    total_time_sec: float
    avg_time_per_batch_ms: float
    avg_time_per_frame_ms: float
    fps: float
    gpu_memory_mb: float
    gpu_peak_memory_mb: float
    device: str


class PipelineBenchmark:
    """
    Klasa do benchmarku pipeline AI.

    Użycie:
        benchmark = PipelineBenchmark(config)
        results = benchmark.run_all()
        benchmark.print_report(results)
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """
        Inicjalizuje benchmark.

        Args:
            config: Konfiguracja benchmarku
        """
        self.config = config
        self.pipeline = None

    def _init_pipeline(self) -> bool:
        """
        Inicjalizuje pipeline.

        Returns:
            True jeśli sukces
        """
        try:
            from packages.pipeline import InferencePipeline

            self.pipeline = InferencePipeline(device=self.config.device)
            logger.info(f"Pipeline zainicjalizowany na: {self.config.device}")
            return True

        except ImportError as e:
            logger.error(f"Nie można zaimportować pipeline: {e}")
            return False
        except Exception as e:
            logger.error(f"Błąd inicjalizacji pipeline: {e}")
            return False

    def _generate_dummy_batch(self, batch_size: int) -> list[np.ndarray]:
        """
        Generuje batch sztucznych obrazów.

        Args:
            batch_size: Rozmiar batcha

        Returns:
            Lista obrazów
        """
        h, w = self.config.image_size
        batch = []

        for _ in range(batch_size):
            # Losowy obraz RGB
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            batch.append(img)

        return batch

    def _get_gpu_memory(self) -> tuple[float, float]:
        """
        Pobiera wykorzystanie pamięci GPU.

        Returns:
            Tuple (aktualne_mb, szczytowe_mb)
        """
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            return current, peak
        return 0.0, 0.0

    def _clear_gpu_memory(self) -> None:
        """Czyści pamięć GPU."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def run_benchmark(self, batch_size: int) -> BenchmarkResult:
        """
        Uruchamia benchmark dla danego batch size.

        Args:
            batch_size: Rozmiar batcha

        Returns:
            Wynik benchmarku
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline nie zainicjalizowany")

        # Wyczyść pamięć przed testem
        self._clear_gpu_memory()

        # Warmup
        logger.info(f"Warmup ({self.config.warmup_iterations} iteracji)...")
        for _ in range(self.config.warmup_iterations):
            batch = self._generate_dummy_batch(batch_size)
            for img in batch:
                self.pipeline.process_frame(img)

        # Właściwy benchmark
        logger.info(f"Benchmark batch_size={batch_size} ({self.config.num_iterations} iteracji)...")

        self._clear_gpu_memory()
        start_time = time.perf_counter()

        for i in range(self.config.num_iterations):
            batch = self._generate_dummy_batch(batch_size)
            for img in batch:
                self.pipeline.process_frame(img)

        end_time = time.perf_counter()

        # Statystyki
        total_time = end_time - start_time
        total_frames = self.config.num_iterations * batch_size

        gpu_memory, gpu_peak = self._get_gpu_memory()

        result = BenchmarkResult(
            batch_size=batch_size,
            iterations=self.config.num_iterations,
            total_time_sec=round(total_time, 3),
            avg_time_per_batch_ms=round(total_time / self.config.num_iterations * 1000, 2),
            avg_time_per_frame_ms=round(total_time / total_frames * 1000, 2),
            fps=round(total_frames / total_time, 2),
            gpu_memory_mb=round(gpu_memory, 1),
            gpu_peak_memory_mb=round(gpu_peak, 1),
            device=self.config.device,
        )

        return result

    def run_all(self) -> list[BenchmarkResult]:
        """
        Uruchamia wszystkie benchmarki.

        Returns:
            Lista wyników
        """
        if not self._init_pipeline():
            return []

        results = []

        for batch_size in self.config.batch_sizes:
            try:
                result = self.run_benchmark(batch_size)
                results.append(result)
                logger.info(f"Batch {batch_size}: {result.fps} FPS, {result.gpu_memory_mb} MB GPU")

            except Exception as e:
                logger.error(f"Błąd benchmarku dla batch_size={batch_size}: {e}")

        return results

    def check_memory_leak(self, iterations: int = 100) -> dict:
        """
        Sprawdza wycieki pamięci.

        Args:
            iterations: Liczba iteracji do testu

        Returns:
            Wyniki testu
        """
        if self.pipeline is None:
            if not self._init_pipeline():
                return {"error": "Pipeline nie zainicjalizowany"}

        self._clear_gpu_memory()

        memory_samples = []

        for i in range(iterations):
            img = self._generate_dummy_batch(1)[0]
            self.pipeline.process_frame(img)

            if i % 10 == 0:
                current, _ = self._get_gpu_memory()
                memory_samples.append(current)

        # Analiza
        if len(memory_samples) < 2:
            return {"error": "Za mało próbek"}

        start_memory = memory_samples[0]
        end_memory = memory_samples[-1]
        memory_growth = end_memory - start_memory

        # Regresja liniowa dla wykrycia trendu
        x = np.arange(len(memory_samples))
        slope, _ = np.polyfit(x, memory_samples, 1)

        has_leak = memory_growth > 100 or slope > 5  # > 100MB wzrost lub > 5MB/10 iteracji

        return {
            "iterations": iterations,
            "start_memory_mb": round(start_memory, 1),
            "end_memory_mb": round(end_memory, 1),
            "memory_growth_mb": round(memory_growth, 1),
            "memory_slope_mb": round(slope, 3),
            "has_potential_leak": has_leak,
            "samples": len(memory_samples),
        }

    def print_report(self, results: list[BenchmarkResult]) -> str:
        """
        Generuje raport benchmarku.

        Args:
            results: Lista wyników

        Returns:
            Raport jako string
        """
        lines = [
            "=" * 70,
            "RAPORT BENCHMARKU PIPELINE AI",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Urządzenie: {self.config.device}",
            f"Rozmiar obrazu: {self.config.image_size}",
            f"Iteracji per test: {self.config.num_iterations}",
            "=" * 70,
            "",
            f"{'Batch':<8} {'Total(s)':<10} {'ms/batch':<12} {'ms/frame':<12} {'FPS':<10} {'GPU(MB)':<10} {'Peak(MB)':<10}",
            "-" * 70,
        ]

        for r in results:
            lines.append(
                f"{r.batch_size:<8} {r.total_time_sec:<10} {r.avg_time_per_batch_ms:<12} "
                f"{r.avg_time_per_frame_ms:<12} {r.fps:<10} {r.gpu_memory_mb:<10} {r.gpu_peak_memory_mb:<10}"
            )

        lines.extend([
            "",
            "=" * 70,
        ])

        # Najlepszy wynik
        if results:
            best = max(results, key=lambda x: x.fps)
            lines.extend([
                "",
                "NAJLEPSZY WYNIK",
                "-" * 40,
                f"Batch size:     {best.batch_size}",
                f"FPS:            {best.fps}",
                f"ms per frame:   {best.avg_time_per_frame_ms}",
                f"GPU Memory:     {best.gpu_memory_mb} MB",
            ])

            # Sprawdź czy osiągnięto cel > 10 FPS
            target_fps = 10
            if best.fps >= target_fps:
                lines.append(f"\n✓ Cel {target_fps} FPS OSIĄGNIĘTY")
            else:
                lines.append(f"\n✗ Cel {target_fps} FPS NIE OSIĄGNIĘTY (brakuje {target_fps - best.fps:.1f} FPS)")

        lines.append("")

        return "\n".join(lines)

    def save_results(
        self,
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Zapisuje wyniki do pliku.

        Args:
            results: Lista wyników
            output_path: Ścieżka wyjściowa

        Returns:
            Ścieżka do pliku
        """
        import json

        if output_path is None:
            output_path = Path("data/quality/benchmark_results.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_iterations": self.config.num_iterations,
                "warmup_iterations": self.config.warmup_iterations,
                "image_size": self.config.image_size,
                "device": self.config.device,
            },
            "results": [
                {
                    "batch_size": r.batch_size,
                    "iterations": r.iterations,
                    "total_time_sec": r.total_time_sec,
                    "avg_time_per_batch_ms": r.avg_time_per_batch_ms,
                    "avg_time_per_frame_ms": r.avg_time_per_frame_ms,
                    "fps": r.fps,
                    "gpu_memory_mb": r.gpu_memory_mb,
                    "gpu_peak_memory_mb": r.gpu_peak_memory_mb,
                }
                for r in results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Zapisano wyniki do {output_path}")
        return output_path


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark wydajności pipeline AI"
    )

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8",
        help="Rozmiary batcha do testowania (domyślnie: 1,2,4,8)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=50,
        help="Liczba iteracji per test (domyślnie: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Liczba iteracji warmup (domyślnie: 5)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=720,
        help="Wysokość obrazu (domyślnie: 720)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1280,
        help="Szerokość obrazu (domyślnie: 1280)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Urządzenie do obliczeń",
    )
    parser.add_argument(
        "--check-memory-leak",
        action="store_true",
        help="Sprawdź wycieki pamięci",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        help="Zapisz wyniki do pliku JSON",
    )

    args = parser.parse_args()

    # Parsuj batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Konfiguracja
    config = BenchmarkConfig(
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup,
        batch_sizes=batch_sizes,
        image_size=(args.image_height, args.image_width),
        device=args.device,
    )

    # Benchmark
    benchmark = PipelineBenchmark(config)

    # Test wycieków pamięci
    if args.check_memory_leak:
        print("\n=== TEST WYCIEKÓW PAMIĘCI ===")
        leak_results = benchmark.check_memory_leak()

        print(f"Iteracji:        {leak_results.get('iterations', 'N/A')}")
        print(f"Start pamięci:   {leak_results.get('start_memory_mb', 'N/A')} MB")
        print(f"Koniec pamięci:  {leak_results.get('end_memory_mb', 'N/A')} MB")
        print(f"Wzrost:          {leak_results.get('memory_growth_mb', 'N/A')} MB")
        print(f"Trend:           {leak_results.get('memory_slope_mb', 'N/A')} MB/10 iter")

        if leak_results.get("has_potential_leak"):
            print("\n⚠ WYKRYTO POTENCJALNY WYCIEK PAMIĘCI")
        else:
            print("\n✓ Brak wycieków pamięci")

        return

    # Główny benchmark
    print("\n=== BENCHMARK PIPELINE AI ===")
    print(f"Urządzenie: {config.device}")
    print(f"Batch sizes: {config.batch_sizes}")
    print(f"Iteracji: {config.num_iterations}")
    print("")

    results = benchmark.run_all()

    if not results:
        print("BŁĄD: Nie udało się uruchomić benchmarku")
        return

    # Raport
    report = benchmark.print_report(results)
    print(report)

    # Zapisz wyniki
    if args.save_results:
        benchmark.save_results(results, args.save_results)


if __name__ == "__main__":
    main()
