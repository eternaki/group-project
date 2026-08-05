#!/usr/bin/env python3
"""
Audyt pipeline: gdzie tracimy klatki i jakiej jakości są wyniki na każdym etapie.

Mierzy na prawdziwych wideo:
0. Treki — ile psów przyjętych, ile odrzuconych i z jakich powodów.
1. Detekcja psa — ile klatek bez psa, ile z wieloma psami.
2. Keypoints — rozkład pewności, ile detekcji bez zmierzonej mordy.
3. Poza głowy — rozkład |yaw_asymmetry| / |roll|, ile poza limitem.
4. Klatka neutralna — jaka frontalność klatki neutralnej każdego treku.
5. AU — ile klamrowanych (niewiarygodnych), ile aktywnych, rozkład confidence, szum.
6. Emocje — jakie reguły się odpalają na klatkach szczytowych.

Użycie:
    python scripts/debug/audit_pipeline.py --limit 12 --fps 1
    python scripts/debug/audit_pipeline.py --videos-dir data/drive_dogs/DOGS --limit 30
"""

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from packages.models.delta_action_units import (
    MIN_AU_CONFIDENCE,
    RATIO_CLAMP_MAX,
    RATIO_CLAMP_MIN,
)
from packages.models.emotion import classify_emotion_from_delta_aus
from packages.models.head_pose import DEFAULT_MAX_ROLL, DEFAULT_MAX_YAW_ASYMMETRY
from packages.pipeline import InferencePipeline, PipelineConfig, VideoDatasetConfig
from packages.pipeline.track_processing import TrackResult

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# Liczba AU w DogFACS — dzielnik przy liczeniu udziału pomiarów klamrowanych
NUM_ACTION_UNITS = 21


@dataclass
class AuditStats:
    """Zliczenia dla całego audytu."""

    videos: int = 0
    videos_failed: int = 0
    frames_total: int = 0
    frames_no_dog: int = 0
    frames_multi_dog: int = 0
    detections_total: int = 0
    detections_no_face: int = 0
    tracks_accepted: int = 0
    tracks_rejected: int = 0
    track_reject_reasons: Counter = field(default_factory=Counter)
    dog_conf: list[float] = field(default_factory=list)
    kp_conf: list[float] = field(default_factory=list)
    yaw_asymmetry: list[float] = field(default_factory=list)
    roll: list[float] = field(default_factory=list)
    neutral_frontality: list[float] = field(default_factory=list)
    au_clamped: Counter = field(default_factory=Counter)
    au_active: Counter = field(default_factory=Counter)
    au_conf: dict[str, list[float]] = field(default_factory=dict)
    au_noise: dict[str, list[float]] = field(default_factory=dict)
    au_frames: int = 0
    peaks: int = 0
    emotions: Counter = field(default_factory=Counter)
    rules: Counter = field(default_factory=Counter)


def iter_videos(videos_dir: Path, limit: int) -> list[Path]:
    """Zwraca listę plików wideo (posortowaną, deterministyczną)."""
    files = sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    return files[:limit]


def extract_frames(video_path: Path, fps: float, max_frames: int) -> list[np.ndarray]:
    """Ekstrahuje klatki z wideo z zadanym próbkowaniem."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(int(video_fps / fps), 1)

    frames: list[np.ndarray] = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


class CountingBBoxModel:
    """
    Detektor psów zliczający swoje wyniki po drodze (etap 1 audytu).

    Statystyki etapu 1 zbierane są z TEGO SAMEGO przebiegu detektora, którego
    używa pipeline. Osobny przebieg audytu podwajał koszt (YOLO na każdej klatce
    dwa razy) i pozwalał rozjechać się licznikom liczonym jako różnica dwóch
    przebiegów — a to na tej różnicy opiera się „ile detekcji nie dało mordy".
    """

    def __init__(self, model: object, stats: AuditStats) -> None:
        """
        Args:
            model: Właściwy detektor psów
            stats: Zliczenia audytu (modyfikowane w miejscu)
        """
        self.model = model
        self.stats = stats

    def predict(self, frame: np.ndarray) -> list:
        """Zwraca detekcje psa, dopisując po drodze statystyki etapu 1."""
        detections = self.model.predict(frame)
        self.stats.detections_total += len(detections)
        if not detections:
            self.stats.frames_no_dog += 1
            return detections
        if len(detections) > 1:
            self.stats.frames_multi_dog += 1
        self.stats.dog_conf.append(float(detections[0].confidence))
        return detections


def audit_track(track: TrackResult, stats: AuditStats) -> None:
    """Etapy 2-5 dla jednego treku: keypoints, poza głowy, AU i szum ratio."""
    for frame in track.frames:
        visibility = np.asarray(frame.keypoints).reshape(-1, 3)[:, 2]
        stats.kp_conf.append(float(np.mean(visibility)))
        stats.yaw_asymmetry.append(abs(float(frame.head_pose.yaw_asymmetry)))
        stats.roll.append(abs(float(frame.head_pose.roll)))

        if not frame.delta_aus:
            continue
        stats.au_frames += 1
        for name, au in frame.delta_aus.items():
            if au.ratio >= RATIO_CLAMP_MAX - 1e-6 or au.ratio <= RATIO_CLAMP_MIN + 1e-6:
                stats.au_clamped[name] += 1
            if au.is_active:
                stats.au_active[name] += 1
            stats.au_conf.setdefault(name, []).append(float(au.confidence))

    # Szum ratio liczy już trek (z liczbą próbek obok) — nie liczymy go tu drugi raz,
    # bo miara liczona na wideo mieszałaby ze sobą różne psy.
    for name, noise in track.au_noise.items():
        stats.au_noise.setdefault(name, []).append(float(noise))


def audit_neutral_and_peaks(track: TrackResult, stats: AuditStats) -> None:
    """Etapy 4 i 6: frontalność klatki neutralnej treku i emocje na peakach."""
    peak_indices = set(track.peak_indices)

    for frame in track.frames:
        if frame.frame_idx == track.neutral_frame_idx:
            stats.neutral_frontality.append(
                abs(float(frame.head_pose.yaw_asymmetry)) / DEFAULT_MAX_YAW_ASYMMETRY
                + abs(float(frame.head_pose.roll)) / DEFAULT_MAX_ROLL
            )
        if frame.frame_idx not in peak_indices:
            continue

        stats.peaks += 1
        emotion = classify_emotion_from_delta_aus(frame.delta_aus)
        stats.emotions[emotion.emotion] += 1
        stats.rules[emotion.rule_applied or "-"] += 1


def audit_video(
    pipeline: InferencePipeline,
    video_path: Path,
    stats: AuditStats,
    config: VideoDatasetConfig,
    max_frames: int,
) -> None:
    """Przetwarza jedno wideo i dopisuje statystyki."""
    frames = extract_frames(video_path, config.fps, max_frames)
    if not frames:
        stats.videos_failed += 1
        return

    stats.frames_total += len(frames)
    detections_before = stats.detections_total

    # Detektor jest opakowany (patrz `main`), więc statystyki etapu 1 powstają
    # w tym samym przebiegu, w którym pipeline szuka psów
    result = pipeline.process_video_for_dataset(frames, config=config)

    detections_total = stats.detections_total - detections_before
    measured_frames = sum(
        len(track.frames) for track in result["tracks"] + result["rejected_tracks"]
    )
    stats.detections_no_face += max(0, detections_total - measured_frames)

    for track in result["tracks"]:
        stats.tracks_accepted += 1
        audit_track(track, stats)
        audit_neutral_and_peaks(track, stats)

    for track in result["rejected_tracks"]:
        stats.tracks_rejected += 1
        # Powód niesie liczby (np. "za mało klatek z mordą: 2 < 3") — do zliczania
        # bierzemy sam rodzaj powodu, inaczej każdy trek miałby własną kategorię.
        stats.track_reject_reasons[track.rejected_reason.split(":")[0]] += 1


def percentiles(values: list[float]) -> dict:
    """Kwantyle rozkładu (puste → zera)."""
    if not values:
        return {"n": 0}
    arr = np.asarray(values)
    return {
        "n": len(values),
        "p10": round(float(np.percentile(arr, 10)), 3),
        "median": round(float(np.median(arr)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "mean": round(float(arr.mean()), 3),
    }


def build_report(stats: AuditStats, elapsed: float) -> dict:
    """Buduje raport w formie słownika."""
    frames = max(stats.frames_total, 1)
    au_frames = max(stats.au_frames, 1)
    detections = max(stats.detections_total, 1)

    return {
        "wideo": {
            "przetworzone": stats.videos,
            "nieudane": stats.videos_failed,
            "czas_s": round(elapsed, 1),
            "s_na_wideo": round(elapsed / max(stats.videos, 1), 1),
        },
        "etap_0_treki": {
            "przyjete": stats.tracks_accepted,
            "odrzucone": stats.tracks_rejected,
            "powody": dict(stats.track_reject_reasons.most_common(5)),
        },
        "etap_1_detekcja_psa": {
            "klatki": stats.frames_total,
            "detekcje": stats.detections_total,
            "bez_psa_proc": round(stats.frames_no_dog / frames * 100, 1),
            "wiele_psow_proc": round(stats.frames_multi_dog / frames * 100, 1),
            "confidence": percentiles(stats.dog_conf),
        },
        "etap_2_keypoints": {
            "detekcje_bez_mordy_proc": round(stats.detections_no_face / detections * 100, 1),
            "confidence": percentiles(stats.kp_conf),
            "ponizej_0_5_proc": round(
                sum(1 for c in stats.kp_conf if c < 0.5) / max(len(stats.kp_conf), 1) * 100, 1
            ),
        },
        "etap_3_poza_glowy": {
            "yaw_asymmetry_abs": percentiles(stats.yaw_asymmetry),
            "roll_abs": percentiles(stats.roll),
            "poza_limitem_proc": round(
                sum(
                    1
                    for y, r in zip(stats.yaw_asymmetry, stats.roll)
                    if y > DEFAULT_MAX_YAW_ASYMMETRY or r > DEFAULT_MAX_ROLL
                )
                / max(len(stats.yaw_asymmetry), 1) * 100,
                1,
            ),
        },
        "etap_4_klatka_neutralna": {
            "suma_katow_abs": percentiles(stats.neutral_frontality),
        },
        "etap_5_au": {
            "klatki_z_au": stats.au_frames,
            "klamrowane_proc_srednio": round(
                sum(stats.au_clamped.values()) / (au_frames * NUM_ACTION_UNITS) * 100, 1
            ),
            "top_klamrowane": {
                name: round(count / au_frames * 100, 1)
                for name, count in stats.au_clamped.most_common(6)
            },
            "top_aktywne": {
                name: round(count / au_frames * 100, 1)
                for name, count in stats.au_active.most_common(6)
            },
            "au_nigdy_aktywne": sorted(set(stats.au_conf) - set(stats.au_active)),
            # Mediana odchylenia standardowego ratio w obrębie wideo.
            # Porównanie: próg aktywacji AU = 1.15 (czyli sygnał 0.15).
            "podloga_szumu_std_ratio": {
                name: round(float(np.median(values)), 3)
                for name, values in sorted(
                    stats.au_noise.items(),
                    key=lambda kv: -float(np.median(kv[1])),
                )
            },
            "ponizej_progu_conf_proc": round(
                sum(
                    1
                    for values in stats.au_conf.values()
                    for v in values
                    if v < MIN_AU_CONFIDENCE
                )
                / max(sum(len(v) for v in stats.au_conf.values()), 1) * 100,
                1,
            ),
        },
        "etap_6_emocje": {
            "peaki": stats.peaks,
            "rozklad": dict(stats.emotions.most_common()),
            "reguly": dict(stats.rules.most_common(8)),
        },
    }


def main() -> int:
    """Uruchamia audyt."""
    parser = argparse.ArgumentParser(description="Audyt jakości pipeline po etapach")
    parser.add_argument("--videos-dir", type=Path, default=Path("data/drive_dogs/DOGS"))
    parser.add_argument("--limit", type=int, default=12, help="Liczba wideo")
    parser.add_argument("--fps", type=float, default=1.0, help="Próbkowanie klatek")
    parser.add_argument("--max-frames", type=int, default=20, help="Limit klatek na wideo")
    parser.add_argument("--num-peaks", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("data/audit_pipeline.json"))
    args = parser.parse_args()

    videos = iter_videos(args.videos_dir, args.limit)
    if not videos:
        print(f"Brak wideo w {args.videos_dir}")
        return 1

    pipeline = InferencePipeline(PipelineConfig(device=args.device))
    pipeline.load()

    # Próbkowanie audytu musi trafić do konfiguracji: wchodzi w znaczniki czasu
    # filtra wygładzającego i w przeliczenie odstępu peaków na klatki.
    config = VideoDatasetConfig(num_peaks=args.num_peaks, fps=args.fps)

    stats = AuditStats()
    pipeline.bbox_model = CountingBBoxModel(pipeline.bbox_model, stats)
    start = time.time()

    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video_path.name}")
        try:
            audit_video(pipeline, video_path, stats, config, args.max_frames)
            stats.videos += 1
        except Exception as e:  # noqa: BLE001 — audyt nie może paść na jednym wideo
            print(f"  ! Błąd: {type(e).__name__}: {e}")
            stats.videos_failed += 1

    report = build_report(stats, time.time() - start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nZapisano: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
