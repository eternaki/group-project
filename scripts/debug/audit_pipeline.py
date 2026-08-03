#!/usr/bin/env python3
"""
Audyt pipeline: gdzie tracimy klatki i jakiej jakości są wyniki na każdym etapie.

Mierzy na prawdziwych wideo:
1. Detekcja psa — ile klatek bez psa, ile z wieloma psami (ryzyko podmiany psa
   między klatkami, bo delta AU liczy się względem klatki neutralnej JEDNEGO psa).
2. Keypoints — rozkład pewności, ile poniżej progu.
3. Poza głowy — rozkład |yaw_asymmetry| / |roll|, ile poza limitem.
4. Klatka neutralna — który indeks wybrany, jaka frontalność.
5. AU — ile klamrowanych (niewiarygodnych), ile aktywnych, rozkład confidence.
6. Emocje — jakie reguły się odpalają.

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
from packages.pipeline import InferencePipeline, PipelineConfig

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@dataclass
class AuditStats:
    """Zliczenia dla całego audytu."""

    videos: int = 0
    videos_failed: int = 0
    frames_total: int = 0
    frames_no_dog: int = 0
    frames_multi_dog: int = 0
    frames_no_keypoints: int = 0
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


def audit_detection(pipeline: InferencePipeline, frames: list[np.ndarray], stats: AuditStats) -> None:
    """Etap 1: detekcja psa — braki i wielopsie klatki."""
    for frame in frames:
        detections = pipeline.bbox_model.predict(frame)
        if not detections:
            stats.frames_no_dog += 1
            continue
        if len(detections) > 1:
            stats.frames_multi_dog += 1
        stats.dog_conf.append(float(detections[0].confidence))


def audit_frame_data(all_frames_data: list[dict], stats: AuditStats) -> None:
    """Etapy 2-3 i 5: keypoints, poza głowy, AU."""
    # Rozrzut ratio w obrębie JEDNEGO wideo = "podłoga szumu". Jeśli jest większy
    # od progu aktywacji (1.15), AU zapala się od drgań keypoints, nie od mimiki.
    ratios_in_video: dict[str, list[float]] = {}

    for data in all_frames_data:
        keypoints = data.get("keypoints")
        if keypoints is None:
            stats.frames_no_keypoints += 1
            continue

        visibility = np.asarray(keypoints).reshape(-1, 3)[:, 2]
        stats.kp_conf.append(float(np.mean(visibility)))

        pose = data.get("head_pose")
        if pose is not None:
            stats.yaw_asymmetry.append(abs(float(pose.yaw_asymmetry)))
            stats.roll.append(abs(float(pose.roll)))

        delta_aus = data.get("delta_aus")
        if not delta_aus:
            continue
        stats.au_frames += 1
        for name, au in delta_aus.items():
            if au.ratio >= RATIO_CLAMP_MAX - 1e-6 or au.ratio <= RATIO_CLAMP_MIN + 1e-6:
                stats.au_clamped[name] += 1
            if au.is_active:
                stats.au_active[name] += 1
            stats.au_conf.setdefault(name, []).append(float(au.confidence))
            ratios_in_video.setdefault(name, []).append(float(au.ratio))

    for name, ratios in ratios_in_video.items():
        if len(ratios) >= 3:
            stats.au_noise.setdefault(name, []).append(float(np.std(ratios)))


def audit_video(
    pipeline: InferencePipeline,
    video_path: Path,
    stats: AuditStats,
    fps: float,
    max_frames: int,
    num_peaks: int,
) -> None:
    """Przetwarza jedno wideo i dopisuje statystyki."""
    frames = extract_frames(video_path, fps, max_frames)
    if not frames:
        stats.videos_failed += 1
        return

    stats.frames_total += len(frames)
    audit_detection(pipeline, frames, stats)

    result = pipeline.process_video_for_dataset(
        frames_list=frames,
        num_peaks=num_peaks,
    )

    audit_frame_data(result.get("all_frames_data", []), stats)

    neutral_idx = result.get("neutral_frame_idx")
    neutral_data = next(
        (d for d in result.get("all_frames_data", []) if d["frame_idx"] == neutral_idx),
        None,
    )
    if neutral_data is not None and neutral_data.get("head_pose") is not None:
        pose = neutral_data["head_pose"]
        stats.neutral_frontality.append(
            abs(float(pose.yaw_asymmetry)) / 0.35 + abs(float(pose.roll)) / 30.0
        )

    for peak in result.get("peak_frames", []):
        stats.peaks += 1
        emotion = peak["emotion"]
        stats.emotions[emotion.emotion] += 1
        stats.rules[emotion.rule_applied or "-"] += 1


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

    return {
        "wideo": {
            "przetworzone": stats.videos,
            "nieudane": stats.videos_failed,
            "czas_s": round(elapsed, 1),
            "s_na_wideo": round(elapsed / max(stats.videos, 1), 1),
        },
        "etap_1_detekcja_psa": {
            "klatki": stats.frames_total,
            "bez_psa_proc": round(stats.frames_no_dog / frames * 100, 1),
            "wiele_psow_proc": round(stats.frames_multi_dog / frames * 100, 1),
            "confidence": percentiles(stats.dog_conf),
        },
        "etap_2_keypoints": {
            "bez_keypoints_proc": round(stats.frames_no_keypoints / frames * 100, 1),
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
                    if y > 0.35 or r > 30
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
                sum(stats.au_clamped.values()) / (au_frames * 21) * 100, 1
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

    stats = AuditStats()
    start = time.time()

    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video_path.name}")
        try:
            audit_video(pipeline, video_path, stats, args.fps, args.max_frames, args.num_peaks)
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
