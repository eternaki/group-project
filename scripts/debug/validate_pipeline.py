#!/usr/bin/env python3
"""Szybka walidacja pipeline na pojedynczym obrazie (dry-run end-to-end)."""

import sys
from pathlib import Path

import cv2

from packages.pipeline import InferencePipeline, PipelineConfig


def main() -> int:
    img_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/test_frame_raw.jpg")
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"BŁĄD: nie można wczytać {img_path}")
        return 1

    print(f"Obraz: {img_path} -> {image.shape}")

    pipeline = InferencePipeline(PipelineConfig(device="cpu"))
    pipeline.load()

    result = pipeline.process_frame(image, frame_id=img_path.stem)

    print("\n=== WYNIK ===")
    print(f"Wykryto psów: {len(result.annotations)}")
    for ann in result.annotations:
        print(f"\nPies #{ann.dog_id}: bbox={ann.bbox} conf={ann.bbox_confidence:.3f}")
        if ann.breed:
            print(f"  Rasa: {ann.breed.class_name} ({ann.breed.confidence:.3f})")
        if ann.keypoints:
            print(
                f"  Keypoints: {ann.keypoints.num_detected}/46 "
                f"conf={ann.keypoints.confidence:.3f}"
            )
        if ann.emotion:
            print(f"  Emocja: {ann.emotion.emotion} ({ann.emotion.confidence:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
