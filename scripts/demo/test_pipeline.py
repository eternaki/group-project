#!/usr/bin/env python3
"""
Скрипт для тестирования InferencePipeline.

Демонстрирует полный флоу обработки изображения:
1. Детекция собак (BBox)
2. Классификация породы
3. Детекция keypoints
4. Классификация эмоций
5. Экспорт в COCO JSON
"""

import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from packages.pipeline import InferencePipeline, PipelineConfig
from packages.data import COCODataset


def main():
    """Главная функция тестирования."""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ INFERENCE PIPELINE")
    print("=" * 60)

    # Конфигурация
    config = PipelineConfig(
        bbox_weights=project_root / "models" / "yolov8m.pt",
        breed_weights=project_root / "models" / "breed.pt",
        keypoints_weights=project_root / "models" / "keypoints_best.pt",
        emotion_weights=project_root / "models" / "emotion.pt",
        breeds_json=project_root / "packages" / "models" / "breeds.json",
        device="cpu",  # Используем CPU для совместимости
        confidence_threshold=0.3,
        max_dogs=10,
    )

    # Создаём pipeline
    pipeline = InferencePipeline(config)
    pipeline.load()

    # Тестовые изображения
    test_images = [
        project_root / "test" / "ShihTzu-original.jpeg",
        project_root / "test" / "spruce-pets-200-types-of-dogs-45a7bd12aacf458cb2e77b841c41abe7.jpg",
    ]

    # Выходная директория
    output_dir = project_root / "test" / "pipeline_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # COCO dataset для экспорта
    coco_dataset = COCODataset()

    print("\n" + "=" * 60)
    print("ОБРАБОТКА ИЗОБРАЖЕНИЙ")
    print("=" * 60)

    for image_path in test_images:
        if not image_path.exists():
            print(f"\n! Пропуск: {image_path.name} не существует")
            continue

        print(f"\n{'='*60}")
        print(f"ИЗОБРАЖЕНИЕ: {image_path.name}")
        print("=" * 60)

        # Загружаем изображение
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  ! Не удалось загрузить: {image_path}")
            continue

        # Конвертируем BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Обрабатываем через pipeline
        result = pipeline.process_frame(image_rgb, frame_id=image_path.name)

        print(f"\nНайдено собак: {len(result.annotations)}")

        # Добавляем в COCO dataset
        image_id = coco_dataset.add_image(
            file_name=image_path.name,
            width=result.width,
            height=result.height,
        )

        # Выводим результаты для каждой собаки
        for ann in result.annotations:
            print(f"\n--- Собака #{ann.dog_id} ---")
            print(f"  BBox: {ann.bbox} (conf: {ann.bbox_confidence:.2f})")

            if ann.breed:
                print(f"  Порода: {ann.breed.class_name} ({ann.breed.confidence:.1%})")

            if ann.keypoints:
                print(f"  Keypoints: {ann.keypoints.num_detected}/46 (conf: {ann.keypoints.confidence:.2f})")

            if ann.emotion:
                print(f"  Эмоция: {ann.emotion.emotion.upper()} ({ann.emotion.confidence:.1%})")

            # Добавляем в COCO
            coco_dataset.add_annotation_from_dog(image_id, ann)

        # Визуализируем
        vis_image = pipeline.visualize(image_rgb, result)

        # Сохраняем визуализацию
        vis_path = output_dir / f"result_{image_path.stem}.jpg"
        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_path), vis_bgr)
        print(f"\nВизуализация сохранена: {vis_path}")

    # Сохраняем COCO JSON
    coco_path = output_dir / "annotations.json"
    coco_dataset.save(coco_path)

    # Валидация
    print("\n" + "=" * 60)
    print("ВАЛИДАЦИЯ COCO DATASET")
    print("=" * 60)
    validation = coco_dataset.validate()
    print(f"Валидный: {validation['valid']}")
    if validation['errors']:
        print(f"Ошибки: {validation['errors']}")
    if validation['warnings']:
        print(f"Предупреждения: {validation['warnings']}")

    # Статистика
    stats = coco_dataset.get_statistics()
    print(f"\nСтатистика:")
    print(f"  Изображений: {stats['total_images']}")
    print(f"  Аннотаций: {stats['total_annotations']}")
    if stats['breeds']:
        print(f"  Породы: {stats['breeds']}")
    if stats['emotions']:
        print(f"  Эмоции: {stats['emotions']}")

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    main()
