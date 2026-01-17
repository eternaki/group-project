#!/usr/bin/env python3
"""
Полный pipeline обработки изображения собаки.

Этапы:
1. Детекция собаки (Bounding Box) - YOLOv8
2. Обрезка изображения собаки
3. Классификация породы - EfficientNet-B4
4. Детекция ключевых точек - SimpleBaseline (ResNet50)
5. Классификация эмоций - (будет добавлено)

На каждом этапе сохраняется промежуточное изображение.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Добавляем путь к packages
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from packages.models.bbox import BBoxConfig, BBoxModel, Detection
from packages.models.breed import BreedConfig, BreedModel, BreedPrediction
from packages.models.keypoints import KeypointsConfig, KeypointsModel, KeypointsPrediction


@dataclass
class PipelineConfig:
    """Конфигурация pipeline."""

    # Пути к весам моделей
    bbox_weights: Path = Path("models/yolov8m.pt")  # Pretrained YOLO
    breed_weights: Path = Path("models/breed.pt")
    keypoints_weights: Path = Path("models/keypoints_best.pt")
    breeds_json: Path = Path("packages/models/breeds.json")

    # Параметры
    confidence_threshold: float = 0.3
    output_dir: Path = Path("test/pipeline_output")


class DogPipeline:
    """
    Полный pipeline для анализа изображений собак.

    Использование:
        pipeline = DogPipeline(config)
        pipeline.load_models()
        results = pipeline.process("path/to/dog.jpg")
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.bbox_model: Optional[BBoxModel] = None
        self.breed_model: Optional[BreedModel] = None
        self.keypoints_model: Optional[KeypointsModel] = None

        # Создаём директорию для выходных файлов
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def load_models(self) -> None:
        """Загружает все доступные модели."""
        print("=" * 60)
        print("ЗАГРУЗКА МОДЕЛЕЙ")
        print("=" * 60)

        # 1. BBox Model (YOLOv8)
        if self.config.bbox_weights.exists():
            print(f"\n[1/3] BBox: {self.config.bbox_weights}")
            bbox_config = BBoxConfig(
                weights_path=self.config.bbox_weights,
                confidence_threshold=self.config.confidence_threshold,
                device="cpu",
            )
            self.bbox_model = BBoxModel(bbox_config)
            self.bbox_model.load()
            print("      ✓ Загружено")
        else:
            print(f"\n[1/3] BBox: НЕ НАЙДЕНО ({self.config.bbox_weights})")
            print("      → Используем ручную обрезку")

        # 2. Breed Model
        if self.config.breed_weights.exists():
            print(f"\n[2/3] Breed: {self.config.breed_weights}")
            breed_config = BreedConfig(
                weights_path=self.config.breed_weights,
                labels_path=self.config.breeds_json,
                device="cpu",
            )
            self.breed_model = BreedModel(breed_config)
            self.breed_model.load()
            print("      ✓ Загружено")
        else:
            print(f"\n[2/3] Breed: НЕ НАЙДЕНО ({self.config.breed_weights})")

        # 3. Keypoints Model
        if self.config.keypoints_weights.exists():
            print(f"\n[3/3] Keypoints: {self.config.keypoints_weights}")
            keypoints_config = KeypointsConfig(
                weights_path=self.config.keypoints_weights,
                confidence_threshold=self.config.confidence_threshold,
            )
            self.keypoints_model = KeypointsModel(keypoints_config)
            self.keypoints_model.load()
            print("      ✓ Загружено")
        else:
            print(f"\n[3/3] Keypoints: НЕ НАЙДЕНО ({self.config.keypoints_weights})")

        print("\n" + "=" * 60)

    def process(self, image_path: str | Path) -> dict:
        """
        Обрабатывает изображение через весь pipeline.

        Args:
            image_path: Путь к изображению

        Returns:
            Словарь с результатами всех этапов
        """
        image_path = Path(image_path)
        base_name = image_path.stem

        print(f"\n{'=' * 60}")
        print(f"ОБРАБОТКА: {image_path.name}")
        print("=" * 60)

        # Загружаем изображение
        original_image = Image.open(image_path).convert("RGB")
        original_np = np.array(original_image)

        print(f"Размер изображения: {original_image.size}")

        results = {
            "image_path": str(image_path),
            "image_size": original_image.size,
            "stages": {},
        }

        # =============================================
        # ЭТАП 1: Детекция собаки (BBox)
        # =============================================
        print(f"\n{'─' * 40}")
        print("ЭТАП 1: Детекция собаки")
        print("─" * 40)

        detection: Optional[Detection] = None
        cropped_np: np.ndarray = original_np

        if self.bbox_model is not None:
            detections = self.bbox_model.filter_dogs_only(original_np)

            if detections:
                detection = detections[0]  # Берём первую (наиболее уверенную)

                print(f"✓ Найдено собак: {len(detections)}")
                print(f"  Лучшая детекция: confidence={detection.confidence:.2%}")
                print(f"  BBox: {detection.bbox}")

                # Сохраняем изображение с bbox
                bbox_image = self._draw_bbox(original_image.copy(), detection)
                bbox_path = self.config.output_dir / f"{base_name}_1_bbox.jpg"
                bbox_image.save(bbox_path, quality=95)
                print(f"  Сохранено: {bbox_path}")

                results["stages"]["bbox"] = {
                    "num_detections": len(detections),
                    "best_confidence": detection.confidence,
                    "bbox": detection.bbox,
                    "output_image": str(bbox_path),
                }

                # Обрезаем изображение
                x, y, w, h = detection.bbox
                cropped_pil = original_image.crop((x, y, x + w, y + h))
                cropped_np = np.array(cropped_pil)
            else:
                print("✗ Собаки не найдены, используем оригинал")
                results["stages"]["bbox"] = {"error": "Собаки не найдены"}
        else:
            print("→ BBox модель недоступна, используем оригинал")
            results["stages"]["bbox"] = {"skipped": True}

        # Сохраняем обрезанное изображение
        cropped_pil = Image.fromarray(cropped_np)
        cropped_path = self.config.output_dir / f"{base_name}_2_cropped.jpg"
        cropped_pil.save(cropped_path, quality=95)
        print(f"\n✓ Обрезанное изображение: {cropped_path}")
        print(f"  Размер: {cropped_pil.size}")
        results["stages"]["crop"] = {
            "size": cropped_pil.size,
            "output_image": str(cropped_path),
        }

        # =============================================
        # ЭТАП 2: Классификация породы
        # =============================================
        print(f"\n{'─' * 40}")
        print("ЭТАП 2: Классификация породы")
        print("─" * 40)

        if self.breed_model is not None:
            breed_pred = self.breed_model.predict(cropped_np)

            print(f"✓ Порода: {breed_pred.class_name}")
            print(f"  Уверенность: {breed_pred.confidence:.2%}")
            print(f"\n  Top-5 предсказания:")
            for i, (cid, name, conf) in enumerate(breed_pred.top_k, 1):
                print(f"    {i}. {name}: {conf:.2%}")

            # Сохраняем изображение с породой
            breed_image = self._draw_breed(cropped_pil.copy(), breed_pred)
            breed_path = self.config.output_dir / f"{base_name}_3_breed.jpg"
            breed_image.save(breed_path, quality=95)
            print(f"\n  Сохранено: {breed_path}")

            results["stages"]["breed"] = {
                "class_id": breed_pred.class_id,
                "class_name": breed_pred.class_name,
                "confidence": breed_pred.confidence,
                "top_k": breed_pred.top_k,
                "output_image": str(breed_path),
            }
        else:
            print("→ Breed модель недоступна")
            results["stages"]["breed"] = {"skipped": True}

        # =============================================
        # ЭТАП 3: Детекция ключевых точек
        # =============================================
        print(f"\n{'─' * 40}")
        print("ЭТАП 3: Детекция ключевых точек")
        print("─" * 40)

        if self.keypoints_model is not None:
            keypoints_pred = self.keypoints_model.predict(cropped_np)

            print(f"✓ Обнаружено keypoints: {keypoints_pred.num_detected} / 46")
            print(f"  Средняя уверенность: {keypoints_pred.confidence:.2%}")

            # Сохраняем изображение с keypoints
            keypoints_image_np = self.keypoints_model.draw_keypoints(
                cropped_np.copy(), keypoints_pred
            )
            keypoints_pil = Image.fromarray(keypoints_image_np)
            keypoints_path = self.config.output_dir / f"{base_name}_4_keypoints.jpg"
            keypoints_pil.save(keypoints_path, quality=95)
            print(f"  Сохранено: {keypoints_path}")

            results["stages"]["keypoints"] = {
                "num_detected": keypoints_pred.num_detected,
                "total": 46,
                "confidence": keypoints_pred.confidence,
                "output_image": str(keypoints_path),
            }
        else:
            print("→ Keypoints модель недоступна")
            results["stages"]["keypoints"] = {"skipped": True}

        # =============================================
        # ЭТАП 4: Классификация эмоций (будущее)
        # =============================================
        print(f"\n{'─' * 40}")
        print("ЭТАП 4: Классификация эмоций")
        print("─" * 40)
        print("→ Будет добавлено в Sprint 5")
        results["stages"]["emotions"] = {"not_implemented": True}

        # =============================================
        # ФИНАЛЬНОЕ КОМБИНИРОВАННОЕ ИЗОБРАЖЕНИЕ
        # =============================================
        print(f"\n{'─' * 40}")
        print("ФИНАЛЬНАЯ ВИЗУАЛИЗАЦИЯ")
        print("─" * 40)

        final_image = self._create_summary(
            original_image,
            cropped_pil,
            results,
        )
        final_path = self.config.output_dir / f"{base_name}_5_final.jpg"
        final_image.save(final_path, quality=95)
        print(f"✓ Сохранено: {final_path}")
        results["final_image"] = str(final_path)

        print(f"\n{'=' * 60}")
        print("ГОТОВО!")
        print("=" * 60)

        return results

    def _draw_bbox(self, image: Image.Image, detection: Detection) -> Image.Image:
        """Рисует bounding box на изображении."""
        draw = ImageDraw.Draw(image)
        x, y, w, h = detection.bbox

        # Рисуем прямоугольник
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline=(0, 255, 0),
            width=4,
        )

        # Подпись
        label = f"Dog: {detection.confidence:.1%}"
        draw.rectangle([(x, y - 30), (x + 150, y)], fill=(0, 255, 0))
        draw.text((x + 5, y - 25), label, fill=(0, 0, 0))

        return image

    def _draw_breed(self, image: Image.Image, prediction: BreedPrediction) -> Image.Image:
        """Добавляет информацию о породе на изображение."""
        draw = ImageDraw.Draw(image)

        # Фон для текста
        draw.rectangle([(0, 0), (image.width, 60)], fill=(0, 0, 0, 180))

        # Текст
        text = f"{prediction.class_name}: {prediction.confidence:.1%}"
        draw.text((10, 10), text, fill=(255, 255, 255))

        # Top-3
        top3_text = " | ".join([f"{name}: {conf:.0%}" for _, name, conf in prediction.top_k[:3]])
        draw.text((10, 35), top3_text, fill=(200, 200, 200))

        return image

    def _create_summary(
        self,
        original: Image.Image,
        cropped: Image.Image,
        results: dict,
    ) -> Image.Image:
        """Создаёт финальное комбинированное изображение."""
        # Определяем размеры
        max_height = 600

        # Масштабируем оригинал
        ratio = max_height / original.height
        orig_resized = original.resize(
            (int(original.width * ratio), max_height),
            Image.Resampling.LANCZOS,
        )

        # Масштабируем обрезанное
        crop_ratio = max_height / cropped.height
        crop_resized = cropped.resize(
            (int(cropped.width * crop_ratio), max_height),
            Image.Resampling.LANCZOS,
        )

        # Создаём итоговое изображение
        total_width = orig_resized.width + crop_resized.width + 20
        summary = Image.new("RGB", (total_width, max_height + 100), (30, 30, 30))

        # Вставляем изображения
        summary.paste(orig_resized, (0, 0))
        summary.paste(crop_resized, (orig_resized.width + 20, 0))

        # Добавляем текст
        draw = ImageDraw.Draw(summary)

        y_text = max_height + 10
        text_lines = ["Dog FACS Pipeline Results:"]

        if "breed" in results["stages"] and "class_name" in results["stages"]["breed"]:
            breed = results["stages"]["breed"]
            text_lines.append(f"  Breed: {breed['class_name']} ({breed['confidence']:.1%})")

        if "keypoints" in results["stages"] and "num_detected" in results["stages"]["keypoints"]:
            kp = results["stages"]["keypoints"]
            text_lines.append(f"  Keypoints: {kp['num_detected']}/{kp['total']} detected")

        for i, line in enumerate(text_lines):
            draw.text((10, y_text + i * 25), line, fill=(255, 255, 255))

        return summary


def main():
    """Главная функция демонстрации pipeline."""
    # Путь к тестовому изображению
    test_images = [
        Path("test/ShihTzu-original.jpeg"),
        Path("test/spruce-pets-200-types-of-dogs-45a7bd12aacf458cb2e77b841c41abe7.jpg"),
    ]

    # Находим первое доступное изображение
    image_path = None
    for path in test_images:
        if path.exists():
            image_path = path
            break

    if image_path is None:
        print("Тестовые изображения не найдены!")
        print("Ожидаемые пути:")
        for path in test_images:
            print(f"  - {path}")
        return

    # Конфигурация
    config = PipelineConfig(
        bbox_weights=Path("models/yolov8m.pt"),  # Если нет - пропускаем
        breed_weights=Path("models/breed.pt"),
        keypoints_weights=Path("models/keypoints_best.pt"),
        output_dir=Path("test/pipeline_output"),
    )

    # Создаём и запускаем pipeline
    pipeline = DogPipeline(config)
    pipeline.load_models()

    # Обрабатываем изображение
    results = pipeline.process(image_path)

    # Выводим пути к результатам
    print(f"\n{'=' * 60}")
    print("ВЫХОДНЫЕ ФАЙЛЫ:")
    print("=" * 60)
    for stage, data in results["stages"].items():
        if isinstance(data, dict) and "output_image" in data:
            print(f"  {stage}: {data['output_image']}")
    if "final_image" in results:
        print(f"  FINAL: {results['final_image']}")


if __name__ == "__main__":
    main()
