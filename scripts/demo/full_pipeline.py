#!/usr/bin/env python3
"""
Полный pipeline обработки изображения собаки.

Этапы для КАЖДОЙ обнаруженной собаки:
1. Детекция собак (Bounding Box) - YOLOv8
2. Обрезка изображения каждой собаки
3. Классификация породы - EfficientNet-B4
4. Детекция ключевых точек - SimpleBaseline (ResNet50)
5. Классификация эмоций - (будет добавлено)

На каждом этапе сохраняется промежуточное изображение.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Добавляем путь к packages
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from packages.models.bbox import BBoxConfig, BBoxModel, Detection
from packages.models.breed import BreedConfig, BreedModel, BreedPrediction
from packages.models.keypoints import KeypointsConfig, KeypointsModel, KeypointsPrediction


# Цвета для разных собак (до 10)
DOG_COLORS = [
    (0, 255, 0),    # Зелёный
    (255, 0, 0),    # Красный
    (0, 0, 255),    # Синий
    (255, 255, 0),  # Жёлтый
    (255, 0, 255),  # Пурпурный
    (0, 255, 255),  # Голубой
    (255, 128, 0),  # Оранжевый
    (128, 0, 255),  # Фиолетовый
    (0, 255, 128),  # Бирюзовый
    (255, 128, 128),# Розовый
]


@dataclass
class DogResult:
    """Результат обработки одной собаки."""
    dog_id: int
    detection: Detection
    cropped_image: np.ndarray
    breed: Optional[BreedPrediction] = None
    keypoints: Optional[KeypointsPrediction] = None
    emotion: Optional[str] = None  # Для будущего


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
    max_dogs: int = 10  # Максимум собак для обработки
    output_dir: Path = Path("test/pipeline_output")


class DogPipeline:
    """
    Полный pipeline для анализа изображений собак.
    Обрабатывает ВСЕ найденные собаки на изображении.

    Использование:
        pipeline = DogPipeline(config)
        pipeline.load_models()
        results = pipeline.process("path/to/dogs.jpg")
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
                max_detections=self.config.max_dogs,
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
        Обрабатывает ВСЕ найденные собаки.

        Args:
            image_path: Путь к изображению

        Returns:
            Словарь с результатами всех этапов для всех собак
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
            "num_dogs": 0,
            "dogs": [],
            "output_images": {},
        }

        # =============================================
        # ЭТАП 1: Детекция ВСЕХ собак (BBox)
        # =============================================
        print(f"\n{'─' * 40}")
        print("ЭТАП 1: Детекция собак")
        print("─" * 40)

        detections: list[Detection] = []

        if self.bbox_model is not None:
            detections = self.bbox_model.filter_dogs_only(original_np)

            if detections:
                print(f"✓ Найдено собак: {len(detections)}")
                for i, det in enumerate(detections):
                    color_name = ["зелёный", "красный", "синий", "жёлтый", "пурпурный",
                                  "голубой", "оранжевый", "фиолетовый", "бирюзовый", "розовый"][i % 10]
                    print(f"  Собака {i+1} ({color_name}): confidence={det.confidence:.1%}, bbox={det.bbox}")

                # Сохраняем изображение со ВСЕМИ bbox
                bbox_image = self._draw_all_bboxes(original_image.copy(), detections)
                bbox_path = self.config.output_dir / f"{base_name}_1_all_bboxes.jpg"
                bbox_image.save(bbox_path, quality=95)
                print(f"\n  Сохранено: {bbox_path}")
                results["output_images"]["all_bboxes"] = str(bbox_path)
            else:
                print("✗ Собаки не найдены")
                # Используем весь оригинал как одну "собаку"
                detections = [Detection(
                    bbox=(0, 0, original_image.width, original_image.height),
                    confidence=1.0,
                )]
        else:
            print("→ BBox модель недоступна, используем оригинал")
            detections = [Detection(
                bbox=(0, 0, original_image.width, original_image.height),
                confidence=1.0,
            )]

        results["num_dogs"] = len(detections)

        # =============================================
        # ОБРАБОТКА КАЖДОЙ СОБАКИ
        # =============================================
        dog_results: list[DogResult] = []

        for dog_id, detection in enumerate(detections):
            print(f"\n{'═' * 60}")
            print(f"СОБАКА {dog_id + 1} / {len(detections)}")
            print("═" * 60)

            # Обрезаем изображение собаки
            x, y, w, h = detection.bbox
            cropped_pil = original_image.crop((x, y, x + w, y + h))
            cropped_np = np.array(cropped_pil)

            dog_result = DogResult(
                dog_id=dog_id,
                detection=detection,
                cropped_image=cropped_np,
            )

            # Сохраняем обрезанное изображение
            cropped_path = self.config.output_dir / f"{base_name}_dog{dog_id + 1}_2_cropped.jpg"
            cropped_pil.save(cropped_path, quality=95)
            print(f"✓ Обрезано: {cropped_pil.size} → {cropped_path}")

            # ─────────────────────────────────────────
            # Классификация породы
            # ─────────────────────────────────────────
            if self.breed_model is not None:
                breed_pred = self.breed_model.predict(cropped_np)
                dog_result.breed = breed_pred

                print(f"✓ Порода: {breed_pred.class_name} ({breed_pred.confidence:.1%})")
                print(f"  Top-3: {', '.join([f'{n}:{c:.0%}' for _, n, c in breed_pred.top_k[:3]])}")

                # Сохраняем с породой
                breed_image = self._draw_breed(cropped_pil.copy(), breed_pred, dog_id)
                breed_path = self.config.output_dir / f"{base_name}_dog{dog_id + 1}_3_breed.jpg"
                breed_image.save(breed_path, quality=95)

            # ─────────────────────────────────────────
            # Детекция ключевых точек
            # ─────────────────────────────────────────
            if self.keypoints_model is not None:
                keypoints_pred = self.keypoints_model.predict(cropped_np)
                dog_result.keypoints = keypoints_pred

                print(f"✓ Keypoints: {keypoints_pred.num_detected}/46 ({keypoints_pred.confidence:.1%})")

                # Сохраняем с keypoints
                keypoints_image_np = self.keypoints_model.draw_keypoints(
                    cropped_np.copy(), keypoints_pred
                )
                keypoints_pil = Image.fromarray(keypoints_image_np)
                keypoints_path = self.config.output_dir / f"{base_name}_dog{dog_id + 1}_4_keypoints.jpg"
                keypoints_pil.save(keypoints_path, quality=95)

            # ─────────────────────────────────────────
            # Эмоции (placeholder)
            # ─────────────────────────────────────────
            dog_result.emotion = "не реализовано"

            dog_results.append(dog_result)

            # Добавляем в результаты
            results["dogs"].append({
                "dog_id": dog_id + 1,
                "bbox": detection.bbox,
                "confidence": detection.confidence,
                "breed": breed_pred.class_name if dog_result.breed else None,
                "breed_confidence": breed_pred.confidence if dog_result.breed else None,
                "keypoints_detected": keypoints_pred.num_detected if dog_result.keypoints else None,
            })

        # =============================================
        # ФИНАЛЬНАЯ ВИЗУАЛИЗАЦИЯ
        # =============================================
        print(f"\n{'─' * 40}")
        print("ФИНАЛЬНАЯ ВИЗУАЛИЗАЦИЯ")
        print("─" * 40)

        # Изображение со всеми аннотациями
        final_annotated = self._draw_all_annotations(original_image.copy(), dog_results)
        annotated_path = self.config.output_dir / f"{base_name}_5_annotated.jpg"
        final_annotated.save(annotated_path, quality=95)
        print(f"✓ Аннотированное: {annotated_path}")
        results["output_images"]["annotated"] = str(annotated_path)

        # Сводная таблица
        summary_image = self._create_summary_grid(original_image, dog_results, base_name)
        summary_path = self.config.output_dir / f"{base_name}_6_summary.jpg"
        summary_image.save(summary_path, quality=95)
        print(f"✓ Сводка: {summary_path}")
        results["output_images"]["summary"] = str(summary_path)

        # Итоговая статистика
        print(f"\n{'=' * 60}")
        print("ИТОГИ")
        print("=" * 60)
        print(f"Обработано собак: {len(dog_results)}")
        for dr in dog_results:
            breed_str = f"{dr.breed.class_name} ({dr.breed.confidence:.0%})" if dr.breed else "N/A"
            kp_str = f"{dr.keypoints.num_detected}/46" if dr.keypoints else "N/A"
            print(f"  Собака {dr.dog_id + 1}: {breed_str}, keypoints: {kp_str}")

        print(f"\n{'=' * 60}")
        print("ГОТОВО!")
        print("=" * 60)

        return results

    def _draw_all_bboxes(self, image: Image.Image, detections: list[Detection]) -> Image.Image:
        """Рисует ВСЕ bounding boxes на изображении."""
        draw = ImageDraw.Draw(image)

        for i, det in enumerate(detections):
            x, y, w, h = det.bbox
            color = DOG_COLORS[i % len(DOG_COLORS)]

            # Рисуем прямоугольник
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline=color,
                width=4,
            )

            # Подпись
            label = f"Dog {i+1}: {det.confidence:.0%}"
            # Фон для текста
            text_bbox = draw.textbbox((x, y - 25), label)
            draw.rectangle(
                [(text_bbox[0] - 2, text_bbox[1] - 2), (text_bbox[2] + 2, text_bbox[3] + 2)],
                fill=color
            )
            draw.text((x, y - 25), label, fill=(0, 0, 0))

        return image

    def _draw_breed(self, image: Image.Image, prediction: BreedPrediction, dog_id: int) -> Image.Image:
        """Добавляет информацию о породе на изображение."""
        draw = ImageDraw.Draw(image)
        color = DOG_COLORS[dog_id % len(DOG_COLORS)]

        # Фон для текста
        draw.rectangle([(0, 0), (image.width, 50)], fill=(0, 0, 0))

        # Текст
        text = f"Dog {dog_id + 1}: {prediction.class_name} ({prediction.confidence:.0%})"
        draw.text((10, 10), text, fill=color)

        return image

    def _draw_all_annotations(self, image: Image.Image, dog_results: list[DogResult]) -> Image.Image:
        """Рисует все аннотации на одном изображении."""
        draw = ImageDraw.Draw(image)

        for dr in dog_results:
            x, y, w, h = dr.detection.bbox
            color = DOG_COLORS[dr.dog_id % len(DOG_COLORS)]

            # BBox
            draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=3)

            # Подпись с породой
            if dr.breed:
                label = f"{dr.dog_id + 1}. {dr.breed.class_name} ({dr.breed.confidence:.0%})"
            else:
                label = f"Dog {dr.dog_id + 1}"

            text_bbox = draw.textbbox((x, y - 20), label)
            draw.rectangle(
                [(text_bbox[0] - 2, text_bbox[1] - 2), (text_bbox[2] + 2, text_bbox[3] + 2)],
                fill=color
            )
            draw.text((x, y - 20), label, fill=(0, 0, 0))

        return image

    def _create_summary_grid(
        self,
        original: Image.Image,
        dog_results: list[DogResult],
        base_name: str,
    ) -> Image.Image:
        """Создаёт сводное изображение со всеми собаками."""
        # Параметры сетки
        thumb_size = 200
        padding = 10
        header_height = 80

        num_dogs = len(dog_results)
        if num_dogs == 0:
            return original

        # Вычисляем размеры
        cols = min(num_dogs, 4)
        rows = (num_dogs + cols - 1) // cols

        grid_width = cols * (thumb_size + padding) + padding
        grid_height = rows * (thumb_size + header_height + padding) + padding

        # Масштабируем оригинал
        orig_ratio = min(400 / original.width, 400 / original.height)
        orig_resized = original.resize(
            (int(original.width * orig_ratio), int(original.height * orig_ratio)),
            Image.Resampling.LANCZOS,
        )

        total_width = orig_resized.width + padding + grid_width
        total_height = max(orig_resized.height, grid_height) + 60

        # Создаём итоговое изображение
        summary = Image.new("RGB", (total_width, total_height), (40, 40, 40))

        # Вставляем оригинал слева
        summary.paste(orig_resized, (padding, padding))

        # Вставляем миниатюры собак
        x_offset = orig_resized.width + padding * 2
        y_offset = padding

        draw = ImageDraw.Draw(summary)

        for i, dr in enumerate(dog_results):
            col = i % cols
            row = i // cols

            x = x_offset + col * (thumb_size + padding)
            y = y_offset + row * (thumb_size + header_height + padding)

            color = DOG_COLORS[dr.dog_id % len(DOG_COLORS)]

            # Миниатюра
            cropped_pil = Image.fromarray(dr.cropped_image)
            cropped_pil.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)

            # Центрируем миниатюру
            paste_x = x + (thumb_size - cropped_pil.width) // 2
            paste_y = y + header_height
            summary.paste(cropped_pil, (paste_x, paste_y))

            # Рамка
            draw.rectangle(
                [(x, y + header_height), (x + thumb_size, y + header_height + thumb_size)],
                outline=color,
                width=3,
            )

            # Заголовок
            draw.rectangle([(x, y), (x + thumb_size, y + header_height)], fill=color)

            # Текст
            dog_label = f"Dog {dr.dog_id + 1}"
            draw.text((x + 5, y + 5), dog_label, fill=(0, 0, 0))

            if dr.breed:
                breed_text = f"{dr.breed.class_name[:15]}"
                conf_text = f"{dr.breed.confidence:.0%}"
                draw.text((x + 5, y + 25), breed_text, fill=(0, 0, 0))
                draw.text((x + 5, y + 45), conf_text, fill=(0, 0, 0))

            if dr.keypoints:
                kp_text = f"KP: {dr.keypoints.num_detected}/46"
                draw.text((x + 5, y + 65), kp_text, fill=(0, 0, 0))

        # Нижняя строка с общей информацией
        info_y = total_height - 50
        draw.rectangle([(0, info_y), (total_width, total_height)], fill=(30, 30, 30))
        info_text = f"Dog FACS Pipeline | {base_name} | {num_dogs} dog(s) detected"
        draw.text((padding, info_y + 15), info_text, fill=(255, 255, 255))

        return summary


def main():
    """Главная функция демонстрации pipeline."""
    # Путь к тестовому изображению
    test_images = [
        Path("test/spruce-pets-200-types-of-dogs-45a7bd12aacf458cb2e77b841c41abe7.jpg"),
        Path("test/ShihTzu-original.jpeg"),
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
        bbox_weights=Path("models/yolov8m.pt"),
        breed_weights=Path("models/breed.pt"),
        keypoints_weights=Path("models/keypoints_best.pt"),
        output_dir=Path("test/pipeline_output"),
        max_dogs=10,
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
    for name, path in results["output_images"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
