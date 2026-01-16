"""
Testy dla modelu detekcji psów (BBoxModel).

Uruchomienie:
    pytest tests/test_models/test_bbox.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from packages.models.bbox import BBoxConfig, BBoxModel, Detection


class TestDetection:
    """Testy dla klasy Detection."""

    def test_detection_creation(self) -> None:
        """Test tworzenia obiektu Detection."""
        det = Detection(
            bbox=(100, 200, 50, 60),
            confidence=0.95,
        )
        assert det.bbox == (100, 200, 50, 60)
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "dog"

    def test_detection_to_dict(self) -> None:
        """Test konwersji Detection do słownika."""
        det = Detection(
            bbox=(100, 200, 50, 60),
            confidence=0.95,
        )
        result = det.to_dict()

        assert result["bbox"] == [100, 200, 50, 60]
        assert result["confidence"] == 0.95
        assert result["class_id"] == 0
        assert result["class_name"] == "dog"

    def test_detection_to_coco(self) -> None:
        """Test konwersji Detection do formatu COCO."""
        det = Detection(
            bbox=(100, 200, 50, 60),
            confidence=0.95,
        )
        result = det.to_coco(image_id=1, annotation_id=42)

        assert result["id"] == 42
        assert result["image_id"] == 1
        assert result["category_id"] == 1
        assert result["bbox"] == [100, 200, 50, 60]
        assert result["area"] == 50 * 60
        assert result["iscrowd"] == 0
        assert result["score"] == 0.95


class TestBBoxConfig:
    """Testy dla klasy BBoxConfig."""

    def test_default_config(self) -> None:
        """Test domyślnej konfiguracji."""
        config = BBoxConfig(weights_path="models/bbox.pt")

        assert config.weights_path == Path("models/bbox.pt")
        assert config.device == "cuda"
        assert config.confidence_threshold == 0.5
        assert config.iou_threshold == 0.45
        assert config.max_detections == 10

    def test_custom_config(self) -> None:
        """Test niestandardowej konfiguracji."""
        config = BBoxConfig(
            weights_path="custom/path.pt",
            device="cpu",
            confidence_threshold=0.7,
            iou_threshold=0.5,
            max_detections=5,
        )

        assert config.weights_path == Path("custom/path.pt")
        assert config.device == "cpu"
        assert config.confidence_threshold == 0.7
        assert config.iou_threshold == 0.5
        assert config.max_detections == 5


class TestBBoxModel:
    """Testy dla klasy BBoxModel."""

    @pytest.fixture
    def config(self) -> BBoxConfig:
        """Fixture dla konfiguracji testowej."""
        return BBoxConfig(
            weights_path="models/bbox.pt",
            device="cpu",
            confidence_threshold=0.5,
        )

    @pytest.fixture
    def model(self, config: BBoxConfig) -> BBoxModel:
        """Fixture dla modelu (niezaładowanego)."""
        return BBoxModel(config)

    def test_model_creation(self, model: BBoxModel) -> None:
        """Test tworzenia modelu."""
        assert not model.is_loaded
        assert model.config.device == "cpu"

    def test_model_repr(self, model: BBoxModel) -> None:
        """Test reprezentacji tekstowej modelu."""
        repr_str = repr(model)
        assert "BBoxModel" in repr_str
        assert "bbox.pt" in repr_str
        assert "loaded=False" in repr_str

    def test_preprocess_valid_image(self, model: BBoxModel) -> None:
        """Test preprocessingu poprawnego obrazu."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = model.preprocess(image)
        assert result is image

    def test_preprocess_empty_image_raises(self, model: BBoxModel) -> None:
        """Test preprocessingu pustego obrazu."""
        image = np.array([])
        with pytest.raises(ValueError, match="pusty"):
            model.preprocess(image)

    def test_preprocess_wrong_dimensions_raises(self, model: BBoxModel) -> None:
        """Test preprocessingu obrazu o złych wymiarach."""
        image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match="3 wymiary"):
            model.preprocess(image)

    def test_preprocess_wrong_channels_raises(self, model: BBoxModel) -> None:
        """Test preprocessingu obrazu o złej liczbie kanałów."""
        image = np.random.randint(0, 255, (480, 640, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="3 lub 4 kanały"):
            model.preprocess(image)

    def test_predict_without_load_raises(self, model: BBoxModel) -> None:
        """Test predykcji bez załadowania modelu."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="nie został załadowany"):
            model.predict(image)

    def test_load_missing_weights_raises(self, model: BBoxModel) -> None:
        """Test ładowania nieistniejących wag."""
        model.config.weights_path = Path("nonexistent/model.pt")
        with pytest.raises(FileNotFoundError):
            model.load()

    def test_postprocess_empty_list(self, model: BBoxModel) -> None:
        """Test postprocessingu pustej listy."""
        result = model.postprocess([])
        assert result == []

    def test_postprocess_detections(self, model: BBoxModel) -> None:
        """Test postprocessingu detekcji."""
        detections = [
            Detection(bbox=(10, 20, 30, 40), confidence=0.9),
            Detection(bbox=(50, 60, 70, 80), confidence=0.8),
        ]
        result = model.postprocess(detections)

        assert len(result) == 2
        assert result[0]["confidence"] == 0.9
        assert result[1]["bbox"] == [50, 60, 70, 80]


class TestBBoxModelIntegration:
    """
    Testy integracyjne dla BBoxModel.

    Te testy wymagają zainstalowanego ultralytics i pobranego modelu.
    Są pomijane jeśli model nie jest dostępny.
    """

    @pytest.fixture
    def pretrained_config(self) -> BBoxConfig:
        """Konfiguracja z pretrenowanym modelem."""
        return BBoxConfig(
            weights_path="yolov8n.pt",  # Mały model do testów
            device="cpu",
            confidence_threshold=0.3,
            max_detections=5,
        )

    @pytest.mark.skipif(
        not Path("yolov8n.pt").exists(),
        reason="Pretrenowany model nie jest dostępny",
    )
    def test_load_pretrained_model(self, pretrained_config: BBoxConfig) -> None:
        """Test ładowania pretrenowanego modelu."""
        model = BBoxModel(pretrained_config)
        model.load()
        assert model.is_loaded

    @pytest.mark.skipif(
        not Path("yolov8n.pt").exists(),
        reason="Pretrenowany model nie jest dostępny",
    )
    def test_predict_on_random_image(self, pretrained_config: BBoxConfig) -> None:
        """Test predykcji na losowym obrazie."""
        model = BBoxModel(pretrained_config)
        model.load()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = model.predict(image)

        assert isinstance(detections, list)
        # Na losowym obrazie może nie być detekcji - to OK

    @pytest.mark.skipif(
        not Path("models/bbox.pt").exists(),
        reason="Fine-tunowany model nie jest dostępny",
    )
    def test_finetuned_model(self) -> None:
        """Test fine-tunowanego modelu."""
        config = BBoxConfig(
            weights_path="models/bbox.pt",
            device="cpu",
        )
        model = BBoxModel(config)
        model.load()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = model.predict(image)

        assert isinstance(detections, list)


class TestBBoxModelMocked:
    """Testy z mockowanym YOLO."""

    @pytest.fixture
    def mock_yolo(self):
        """Mock dla klasy YOLO."""
        with patch("packages.models.bbox.YOLO") as mock:
            # Przygotuj mock wyników
            mock_box = MagicMock()
            mock_box.xyxy = [MagicMock()]
            mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array(
                [100, 200, 150, 260]
            )
            mock_box.conf = [MagicMock()]
            mock_box.conf[0].cpu.return_value.numpy.return_value = 0.95
            mock_box.cls = [MagicMock()]
            mock_box.cls[0].cpu.return_value.numpy.return_value = 0

            mock_result = MagicMock()
            mock_result.boxes = [mock_box]

            mock_instance = MagicMock()
            mock_instance.return_value = [mock_result]

            mock.return_value = mock_instance

            yield mock

    def test_predict_with_mock(self, mock_yolo) -> None:
        """Test predykcji z mockowanym YOLO."""
        # Utwórz plik tymczasowy dla wag
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            weights_path = Path(f.name)

        try:
            config = BBoxConfig(
                weights_path=weights_path,
                device="cpu",
            )
            model = BBoxModel(config)
            model.load()

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = model.predict(image)

            assert len(detections) == 1
            assert detections[0].bbox == (100, 200, 50, 60)  # x, y, w, h
            assert detections[0].confidence == 0.95

        finally:
            weights_path.unlink(missing_ok=True)
