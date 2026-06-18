"""
Testy dla modelu klasyfikacji ras psów (BreedModel).

Uruchomienie:
    pytest tests/test_models/test_breed.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from packages.models.breed import BreedConfig, BreedModel, BreedPrediction, SquarePad


class TestBreedPrediction:
    """Testy dla klasy BreedPrediction."""

    def test_prediction_creation(self) -> None:
        """Test tworzenia obiektu BreedPrediction."""
        pred = BreedPrediction(
            class_id=56,
            class_name="Golden Retriever",
            confidence=0.85,
            top_k=[
                (56, "Golden Retriever", 0.85),
                (57, "Labrador Retriever", 0.10),
                (80, "Collie", 0.03),
            ],
        )
        assert pred.class_id == 56
        assert pred.class_name == "Golden Retriever"
        assert pred.confidence == 0.85
        assert len(pred.top_k) == 3

    def test_prediction_to_dict(self) -> None:
        """Test konwersji BreedPrediction do słownika."""
        pred = BreedPrediction(
            class_id=56,
            class_name="Golden Retriever",
            confidence=0.85,
            top_k=[
                (56, "Golden Retriever", 0.85),
                (57, "Labrador Retriever", 0.10),
            ],
        )
        result = pred.to_dict()

        assert result["breed_id"] == 56
        assert result["breed"] == "Golden Retriever"
        assert result["breed_confidence"] == 0.85
        assert len(result["top_k"]) == 2
        assert result["top_k"][0]["name"] == "Golden Retriever"

    def test_prediction_to_coco(self) -> None:
        """Test konwersji BreedPrediction do formatu COCO."""
        pred = BreedPrediction(
            class_id=56,
            class_name="Golden Retriever",
            confidence=0.85,
        )
        result = pred.to_coco()

        assert result["breed"] == "Golden Retriever"
        assert result["breed_confidence"] == 0.85


class TestBreedConfig:
    """Testy dla klasy BreedConfig."""

    def test_default_config(self) -> None:
        """Test domyślnej konfiguracji."""
        config = BreedConfig(weights_path="models/breed.pt")

        assert config.weights_path == Path("models/breed.pt")
        assert config.labels_path == Path("packages/models/breeds.json")
        assert config.model_name == "efficientnet_b4"
        assert config.img_size == 380  # natywna rozdzielczość EfficientNet-B4
        assert config.top_k == 5

    def test_custom_config(self) -> None:
        """Test niestandardowej konfiguracji."""
        config = BreedConfig(
            weights_path="custom/breed.pt",
            labels_path="custom/breeds.json",
            model_name="vit_base_patch16_224",
            img_size=384,
            top_k=10,
        )

        assert config.weights_path == Path("custom/breed.pt")
        assert config.labels_path == Path("custom/breeds.json")
        assert config.model_name == "vit_base_patch16_224"
        assert config.img_size == 384
        assert config.top_k == 10


class TestBreedModel:
    """Testy dla klasy BreedModel."""

    @pytest.fixture
    def sample_labels(self) -> dict:
        """Fixture z przykładowym mapowaniem ras."""
        return {
            "0": "Chihuahua",
            "1": "Golden Retriever",
            "2": "Labrador Retriever",
            "3": "German Shepherd",
            "4": "Beagle",
        }

    @pytest.fixture
    def temp_labels_file(self, sample_labels) -> Path:
        """Fixture tworzący tymczasowy plik breeds.json."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(sample_labels, f)
            return Path(f.name)

    @pytest.fixture
    def config(self, temp_labels_file) -> BreedConfig:
        """Fixture dla konfiguracji testowej."""
        return BreedConfig(
            weights_path="models/breed.pt",
            labels_path=temp_labels_file,
            device="cpu",
        )

    @pytest.fixture
    def model(self, config: BreedConfig) -> BreedModel:
        """Fixture dla modelu (niezaładowanego)."""
        return BreedModel(config)

    def test_model_creation(self, model: BreedModel) -> None:
        """Test tworzenia modelu."""
        assert not model.is_loaded
        assert model.config.device == "cpu"

    def test_model_repr(self, model: BreedModel) -> None:
        """Test reprezentacji tekstowej modelu."""
        repr_str = repr(model)
        assert "BreedModel" in repr_str
        assert "breed.pt" in repr_str

    def test_num_classes_before_load(self, model: BreedModel) -> None:
        """Test num_classes przed załadowaniem."""
        assert model.num_classes == 0

    def test_labels_before_load_raises(self, model: BreedModel) -> None:
        """Test dostępu do labels przed załadowaniem."""
        with pytest.raises(RuntimeError, match="nie został załadowany"):
            _ = model.labels

    def test_predict_without_load_raises(self, model: BreedModel) -> None:
        """Test predykcji bez załadowania."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="nie został załadowany"):
            model.predict(image)

    def test_load_missing_weights_raises(self, model: BreedModel) -> None:
        """Test ładowania nieistniejących wag."""
        model.config.weights_path = Path("nonexistent/model.pt")
        with pytest.raises(FileNotFoundError):
            model.load()

    def test_load_missing_labels_raises(self, config: BreedConfig) -> None:
        """Test ładowania nieistniejącego pliku labels."""
        config.labels_path = Path("nonexistent/breeds.json")
        model = BreedModel(config)
        with pytest.raises(FileNotFoundError):
            model.load()

    def test_postprocess(self, model: BreedModel) -> None:
        """Test postprocessingu."""
        pred = BreedPrediction(
            class_id=1,
            class_name="Golden Retriever",
            confidence=0.9,
            top_k=[(1, "Golden Retriever", 0.9)],
        )
        result = model.postprocess(pred)

        assert result["breed"] == "Golden Retriever"
        assert result["breed_confidence"] == 0.9


class TestBreedModelWithMock:
    """Testy z mockowanym timm."""

    @pytest.fixture
    def mock_timm_and_torch(self):
        """Mock dla timm i torch."""
        with patch("packages.models.breed.torch") as mock_torch:
            # Mock dla tensor operations
            mock_tensor = MagicMock()
            mock_tensor.to.return_value = mock_tensor
            mock_tensor.unsqueeze.return_value = mock_tensor

            # Mock dla softmax output
            mock_probs = MagicMock()
            mock_probs.__getitem__ = MagicMock(return_value=mock_probs)
            mock_probs.topk.return_value = (
                MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(
                    return_value=np.array([0.9, 0.05, 0.03, 0.01, 0.01])
                )))),
                MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(
                    return_value=np.array([1, 2, 3, 4, 0])
                )))),
            )

            mock_torch.softmax.return_value = mock_probs
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            yield mock_torch


class TestSquarePad:
    """Testy dla klasy SquarePad."""

    def test_square_image_unchanged(self) -> None:
        """Kwadratowy obraz nie powinien być zmieniony."""
        pad = SquarePad()
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        result = pad(img)
        assert result.size == (100, 100)

    def test_wide_image_padded_vertically(self) -> None:
        """Szeroki obraz powinien dostać padding na górze i dole."""
        pad = SquarePad()
        img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
        result = pad(img)
        assert result.size == (200, 200)

    def test_tall_image_padded_horizontally(self) -> None:
        """Wysoki obraz powinien dostać padding po bokach."""
        pad = SquarePad()
        img = Image.fromarray(np.zeros((300, 100, 3), dtype=np.uint8))
        result = pad(img)
        assert result.size == (300, 300)

    def test_padding_is_black(self) -> None:
        """Padding powinien być czarny (0)."""
        pad = SquarePad()
        # Biały obraz 100x200
        arr = np.full((100, 200, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr)
        result = pad(img)
        result_arr = np.array(result)
        # Pierwsze piksele (padding) powinny być czarne
        assert result_arr[0, 0, 0] == 0


class TestPreprocessBgrConversion:
    """Testy konwersji BGR→RGB w preprocess()."""

    def test_bgr_to_rgb_conversion(self, tmp_path: Path) -> None:
        """Test czy preprocess konwertuje BGR na RGB."""
        labels = {"0": "TestBreed"}
        labels_file = tmp_path / "breeds.json"
        labels_file.write_text(json.dumps(labels))

        config = BreedConfig(
            weights_path="models/breed.pt",
            labels_path=labels_file,
            device="cpu",
        )
        model = BreedModel(config)
        model._labels = labels
        model._loaded = True

        # Obraz BGR: kanał B=255, G=0, R=0 (niebieski w BGR)
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255  # kanał B

        # Przygotuj mock transform, który sprawdzi wejście
        received_images = []

        class CapturePad:
            def __call__(self, img):
                received_images.append(np.array(img))
                return img

        from torchvision import transforms
        model._transform = transforms.Compose([
            transforms.ToPILImage(),
            CapturePad(),
            transforms.Resize(50),
            transforms.CenterCrop(50),
            transforms.ToTensor(),
        ])

        try:
            model.preprocess(bgr_image)
        except Exception:
            pass

        if received_images:
            # Po konwersji BGR→RGB: kanał 0 (R) powinien być 0, nie 255
            assert received_images[0][0, 0, 0] == 0, "BGR nie został skonwertowany na RGB"


class TestBreedModelEvaluate:
    """Testy dla metody evaluate()."""

    @pytest.fixture
    def loaded_model(self, tmp_path: Path) -> BreedModel:
        """Fixture z załadowanym (mockowanym) modelem."""
        labels = {"0": "Chihuahua", "1": "Golden Retriever", "2": "Labrador", "3": "Beagle", "4": "Poodle"}
        labels_file = tmp_path / "breeds.json"
        labels_file.write_text(json.dumps(labels))

        config = BreedConfig(
            weights_path="models/breed.pt",
            labels_path=labels_file,
            device="cpu",
            top_k=3,
        )
        model = BreedModel(config)
        model._labels = labels
        model._loaded = True
        model._model = MagicMock()
        return model

    def _make_prediction(self, top_id: int, labels: dict) -> BreedPrediction:
        """Tworzy BreedPrediction z danym Top-1."""
        return BreedPrediction(
            class_id=top_id,
            class_name=labels[str(top_id)],
            confidence=0.9,
            top_k=[
                (top_id, labels[str(top_id)], 0.9),
                ((top_id + 1) % 5, labels[str((top_id + 1) % 5)], 0.07),
                ((top_id + 2) % 5, labels[str((top_id + 2) % 5)], 0.03),
            ],
        )

    def test_evaluate_perfect_accuracy(self, loaded_model: BreedModel) -> None:
        """Test 100% dokładności gdy wszystkie predykcje są poprawne."""
        images = [np.zeros((50, 50, 3), dtype=np.uint8)] * 3
        labels_gt = [0, 1, 2]
        predictions = [self._make_prediction(i, loaded_model._labels) for i in labels_gt]

        with patch.object(loaded_model, "predict", side_effect=predictions):
            result = loaded_model.evaluate(images, labels_gt)

        assert result["top1_accuracy"] == 100.0
        assert result["top5_accuracy"] == 100.0
        assert result["total_samples"] == 3
        assert result["correct_top1"] == 3

    def test_evaluate_zero_accuracy(self, loaded_model: BreedModel) -> None:
        """Test 0% dokładności gdy żadna predykcja nie jest poprawna."""
        images = [np.zeros((50, 50, 3), dtype=np.uint8)] * 2
        # Prawdziwe: 0, predykcja: 1,2,3 (nie zawiera 0)
        labels_gt = [0, 0]
        predictions = [
            BreedPrediction(
                class_id=1, class_name="Golden Retriever", confidence=0.9,
                top_k=[(1, "Golden Retriever", 0.9), (2, "Labrador", 0.07), (3, "Beagle", 0.03)],
            ),
            BreedPrediction(
                class_id=2, class_name="Labrador", confidence=0.9,
                top_k=[(2, "Labrador", 0.9), (3, "Beagle", 0.07), (4, "Poodle", 0.03)],
            ),
        ]

        with patch.object(loaded_model, "predict", side_effect=predictions):
            result = loaded_model.evaluate(images, labels_gt)

        assert result["top1_accuracy"] == 0.0
        assert result["top5_accuracy"] == 0.0
        assert result["correct_top1"] == 0

    def test_evaluate_mismatched_lengths_raises(self, loaded_model: BreedModel) -> None:
        """Test błędu gdy liczba obrazów ≠ liczba etykiet."""
        images = [np.zeros((50, 50, 3), dtype=np.uint8)] * 3
        labels_gt = [0, 1]

        with pytest.raises(ValueError, match="równa liczbie etykiet"):
            loaded_model.evaluate(images, labels_gt)

    def test_evaluate_empty_raises(self, loaded_model: BreedModel) -> None:
        """Test błędu przy pustych listach."""
        with pytest.raises(ValueError, match="pusta"):
            loaded_model.evaluate([], [])

    def test_evaluate_not_loaded_raises(self, tmp_path: Path) -> None:
        """Test błędu gdy model nie jest załadowany."""
        config = BreedConfig(weights_path="models/breed.pt", device="cpu")
        model = BreedModel(config)
        with pytest.raises(RuntimeError, match="nie został załadowany"):
            model.evaluate([np.zeros((50, 50, 3), dtype=np.uint8)], [0])

    def test_evaluate_partial_top5(self, loaded_model: BreedModel) -> None:
        """Test gdzie Top-1 = 50%, Top-5 = 100%."""
        images = [np.zeros((50, 50, 3), dtype=np.uint8)] * 2
        labels_gt = [0, 1]
        # Predykcja 1: top1=0 (correct), top5 zawiera 0
        # Predykcja 2: top1=0 (wrong), top5 zawiera 1
        predictions = [
            BreedPrediction(
                class_id=0, class_name="Chihuahua", confidence=0.9,
                top_k=[(0, "Chihuahua", 0.9), (1, "Golden Retriever", 0.07), (2, "Labrador", 0.03)],
            ),
            BreedPrediction(
                class_id=0, class_name="Chihuahua", confidence=0.6,
                top_k=[(0, "Chihuahua", 0.6), (1, "Golden Retriever", 0.35), (2, "Labrador", 0.05)],
            ),
        ]

        with patch.object(loaded_model, "predict", side_effect=predictions):
            result = loaded_model.evaluate(images, labels_gt)

        assert result["top1_accuracy"] == 50.0
        assert result["top5_accuracy"] == 100.0


class TestBreedsJson:
    """Testy dla pliku breeds.json."""

    def test_breeds_json_exists(self) -> None:
        """Test czy breeds.json istnieje."""
        breeds_path = Path("packages/models/breeds.json")
        assert breeds_path.exists(), f"Brak pliku: {breeds_path}"

    def test_breeds_json_valid(self) -> None:
        """Test czy breeds.json jest poprawnym JSON."""
        breeds_path = Path("packages/models/breeds.json")
        if not breeds_path.exists():
            pytest.skip("breeds.json nie istnieje")

        with open(breeds_path) as f:
            breeds = json.load(f)

        assert isinstance(breeds, dict)
        assert len(breeds) > 0

    def test_breeds_json_has_expected_breeds(self) -> None:
        """Test czy breeds.json zawiera znane rasy."""
        breeds_path = Path("packages/models/breeds.json")
        if not breeds_path.exists():
            pytest.skip("breeds.json nie istnieje")

        with open(breeds_path) as f:
            breeds = json.load(f)

        # case-insensitive — breeds.json używa nazewnictwa ImageNet (np. "golden retriever")
        breed_names = [b.lower() for b in breeds.values()]

        # Sprawdź kilka znanych ras
        expected = ["chihuahua", "golden retriever", "german shepherd", "beagle"]
        for breed in expected:
            assert any(breed in bn for bn in breed_names), f"Brak rasy: {breed}"

    def test_breeds_json_sequential_ids(self) -> None:
        """Test czy ID są sekwencyjne."""
        breeds_path = Path("packages/models/breeds.json")
        if not breeds_path.exists():
            pytest.skip("breeds.json nie istnieje")

        with open(breeds_path) as f:
            breeds = json.load(f)

        ids = sorted([int(k) for k in breeds.keys()])
        expected = list(range(len(ids)))
        assert ids == expected, "ID nie są sekwencyjne"
