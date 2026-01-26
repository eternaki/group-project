"""
Testy dla modułu delta action units (DeltaActionUnit, DeltaActionUnitsExtractor).

Uruchomienie:
    pytest tests/test_models/test_delta_action_units.py -v
"""

import numpy as np
import pytest

from packages.models.delta_action_units import (
    DeltaActionUnit,
    DeltaActionUnitsExtractor,
    ACTION_UNIT_NAMES,
)


class TestDeltaActionUnit:
    """Testy dla klasy DeltaActionUnit."""

    def test_creation(self) -> None:
        """Test tworzenia obiektu DeltaActionUnit."""
        au = DeltaActionUnit(
            name="AU101",
            ratio=1.25,
            delta=0.25,
            is_active=True,
            confidence=0.9,
        )

        assert au.name == "AU101"
        assert au.ratio == 1.25
        assert au.delta == 0.25
        assert au.is_active is True
        assert au.confidence == 0.9

    def test_to_dict(self) -> None:
        """Test konwersji do słownika."""
        au = DeltaActionUnit(
            name="AU102",
            ratio=1.15,
            delta=0.15,
            is_active=True,
            confidence=0.85,
        )

        result = au.to_dict()

        assert result["name"] == "AU102"
        assert result["ratio"] == 1.15
        assert result["delta"] == 0.15
        assert result["is_active"] is True
        assert result["confidence"] == 0.85

    def test_no_activation(self) -> None:
        """Test AU bez aktywacji (delta = 0)."""
        au = DeltaActionUnit(
            name="AU12",
            ratio=1.0,
            delta=0.0,
            is_active=False,
            confidence=0.95,
        )

        assert au.ratio == 1.0
        assert au.delta == 0.0
        assert au.is_active is False

    def test_negative_delta(self) -> None:
        """Test AU z ujemną deltą (zmniejszenie)."""
        au = DeltaActionUnit(
            name="EAD103",
            ratio=0.85,
            delta=-0.15,
            is_active=False,
            confidence=0.9,
        )

        assert au.ratio == 0.85
        assert au.delta == -0.15
        assert au.is_active is False


class TestDeltaActionUnitsExtractor:
    """Testy dla klasy DeltaActionUnitsExtractor."""

    @pytest.fixture
    def neutral_keypoints(self) -> np.ndarray:
        """Fixture dla neutral keypoints (20 punktów × 3 wartości)."""
        # Prostokątna twarz psa z symetrycznymi wartościami
        kp = np.array([
            # left_eye, right_eye
            [100, 150, 0.95],
            [200, 150, 0.95],
            # nose
            [150, 200, 0.95],
            # left_ear_base, right_ear_base
            [80, 100, 0.9],
            [220, 100, 0.9],
            # left_ear_tip, right_ear_tip
            [70, 50, 0.85],
            [230, 50, 0.85],
            # left_mouth, right_mouth
            [120, 220, 0.9],
            [180, 220, 0.9],
            # upper_lip, lower_lip
            [150, 210, 0.9],
            [150, 230, 0.9],
            # chin
            [150, 250, 0.85],
            # forehead (estimated)
            [150, 120, 0.8],
            # left_eye_outer, right_eye_outer
            [90, 150, 0.9],
            [210, 150, 0.9],
            # left_eye_inner, right_eye_inner
            [110, 150, 0.9],
            [190, 150, 0.9],
            # nose_bridge (estimated)
            [150, 180, 0.85],
            # left_cheek, right_cheek (estimated)
            [100, 190, 0.8],
            [200, 190, 0.8],
        ], dtype=np.float32).flatten()  # 60 values

        return kp

    @pytest.fixture
    def extractor(self, neutral_keypoints: np.ndarray) -> DeltaActionUnitsExtractor:
        """Fixture dla extractor z neutral baseline."""
        return DeltaActionUnitsExtractor(neutral_keypoints)

    def test_extractor_initialization(
        self, neutral_keypoints: np.ndarray
    ) -> None:
        """Test inicjalizacji extractor."""
        extractor = DeltaActionUnitsExtractor(neutral_keypoints)

        assert extractor.neutral_keypoints is not None
        assert extractor.neutral_distances is not None
        assert len(extractor.neutral_distances) > 0

    def test_neutral_vs_neutral_no_change(
        self, neutral_keypoints: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: neutral vs neutral powinno dać ratio ~1.0 dla wszystkich AU."""
        delta_aus = extractor.extract(neutral_keypoints)

        assert len(delta_aus) == len(ACTION_UNIT_NAMES)

        for au_name, au in delta_aus.items():
            # Ratio powinno być ~1.0 (tolerancja 0.05 z powodu floating point)
            assert 0.95 <= au.ratio <= 1.05, f"{au_name} ratio = {au.ratio}"
            # Delta powinna być ~0.0
            assert abs(au.delta) < 0.05, f"{au_name} delta = {au.delta}"
            # Nie powinno być aktywacji
            assert au.is_active is False, f"{au_name} should not be active"

    def test_mouth_open_activation(
        self, neutral_keypoints: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: otwarcie ust powinno aktywować AU12 i AU26."""
        target_kp = neutral_keypoints.copy().reshape(20, 3)

        # Zwiększ dystans między upper_lip i lower_lip (otwarte usta)
        target_kp[9, 1] -= 5   # upper_lip: y - 5
        target_kp[10, 1] += 10  # lower_lip: y + 10
        target_kp[11, 1] += 15  # chin: y + 15 (jaw drop)

        target_kp_flat = target_kp.flatten()
        delta_aus = extractor.extract(target_kp_flat)

        # AU12 (Lip Corner Puller) lub AU26 (Jaw Drop) powinno być aktywne
        assert delta_aus["AU12"].delta > 0 or delta_aus["AU26"].delta > 0
        # Przynajmniej jeden AU powinien przekroczyć threshold
        assert delta_aus["AU12"].is_active or delta_aus["AU26"].is_active

    def test_ears_forward_activation(
        self, neutral_keypoints: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: przesunięcie uszu do przodu powinno aktywować EAD102."""
        target_kp = neutral_keypoints.copy().reshape(20, 3)

        # Przesuń ear_tip do przodu (zmniejsz y)
        target_kp[5, 1] -= 15  # left_ear_tip: y - 15
        target_kp[6, 1] -= 15  # right_ear_tip: y - 15

        target_kp_flat = target_kp.flatten()
        delta_aus = extractor.extract(target_kp_flat)

        # EAD102 (Ears Forward) powinno mieć dodatnią deltę
        assert delta_aus["EAD102"].delta > 0

    def test_ears_back_activation(
        self, neutral_keypoints: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: spłaszczenie uszu powinno aktywować EAD103."""
        target_kp = neutral_keypoints.copy().reshape(20, 3)

        # Spłaszcz uszy (zmniejsz ear height)
        # Przesuń ear_tip w kierunku ear_base
        target_kp[5, 1] += 30  # left_ear_tip: y + 30 (closer to base)
        target_kp[6, 1] += 30  # right_ear_tip: y + 30

        target_kp_flat = target_kp.flatten()
        delta_aus = extractor.extract(target_kp_flat)

        # EAD103 (Ears Flattener) powinno być aktywne
        assert delta_aus["EAD103"].delta > 0

    def test_custom_activation_threshold(
        self, neutral_keypoints: np.ndarray
    ) -> None:
        """Test: custom activation threshold."""
        extractor_strict = DeltaActionUnitsExtractor(
            neutral_keypoints, activation_threshold=1.25  # Wyższy próg
        )

        target_kp = neutral_keypoints.copy().reshape(20, 3)
        # Małe otwarcie ust (tylko 10% wzrost)
        target_kp[10, 1] += 5

        target_kp_flat = target_kp.flatten()
        delta_aus = extractor_strict.extract(target_kp_flat)

        # Z threshold=1.25 małe zmiany nie powinny aktywować AU
        active_count = sum(1 for au in delta_aus.values() if au.is_active)
        # Przy 10% wzroście (ratio ~1.10) < 1.25, więc nie powinno być aktywacji
        assert active_count == 0

    def test_confidence_from_keypoints(
        self, neutral_keypoints: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: confidence AU pochodzi z visibility keypoints."""
        target_kp = neutral_keypoints.copy().reshape(20, 3)

        # Ustaw niską visibility dla eye keypoints
        target_kp[0, 2] = 0.3  # left_eye visibility = 0.3
        target_kp[1, 2] = 0.3  # right_eye visibility = 0.3

        target_kp_flat = target_kp.flatten()
        delta_aus = extractor.extract(target_kp_flat)

        # AU związane z oczami powinny mieć niższą confidence
        # (AU115, AU116, AU117, AU121)
        for au_name in ["AU115", "AU116", "AU117", "AU121"]:
            assert delta_aus[au_name].confidence < 0.9

    def test_extract_returns_all_aus(
        self, neutral_keypoints: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: extract() zwraca wszystkie 12 AU."""
        delta_aus = extractor.extract(neutral_keypoints)

        assert len(delta_aus) == 12
        expected_aus = [
            "AU101", "AU102", "AU12", "AU115", "AU116", "AU117",
            "AU121", "EAD102", "EAD103", "AD19", "AD37", "AU26",
        ]
        for au_name in expected_aus:
            assert au_name in delta_aus

    def test_invalid_keypoints_shape_raises(
        self, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: nieprawidłowa liczba keypoints powinna rzucić wyjątek."""
        invalid_kp = np.zeros(30)  # Tylko 10 keypoints zamiast 20

        with pytest.raises((ValueError, IndexError)):
            extractor.extract(invalid_kp)

    def test_empty_keypoints_raises(
        self, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: puste keypoints powinny rzucić wyjątek."""
        empty_kp = np.array([])

        with pytest.raises((ValueError, IndexError)):
            extractor.extract(empty_kp)


class TestActionUnitNames:
    """Testy dla stałych z nazwami AU."""

    def test_action_unit_names_count(self) -> None:
        """Test: powinno być 12 AU."""
        assert len(ACTION_UNIT_NAMES) == 12

    def test_action_unit_names_unique(self) -> None:
        """Test: wszystkie nazwy AU powinny być unikalne."""
        assert len(ACTION_UNIT_NAMES) == len(set(ACTION_UNIT_NAMES))

    def test_action_unit_names_official_dogfacs(self) -> None:
        """Test: nazwy powinny być oficjalnymi kodami DogFACS."""
        expected = {
            "AU101", "AU102", "AU12", "AU115", "AU116", "AU117",
            "AU121", "EAD102", "EAD103", "AD19", "AD37", "AU26",
        }
        assert set(ACTION_UNIT_NAMES) == expected
