"""
Testy dla modułu delta action units (DeltaActionUnit, DeltaActionUnitsExtractor).

Uruchomienie:
    pytest tests/test_models/test_delta_action_units.py -v
"""

import numpy as np
import pytest

from packages.data.schemas import KP, NUM_KEYPOINTS
from packages.models.delta_action_units import (
    ACTION_UNIT_NAMES,
    NUM_ACTION_UNITS,
    RATIO_CLAMP_MAX,
    RATIO_CLAMP_MIN,
    DeltaActionUnit,
    DeltaActionUnitsExtractor,
    extract_delta_action_units,
)

# =============================================================================
# Fixture: realistyczna twarz psa (46 keypoints)
# =============================================================================

def make_dog_face_kp() -> np.ndarray:
    """
    Tworzy realistyczną tablicę 46 keypoints dla frontalnej twarzy psa.

    Układ: centrum (150, 150), odległość oczu ~100px.
    Wszystkie keypoints widoczne (visibility=0.95).

    Returns:
        Tablica (138,) z wartościami [x0, y0, v0, x1, y1, v1, ...]
    """
    kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    kp[:, 2] = 0.95  # Domyślna widoczność

    # Oczy (0-7): kąty i powieki
    kp[KP.LEFT_EYE_INNER]  = [105, 150, 0.95]
    kp[KP.LEFT_EYE_TOP]    = [100, 144, 0.95]
    kp[KP.LEFT_EYE_OUTER]  = [95, 150, 0.95]
    kp[KP.LEFT_EYE_BOTTOM] = [100, 156, 0.95]
    kp[KP.RIGHT_EYE_INNER] = [195, 150, 0.95]
    kp[KP.RIGHT_EYE_TOP]   = [200, 144, 0.95]
    kp[KP.RIGHT_EYE_OUTER] = [205, 150, 0.95]
    kp[KP.RIGHT_EYE_BOTTOM]= [200, 156, 0.95]

    # Brwi (8-13)
    kp[KP.LEFT_BROW_INNER]  = [108, 137, 0.9]
    kp[KP.LEFT_BROW_CENTER] = [100, 134, 0.9]
    kp[KP.LEFT_BROW_OUTER]  = [92, 137, 0.9]
    kp[KP.RIGHT_BROW_INNER] = [192, 137, 0.9]
    kp[KP.RIGHT_BROW_CENTER]= [200, 134, 0.9]
    kp[KP.RIGHT_BROW_OUTER] = [208, 137, 0.9]

    # Uszy (14-21)
    kp[KP.LEFT_EAR_BASE_FRONT]  = [80, 120, 0.9]
    kp[KP.LEFT_EAR_BASE_BACK]   = [75, 115, 0.85]
    kp[KP.LEFT_EAR_MID]         = [72, 95, 0.85]
    kp[KP.LEFT_EAR_TIP]         = [68, 70, 0.8]
    kp[KP.RIGHT_EAR_BASE_FRONT] = [220, 120, 0.9]
    kp[KP.RIGHT_EAR_BASE_BACK]  = [225, 115, 0.85]
    kp[KP.RIGHT_EAR_MID]        = [228, 95, 0.85]
    kp[KP.RIGHT_EAR_TIP]        = [232, 70, 0.8]

    # Nos (22-25)
    kp[KP.NOSE_TIP]       = [150, 200, 0.95]
    kp[KP.NOSE_LEFT_WING] = [143, 196, 0.9]
    kp[KP.NOSE_RIGHT_WING]= [157, 196, 0.9]
    kp[KP.NOSE_BRIDGE]    = [150, 175, 0.9]

    # Usta (26-33)
    kp[KP.MOUTH_LEFT_CORNER]  = [120, 218, 0.9]
    kp[KP.UPPER_LIP_LEFT]     = [132, 214, 0.9]
    kp[KP.UPPER_LIP_CENTER]   = [150, 212, 0.9]
    kp[KP.UPPER_LIP_RIGHT]    = [168, 214, 0.9]
    kp[KP.MOUTH_RIGHT_CORNER] = [180, 218, 0.9]
    kp[KP.LOWER_LIP_RIGHT]    = [168, 226, 0.9]
    kp[KP.LOWER_LIP_CENTER]   = [150, 228, 0.9]
    kp[KP.LOWER_LIP_LEFT]     = [132, 226, 0.9]

    # Pysk (34-37)
    kp[KP.MUZZLE_TOP]   = [150, 207, 0.9]
    kp[KP.MUZZLE_LEFT]  = [122, 222, 0.85]
    kp[KP.MUZZLE_RIGHT] = [178, 222, 0.85]
    kp[KP.CHIN]         = [150, 250, 0.85]

    # Kontur (38-45)
    kp[KP.FOREHEAD_CENTER]  = [150, 115, 0.8]
    kp[KP.FOREHEAD_LEFT]    = [120, 118, 0.8]
    kp[KP.FOREHEAD_RIGHT]   = [180, 118, 0.8]
    kp[KP.LEFT_CHEEK_UPPER] = [88, 162, 0.8]
    kp[KP.LEFT_CHEEK_LOWER] = [88, 202, 0.8]
    kp[KP.RIGHT_CHEEK_UPPER]= [212, 162, 0.8]
    kp[KP.RIGHT_CHEEK_LOWER]= [212, 202, 0.8]
    kp[KP.JAW_CENTER]       = [150, 245, 0.85]

    return kp.flatten()


# =============================================================================
# Testy klasy DeltaActionUnit
# =============================================================================

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

    def test_to_dict_contains_all_fields(self) -> None:
        """Test: to_dict() zawiera wszystkie wymagane pola."""
        au = DeltaActionUnit(
            name="EAD101",
            ratio=1.15,
            delta=0.15,
            is_active=True,
            confidence=0.85,
        )

        result = au.to_dict()

        assert result["name"] == "EAD101"
        assert "ratio" in result
        assert "delta" in result
        assert "is_active" in result
        assert "confidence" in result

    def test_no_activation_at_neutral(self) -> None:
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
        """Test AU z ujemną deltą (zmniejszenie dystansu)."""
        au = DeltaActionUnit(
            name="EAD103",
            ratio=0.85,
            delta=-0.15,
            is_active=False,
            confidence=0.9,
        )

        assert au.ratio == 0.85
        assert au.delta == pytest.approx(-0.15)


# =============================================================================
# Testy klasy DeltaActionUnitsExtractor
# =============================================================================

class TestDeltaActionUnitsExtractor:
    """Testy dla klasy DeltaActionUnitsExtractor."""

    @pytest.fixture
    def neutral_kp(self) -> np.ndarray:
        """Fixture: neutralne keypoints (46 punktów × 3 wartości = 138)."""
        return make_dog_face_kp()

    @pytest.fixture
    def extractor(self, neutral_kp: np.ndarray) -> DeltaActionUnitsExtractor:
        """Fixture: ekstraktor z neutralną bazą."""
        return DeltaActionUnitsExtractor(neutral_kp)

    def test_initialization_succeeds(self, neutral_kp: np.ndarray) -> None:
        """Test: inicjalizacja z prawidłową liczbą keypoints."""
        extractor = DeltaActionUnitsExtractor(neutral_kp)
        assert extractor.neutral_kp is not None
        assert extractor.neutral_distances is not None

    def test_invalid_keypoints_count_raises(self) -> None:
        """Test: nieprawidłowa liczba wartości keypoints rzuca ValueError."""
        with pytest.raises(ValueError, match="138"):
            DeltaActionUnitsExtractor(np.zeros(60))  # Stare 20 keypoints

    def test_neutral_vs_neutral_ratio_is_one(
        self, neutral_kp: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: neutral vs neutral powinno dać ratio ≈ 1.0 dla wszystkich AU."""
        delta_aus = extractor.extract(neutral_kp)

        assert len(delta_aus) == NUM_ACTION_UNITS

        for au_name, au in delta_aus.items():
            assert 0.90 <= au.ratio <= 1.10, (
                f"{au_name} ratio = {au.ratio:.3f} (oczekiwano ≈ 1.0)"
            )

    def test_extract_returns_all_21_aus(
        self, neutral_kp: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: extract() zwraca wszystkie 21 AU."""
        delta_aus = extractor.extract(neutral_kp)

        assert len(delta_aus) == 21
        for au_name in ACTION_UNIT_NAMES:
            assert au_name in delta_aus, f"Brakuje AU: {au_name}"

    def test_jaw_drop_increases_au26(
        self, neutral_kp: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: opuszczenie żuchwy zwiększa ratio AU26 (Jaw Drop)."""
        target_kp = neutral_kp.copy().reshape(NUM_KEYPOINTS, 3)
        # Opuść podbródek o 20px
        target_kp[KP.CHIN, 1] += 20

        delta_aus = extractor.extract(target_kp.flatten())

        assert delta_aus["AU26"].ratio > 1.0, "AU26 ratio powinno wzrosnąć przy jaw drop"

    def test_ears_flatten_decreases_ead103(
        self, neutral_kp: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: spłaszczenie uszu zmniejsza ratio EAD103 (Ears Flattener).

        EAD103 jest AU aktywowanym przez zmniejszenie — wyższe uszy (mniejszy kąt)
        dają ratio = 1.0, spłaszczone uszy (czubki niżej względem podstawy) → ratio < 1.0.
        """
        target_kp = neutral_kp.copy().reshape(NUM_KEYPOINTS, 3)
        # Opuść czubki uszu (y+) → uszy spłaszczone bliżej głowy
        target_kp[KP.LEFT_EAR_TIP, 1] += 25
        target_kp[KP.RIGHT_EAR_TIP, 1] += 25

        delta_aus = extractor.extract(target_kp.flatten())

        assert delta_aus["EAD103"].ratio < 1.0, (
            "EAD103 ratio powinno zmaleć przy spłaszczeniu "
            "(AU aktywowany przez zmniejszenie)"
        )

    def test_lips_open_increases_au25(
        self, neutral_kp: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: rozchylenie warg zwiększa ratio AU25 (Lips Part)."""
        target_kp = neutral_kp.copy().reshape(NUM_KEYPOINTS, 3)
        # Rozsuń wargi: górna w górę, dolna w dół
        target_kp[KP.UPPER_LIP_CENTER, 1] -= 5
        target_kp[KP.LOWER_LIP_CENTER, 1] += 10

        delta_aus = extractor.extract(target_kp.flatten())

        assert delta_aus["AU25"].ratio > 1.0, "AU25 ratio powinno wzrosnąć przy rozchyleniu"

    def test_invalid_target_keypoints_raises(
        self, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: nieprawidłowa liczba wartości w target_kp rzuca ValueError."""
        with pytest.raises(ValueError):
            extractor.extract(np.zeros(30))

    def test_confidence_reflects_keypoint_visibility(
        self, neutral_kp: np.ndarray, extractor: DeltaActionUnitsExtractor
    ) -> None:
        """Test: confidence AU odzwierciedla widoczność keypoints."""
        target_kp = neutral_kp.copy().reshape(NUM_KEYPOINTS, 3)
        # Obniż widoczność oczu
        target_kp[KP.LEFT_EYE_TOP, 2] = 0.2
        target_kp[KP.LEFT_EYE_BOTTOM, 2] = 0.2
        target_kp[KP.RIGHT_EYE_TOP, 2] = 0.2
        target_kp[KP.RIGHT_EYE_BOTTOM, 2] = 0.2

        delta_aus = extractor.extract(target_kp.flatten())

        # AU143 (Lid Tightener) używa punktów powiek — confidence powinna być niska
        assert delta_aus["AU143"].confidence < 0.7

    def test_custom_activation_threshold(self, neutral_kp: np.ndarray) -> None:
        """Test: wyższy próg aktywacji oznacza mniej aktywnych AU."""
        strict_extractor = DeltaActionUnitsExtractor(
            neutral_kp, activation_threshold=1.50  # Bardzo wysoki próg
        )

        target_kp = neutral_kp.copy().reshape(NUM_KEYPOINTS, 3)
        target_kp[KP.CHIN, 1] += 5  # Mała zmiana

        delta_aus = strict_extractor.extract(target_kp.flatten())

        # Przy progu 1.50 mała zmiana nie powinna aktywować żadnego AU
        active_aus = [au for au in delta_aus.values() if au.is_active]
        assert len(active_aus) == 0


class TestExtractDeltaActionUnitsFunction:
    """Testy dla publicznej funkcji extract_delta_action_units."""

    def test_convenience_function_returns_all_aus(self) -> None:
        """Test: funkcja pomocnicza zwraca wszystkie 21 AU."""
        neutral_kp = make_dog_face_kp()
        target_kp = make_dog_face_kp()

        result = extract_delta_action_units(neutral_kp, target_kp)

        assert len(result) == NUM_ACTION_UNITS
        assert all(au_name in result for au_name in ACTION_UNIT_NAMES)

    def test_convenience_function_accepts_threshold(self) -> None:
        """Test: funkcja pomocnicza akceptuje custom próg."""
        neutral_kp = make_dog_face_kp()
        target_kp = make_dog_face_kp()

        result = extract_delta_action_units(
            neutral_kp, target_kp, activation_threshold=1.30
        )

        assert isinstance(result, dict)


# =============================================================================
# Testy stałych ACTION_UNIT_NAMES
# =============================================================================

class TestActionUnitNames:
    """Testy dla stałych z nazwami AU."""

    def test_action_unit_names_count_is_21(self) -> None:
        """Test: powinno być dokładnie 21 AU zgodnie z DogFACS."""
        assert len(ACTION_UNIT_NAMES) == 21

    def test_action_unit_names_unique(self) -> None:
        """Test: wszystkie nazwy AU są unikalne."""
        assert len(ACTION_UNIT_NAMES) == len(set(ACTION_UNIT_NAMES))

    def test_num_action_units_matches_list(self) -> None:
        """Test: NUM_ACTION_UNITS jest równe len(ACTION_UNIT_NAMES)."""
        assert NUM_ACTION_UNITS == len(ACTION_UNIT_NAMES)

    def test_official_dogfacs_upper_face_aus(self) -> None:
        """Test: górna twarz — AU101, AU143, AU145."""
        upper_face = {"AU101", "AU143", "AU145"}
        assert upper_face.issubset(set(ACTION_UNIT_NAMES))

    def test_official_dogfacs_lower_face_aus(self) -> None:
        """Test: dolna twarz — AU109, AU110, AU12, AU116, AU118, AU25, AU26, AU27."""
        lower_face = {"AU109", "AU110", "AU12", "AU116", "AU118", "AU25", "AU26", "AU27"}
        assert lower_face.issubset(set(ACTION_UNIT_NAMES))

    def test_official_dogfacs_action_descriptors(self) -> None:
        """Test: Action Descriptors — AD19, AD33, AD35, AD37, AD137."""
        ads = {"AD19", "AD33", "AD35", "AD37", "AD137"}
        assert ads.issubset(set(ACTION_UNIT_NAMES))

    def test_official_dogfacs_ear_descriptors(self) -> None:
        """Test: Ear Descriptors — EAD101, EAD102, EAD103, EAD104, EAD105."""
        eads = {"EAD101", "EAD102", "EAD103", "EAD104", "EAD105"}
        assert eads.issubset(set(ACTION_UNIT_NAMES))

    def test_no_invalid_au_codes(self) -> None:
        """Test: usunięte AU (nieistniejące w DogFACS) nie są obecne."""
        invalid_codes = {"AU102", "AU115", "AU117", "AU121"}
        for code in invalid_codes:
            assert code not in ACTION_UNIT_NAMES, (
                f"{code} nie istnieje w DogFACS i nie powinien być w ACTION_UNIT_NAMES"
            )


class TestClampedRatioUnreliable:
    """Klamrowanie stosunku = wyrodne pomiary (mały mianownik / skrajna zmiana).

    Najczęściej dotyczy uszu (EAD): punkty uszu są ruchome i słabo lokalizowane,
    więc stosunki odległości "wybuchają" do granic clampu i dawały fałszywe aktywne AU.
    Klamrowany pomiar uznajemy za niewiarygodny: is_active=False, confidence=0.
    """

    def test_clamped_max_ratio_not_active(self) -> None:
        """Stosunek dobity do RATIO_CLAMP_MAX → AU nieaktywny, confidence 0."""
        neutral = make_dog_face_kp().reshape(NUM_KEYPOINTS, 3).copy()
        # Końce uszu bardzo blisko siebie w neutralu → mały mianownik EAD105
        neutral[KP.LEFT_EAR_TIP] = [148, 70, 0.9]
        neutral[KP.RIGHT_EAR_TIP] = [152, 70, 0.9]
        target = make_dog_face_kp()  # normalne, rozstawione uszy → ogromny stosunek
        aus = DeltaActionUnitsExtractor(neutral.flatten()).extract(target)
        assert aus["EAD105"].ratio == pytest.approx(RATIO_CLAMP_MAX), "powinno dobić do MAX"
        assert aus["EAD105"].is_active is False, "klamrowany pomiar nie jest aktywny"
        assert aus["EAD105"].confidence == 0.0, "klamrowany pomiar ma confidence 0"

    def test_clamped_min_ratio_not_active(self) -> None:
        """Stosunek dobity do RATIO_CLAMP_MIN → AU nieaktywny, confidence 0."""
        neutral = make_dog_face_kp()  # normalne uszy
        target = make_dog_face_kp().reshape(NUM_KEYPOINTS, 3).copy()
        target[KP.LEFT_EAR_TIP] = [148, 70, 0.9]   # uszy zbite → maleńki licznik
        target[KP.RIGHT_EAR_TIP] = [152, 70, 0.9]
        aus = DeltaActionUnitsExtractor(neutral).extract(target.flatten())
        assert aus["EAD105"].ratio == pytest.approx(RATIO_CLAMP_MIN), "powinno dobić do MIN"
        assert aus["EAD105"].is_active is False
        assert aus["EAD105"].confidence == 0.0

    def test_unclamped_activation_unaffected(self) -> None:
        """Niewyklamrowana realna zmiana nadal daje aktywne AU z confidence."""
        neutral = make_dog_face_kp()
        target = make_dog_face_kp().reshape(NUM_KEYPOINTS, 3).copy()
        target[KP.CHIN, 1] += 20  # umiarkowane opuszczenie żuchwy (AU26), bez clampu
        aus = DeltaActionUnitsExtractor(neutral).extract(target.flatten())
        assert RATIO_CLAMP_MIN < aus["AU26"].ratio < RATIO_CLAMP_MAX
        assert aus["AU26"].is_active is True
        assert aus["AU26"].confidence > 0.0


class TestMedianBaseline:
    """Testy bazy median (wariant A+): wiele klatek neutralnych → median jako baza.

    Pojedyncza klatka neutralna bywa zaszumiona (drganie keypoints). Median z kilku
    klatek daje stabilną bazę i jest odporna na pojedynczy odstający kadr.
    """

    def test_list_of_identical_frames_equals_that_frame(self) -> None:
        """Lista identycznych klatek → baza równa tej klatce (sanity)."""
        frame = make_dog_face_kp()
        extractor = DeltaActionUnitsExtractor([frame, frame.copy(), frame.copy()])
        np.testing.assert_allclose(
            extractor.neutral_kp, frame.reshape(NUM_KEYPOINTS, 3), rtol=1e-5
        )

    def test_2d_array_neutral_accepted(self) -> None:
        """Wejście jako tablica 2D (N, 138) jest akceptowane jak lista klatek."""
        frame = make_dog_face_kp()
        stacked = np.stack([frame, frame.copy(), frame.copy()])  # (3, 138)
        extractor = DeltaActionUnitsExtractor(stacked)
        np.testing.assert_allclose(
            extractor.neutral_kp, frame.reshape(NUM_KEYPOINTS, 3), rtol=1e-5
        )

    def test_median_baseline_robust_to_single_outlier_frame(self) -> None:
        """Median ignoruje pojedynczy odstający kadr neutralny.

        4 czyste klatki (chin y=250) + 1 odstająca (chin y=400). Median chin y = 250.
        Klatka docelowa = czysta → AU26 (Jaw Drop) ratio ≈ 1.0, mimo outliera w bazie.
        """
        clean = make_dog_face_kp().reshape(NUM_KEYPOINTS, 3)
        outlier = clean.copy()
        outlier[KP.CHIN, 1] = 400  # drastycznie opuszczony podbródek (błąd detekcji)

        frames = [clean.flatten()] * 4 + [outlier.flatten()]
        extractor = DeltaActionUnitsExtractor(frames)

        delta_aus = extractor.extract(clean.flatten())
        assert delta_aus["AU26"].ratio == pytest.approx(1.0, abs=0.05), (
            "Median powinna odrzucić outlier — AU26 ≈ 1.0 dla czystej klatki docelowej"
        )

    def test_single_frame_backward_compatible(self) -> None:
        """Pojedyncza klatka (138,) działa jak dotychczas (kompatybilność wsteczna)."""
        frame = make_dog_face_kp()
        extractor = DeltaActionUnitsExtractor(frame)
        np.testing.assert_allclose(
            extractor.neutral_kp, frame.reshape(NUM_KEYPOINTS, 3), rtol=1e-5
        )

    def test_empty_neutral_list_raises(self) -> None:
        """Pusta lista klatek neutralnych rzuca ValueError."""
        with pytest.raises(ValueError, match="pusta|empty|co najmniej"):
            DeltaActionUnitsExtractor([])

    def test_neutral_frame_wrong_length_raises(self) -> None:
        """Klatka o złej długości w liście rzuca ValueError."""
        good = make_dog_face_kp()
        with pytest.raises(ValueError, match="138"):
            DeltaActionUnitsExtractor([good, np.zeros(60)])


class TestAUImprovements:
    """Testy ulepszeń AU (wariant A): AD33≠AD35, AD19 język, EAD104 rotacja."""

    def test_ad33_distinct_from_ad35(self) -> None:
        """AD33 (wydęte policzki) i AD35 (szczelina warg) reagują na RÓŻNE zmiany."""
        neutral = make_dog_face_kp()
        # Poszerz pysk (policzki), wargi bez zmian -> AD33 rośnie, AD35 ~bez zmian
        target = neutral.reshape(NUM_KEYPOINTS, 3).copy()
        target[KP.MUZZLE_LEFT][0] -= 25
        target[KP.MUZZLE_RIGHT][0] += 25
        aus = DeltaActionUnitsExtractor(neutral).extract(target.flatten())
        assert aus["AD33"].ratio > 1.15, "AD33 powinno wzrosnąć przy szerszym pysku"
        assert abs(aus["AD35"].ratio - 1.0) < 0.05, "AD35 nie powinno reagować na pysk"

    def test_ad35_lip_bite_on_closing_lips(self) -> None:
        """AD35 (Lip Bite) aktywuje się przy zmniejszeniu szczeliny warg."""
        neutral = make_dog_face_kp()
        target = neutral.reshape(NUM_KEYPOINTS, 3).copy()
        # Dolna warga blisko górnej (mała szczelina)
        target[KP.LOWER_LIP_CENTER][1] = target[KP.UPPER_LIP_CENTER][1] + 2
        aus = DeltaActionUnitsExtractor(neutral).extract(target.flatten())
        assert aus["AD35"].is_active, "AD35 powinno się aktywować przy zwartych wargach"

    def test_ad19_uses_tongue_keypoint(self) -> None:
        """AD19 (Tongue Show) reaguje na widoczny czubek języka poniżej wargi."""
        neutral = make_dog_face_kp().reshape(NUM_KEYPOINTS, 3).copy()
        neutral[KP.TONGUE_TIP] = [150, 228, 0.1]  # język schowany, niska widoczność
        target = neutral.copy()
        target[KP.TONGUE_TIP] = [150, 260, 0.95]  # język wystawiony, poniżej wargi
        aus = DeltaActionUnitsExtractor(neutral.flatten()).extract(target.flatten())
        assert aus["AD19"].ratio > 1.15, "AD19 powinno wzrosnąć przy widocznym języku"
