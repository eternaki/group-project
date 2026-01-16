# DogFACS Research - Metodologia i Mapowanie Emocji

**Data:** 2025-01-16
**Autor:** Danylo Lohachov (U1), z pomocą Claude AI
**Sprint:** 1 - Project Setup
**Story:** 1.2 - DogFACS & COCO Research

---

## 1. Wprowadzenie do DogFACS

### 1.1 Czym jest DogFACS?

**DogFACS (Dog Facial Action Coding System)** to naukowe narzędzie obserwacyjne do identyfikacji i kodowania ruchów twarzy psów. System został zaadaptowany z oryginalnego FACS dla ludzi (Ekman & Friesen, 1978) i dostosowany do anatomii twarzy psów.

**Kluczowe cechy:**
- Obiektywny system kodowania bez założeń o emocjach
- Oparty na anatomii mięśni twarzy psa
- Wymaga certyfikacji do użycia
- Nie jest etogramem wyrażeń twarzy

### 1.2 Twórcy DogFACS

DogFACS został opracowany przez:
- **Bridget M. Waller** (Nottingham Trent University)
- **Juliane Kaminski** (University of Portsmouth)
- **Anne M. Burrows** (Duquesne University)
- **Cátia Caeiro** (University of Lincoln)
- **Kate Peirce** (University of Portsmouth)

Rozwój wspierany przez WALTHAM® Foundation Research Grant.

**Oficjalna strona:** [animalfacs.com](https://animalfacs.com/dogfacs_new)

---

## 2. Action Units (AU) - Jednostki Akcji

### 2.1 Struktura DogFACS

DogFACS identyfikuje **16 niezależnych ruchów twarzy i uszu**:
- **11 Action Units (AU)** - ruchy mięśni twarzy
- **5 Ear Action Descriptors (EAD)** - ruchy uszu

### 2.2 Zidentyfikowane Action Units

| Kod | Nazwa | Opis | Mięsień |
|-----|-------|------|---------|
| **AU101** | Inner Brow Raiser | Podniesienie wewnętrznych brwi ("puppy dog eyes") | Levator anguli oculi medialis |
| **AU145** | Blink | Mruganie | Orbicularis oculi |
| **AU25** | Lips Part | Rozchylenie warg | Orbicularis oris |
| **AU26** | Jaw Drop | Opuszczenie szczęki | Digastric, Masseter |
| **AD137** | Nose Lick | Oblizanie nosa | Język |

### 2.3 Ear Action Descriptors (EAD)

| Kod | Nazwa | Opis |
|-----|-------|------|
| **EAD102** | Ears Adductor | Przyciąganie uszu do środka |
| **EAD103** | Ears Flattener | Spłaszczenie uszu |
| **EAD104** | Ears Rotator | Obracanie uszu |

> **Uwaga:** Pełna lista AU dostępna jest w podręczniku DogFACS wymagającym rejestracji na animalfacs.com

---

## 3. Mapowanie Emocji

### 3.1 Podstawowe Emocje Psów

Na podstawie badań naukowych, definiujemy **6 kategorii emocji** dla naszego projektu:

| ID | Emocja | Angielski | Opis |
|----|--------|-----------|------|
| 0 | Szczęśliwy | happy | Pozytywny stan, radość |
| 1 | Smutny | sad | Negatywny stan, przygnębienie |
| 2 | Zły | angry | Agresja, złość |
| 3 | Przestraszony | fearful | Strach, lęk |
| 4 | Zrelaksowany | relaxed | Spokój, odpoczynek |
| 5 | Neutralny | neutral | Brak wyraźnej emocji |

### 3.2 Mapowanie DogFACS → Emocje

Na podstawie badań (Waller et al., 2013; Kaminski et al., 2017; Bremhorst et al., 2022):

#### Happy (Szczęśliwy)
- AU25 (rozchylone wargi)
- EAD102 (uszy skierowane do przodu)
- Zrelaksowana pozycja twarzy
- Możliwe "dog smile" - otwarte usta

#### Sad (Smutny)
- EAD103 (spłaszczone uszy)
- Opuszczone brwi
- Opuszczony wzrok
- Zredukowana aktywność twarzy

#### Angry (Zły)
- AU25 + AU26 (odsłonięte zęby)
- Zmarszczone czoło
- Intensywne spojrzenie
- EAD - uszy do przodu lub do tyłu

#### Fearful (Przestraszony)
- EAD103 (spłaszczone uszy do tyłu)
- AU145 (częste mruganie)
- Odwrócony wzrok
- AD137 (oblizywanie nosa - sygnał stresu)

#### Relaxed (Zrelaksowany)
- Miękkie, lekko przymknięte oczy
- Neutralna pozycja uszu
- Zamknięte lub lekko otwarte usta
- Brak napięcia mięśni twarzy

#### Neutral (Neutralny)
- Brak wyraźnych sygnałów emocjonalnych
- Standardowa pozycja wszystkich elementów twarzy
- Baseline dla porównań

### 3.3 Ograniczenia

Badania wskazują na trudności w rozróżnieniu niektórych emocji:
- Strach vs frustracja mogą wyglądać podobnie
- Uszy spłaszczone do tyłu + otwarte usta występują w obu stanach
- Kontekst behawioralny jest ważny dla interpretacji

---

## 4. Keypoints Schema - 20 Punktów Kluczowych

### 4.1 Definicja Keypoints

Na podstawie DogFLW (46 punktów) i wymagań projektu, definiujemy **20 kluczowych punktów** twarzy psa:

| ID | Nazwa | Opis anatomiczny | Visibility |
|----|-------|------------------|------------|
| 0 | left_eye_inner | Wewnętrzny kącik lewego oka | 2 |
| 1 | left_eye_outer | Zewnętrzny kącik lewego oka | 2 |
| 2 | right_eye_inner | Wewnętrzny kącik prawego oka | 2 |
| 3 | right_eye_outer | Zewnętrzny kącik prawego oka | 2 |
| 4 | nose_tip | Czubek nosa | 2 |
| 5 | nose_left | Lewy skrzydełko nosa | 2 |
| 6 | nose_right | Prawy skrzydełko nosa | 2 |
| 7 | left_ear_base | Podstawa lewego ucha | 1-2 |
| 8 | left_ear_tip | Czubek lewego ucha | 1-2 |
| 9 | right_ear_base | Podstawa prawego ucha | 1-2 |
| 10 | right_ear_tip | Czubek prawego ucha | 1-2 |
| 11 | mouth_left | Lewy kącik pyska | 2 |
| 12 | mouth_right | Prawy kącik pyska | 2 |
| 13 | upper_lip | Górna warga (środek) | 2 |
| 14 | lower_lip | Dolna warga (środek) | 2 |
| 15 | chin | Podbródek | 2 |
| 16 | left_brow | Lewa brew | 2 |
| 17 | right_brow | Prawa brew | 2 |
| 18 | forehead | Czoło (środek) | 2 |
| 19 | muzzle_center | Środek pyska | 2 |

### 4.2 Visibility Flags (COCO standard)

- **0** = nie oznaczony (not labeled)
- **1** = oznaczony ale niewidoczny (labeled but not visible)
- **2** = oznaczony i widoczny (labeled and visible)

### 4.3 Skeleton Connections

Definiujemy połączenia między punktami do wizualizacji:

```python
SKELETON = [
    # Oczy
    [0, 1],   # left_eye
    [2, 3],   # right_eye
    [0, 2],   # between_eyes

    # Nos
    [4, 5],   # nose_left
    [4, 6],   # nose_right

    # Uszy
    [7, 8],   # left_ear
    [9, 10],  # right_ear

    # Usta
    [11, 13], # left_mouth_to_upper
    [12, 13], # right_mouth_to_upper
    [11, 14], # left_mouth_to_lower
    [12, 14], # right_mouth_to_lower
    [14, 15], # lower_lip_to_chin

    # Brwi
    [16, 18], # left_brow_to_forehead
    [17, 18], # right_brow_to_forehead

    # Pysk
    [4, 19],  # nose_to_muzzle
    [19, 13], # muzzle_to_upper_lip
]
```

---

## 5. Wnioski i Rekomendacje

### 5.1 Dla projektu Dog FACS Dataset

1. **Używamy 6 kategorii emocji** - happy, sad, angry, fearful, relaxed, neutral
2. **20 keypoints** wystarcza do podstawowej analizy wyrażeń twarzy
3. **Automatyczna detekcja** keypoints umożliwi klasyfikację emocji
4. **Kontekst wideo** może poprawić dokładność klasyfikacji

### 5.2 Dokładność oczekiwana

Na podstawie literatury:
- Rozpoznawanie happy/angry: **>80%** dokładności
- Rozpoznawanie fearful: **<70%** dokładności (trudniejsze)
- Overall accuracy target: **>70%**

### 5.3 Następne kroki

1. ✅ Research DogFACS zakończony
2. → Zdefiniowanie schematu COCO (Story 1.2 kontynuacja)
3. → Analiza datasetów (Story 1.3)
4. → Research architektury modeli (Story 1.4)

---

## 6. Źródła

1. [AnimalFACS - DogFACS](https://animalfacs.com/dogfacs_new) - Oficjalna strona DogFACS
2. [Dog facial landmarks detection - Nature Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-07040-3) - DogFLW Dataset
3. [Current Advances in Assessment of Dog's Emotions - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8614696/) - Przegląd metod
4. [Explainable automated recognition of emotional states - Nature (2022)](https://www.nature.com/articles/s41598-022-27079-w) - Automatyczne rozpoznawanie
5. [Dogs and humans respond to emotionally competent stimuli - Nature (2017)](https://www.nature.com/articles/s41598-017-15091-4) - Porównanie reakcji
6. [DogFACS: Interpreting Your Dog's Facial Expressions](https://dogshowconfidential.com/dogfacs-interpreting-your-dogs-facial-expressions/) - Wprowadzenie

---

*Dokument wygenerowany w ramach projektu Dog FACS Dataset dla Politechniki Gdańskiej (WETI).*
