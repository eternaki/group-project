# Instrukcja Weryfikacji Anotacji - Dog FACS Dataset

## Wprowadzenie

Ten dokument zawiera wytyczne dla annotatorów weryfikujących automatyczne anotacje emocji psów. Celem weryfikacji jest zapewnienie wysokiej jakości datasetu poprzez manualną kontrolę i korektę anotacji AI.

---

## Cele Weryfikacji

1. **Weryfikacja poprawności emocji** - Sprawdzenie czy etykieta emocji odpowiada rzeczywistemu stanowi psa
2. **Weryfikacja bounding boxów** - Sprawdzenie czy ramka ograniczająca prawidłowo obejmuje psa
3. **Weryfikacja keypoints** - Sprawdzenie czy punkty kluczowe twarzy są poprawnie umiejscowione
4. **Identyfikacja błędów** - Oznaczanie problemów do późniejszej analizy

---

## Kategorie Emocji

### 1. Happy (Szczęśliwy)
**Cechy charakterystyczne:**
- Machanie ogonem
- "Uśmiech" - otwarta buzia z widocznym językiem
- Rozluźnione uszy (często do tyłu)
- Podskakiwanie, skakanie z radości
- Żywy wyraz oczu

**Kontekst:** Zabawa, powitanie właściciela, interakcja z innymi psami

### 2. Sad (Smutny)
**Cechy charakterystyczne:**
- Opuszczone uszy
- Przygnębiony wyraz oczu ("smutne oczy")
- Ogon między nogami lub nisko
- Powolne ruchy
- Brak zainteresowania otoczeniem

**Kontekst:** Rozłąka z właścicielem, samotność, choroba

### 3. Angry (Zły/Agresywny)
**Cechy charakterystyczne:**
- Warczenie, pokazywanie zębów
- Napięte ciało
- Uszy do przodu lub do tyłu (spłaszczone)
- Uniesiona sierść na grzbiecie
- Intensywne spojrzenie

**Kontekst:** Ochrona terytorium, konfrontacja, zagrożenie

### 4. Relaxed (Zrelaksowany)
**Cechy charakterystyczne:**
- Rozluźnione ciało
- Spokojny wyraz pyska
- Uszy w neutralnej pozycji
- Powolne, miarowe oddychanie
- Może być w pozycji leżącej

**Kontekst:** Odpoczynek, sen, spokojne otoczenie

### 5. Fearful (Przestraszony)
**Cechy charakterystyczne:**
- Ogon między nogami
- Przygarbiona postawa
- Uszy spłaszczone do tyłu
- Unikanie kontaktu wzrokowego
- Drżenie, chowanie się

**Kontekst:** Burza, głośne dźwięki, nieznane środowisko

### 6. Neutral (Neutralny)
**Cechy charakterystyczne:**
- Brak wyraźnych cech emocjonalnych
- Uszy w neutralnej pozycji
- Ogon w naturalnej pozycji
- Spokojny, ale czujny

**Kontekst:** Codzienna aktywność, chodzenie, obserwacja

---

## Procedura Weryfikacji

### Krok 1: Przegląd obrazu
1. Otwórz obraz w narzędziu weryfikacji
2. Oceń ogólną jakość obrazu (czy pies jest wyraźnie widoczny)
3. Zidentyfikuj liczbę psów na obrazie

### Krok 2: Weryfikacja Bounding Box
1. Sprawdź czy ramka obejmuje całego psa
2. Ramka nie powinna być zbyt duża (dużo tła)
3. Ramka nie powinna obcinać części psa

**Akceptuj jeśli:** Ramka prawidłowo obejmuje psa z małym marginesem
**Odrzuć jeśli:** Ramka znacząco obcina psa lub obejmuje inne obiekty

### Krok 3: Weryfikacja Emocji
1. Obserwuj mimikę pyska, pozycję uszu, ogona
2. Weź pod uwagę kontekst (jeśli widoczny)
3. Porównaj z cechami charakterystycznymi dla każdej emocji

**Akceptuj jeśli:** Etykieta emocji odpowiada obserwowanym cechom
**Popraw jeśli:** Emocja jest błędna - wybierz prawidłową
**Odrzuć jeśli:** Nie można jednoznacznie określić emocji

### Krok 4: Weryfikacja Keypoints
1. Sprawdź pozycje punktów na pysku (nos, oczy, uszy)
2. Punkty powinny być widoczne i prawidłowo umiejscowione

### Krok 5: Dokumentacja
1. Dla każdej anotacji wybierz: Akceptuj / Popraw / Odrzuć
2. W przypadku poprawy - wybierz prawidłową emocję
3. Dodaj notatkę jeśli potrzebna (np. "Słaba jakość obrazu")

---

## Przypadki Problematyczne

### Obrazy niskiej jakości
- Rozmazane obrazy
- Zbyt ciemne/jasne
- Pies częściowo zasłonięty

**Zalecenie:** Odrzuć jeśli nie można jednoznacznie ocenić emocji

### Wiele psów
- Każdy pies powinien mieć osobną anotację
- Sprawdź czy bounding boxy się nie nakładają

### Mieszane emocje
- Niektóre psy mogą wykazywać cechy wielu emocji
- Wybierz dominującą emocję
- Dodaj notatkę o mieszanych cechach

### Rasy z trudną mimiką
- Niektóre rasy (np. Bulldog, Shar Pei) mają specyficzną budowę pyska
- Zwróć szczególną uwagę na inne cechy (uszy, ogon, postawa)

---

## Wskazówki Praktyczne

1. **Rób przerwy** - Zmęczenie wpływa na jakość oceny
2. **Bądź konsekwentny** - Stosuj te same kryteria dla wszystkich obrazów
3. **W razie wątpliwości** - Dodaj notatkę i przejdź dalej
4. **Zapisuj regularnie** - Eksportuj korekty co 50-100 obrazów
5. **Komunikuj problemy** - Zgłaszaj powtarzające się problemy

---

## Metryki Jakości

### Oczekiwana zgodność
- Cel: > 80% zgodności między annotatorami
- Cohen's Kappa > 0.6 (dobra zgodność)

### Czas weryfikacji
- Średnio 30-60 sekund na obraz
- Szybsze dla jednoznacznych przypadków

---

## Kontakt

W przypadku pytań lub problemów technicznych:
- Sprawdź dokumentację w `docs/`
- Zgłoś problem przez system śledzenia błędów

---

*Ostatnia aktualizacja: Styczeń 2026*
