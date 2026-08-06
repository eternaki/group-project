# Projekt: pipeline wielu psów + wiarygodne dane dla sieci AU

**Data:** 2026-08-03
**Status:** zatwierdzony do implementacji
**Kontekst:** Sprint 13–16 (dane → weryfikacja → sieć AU)

---

## 1. Problem (zmierzony, nie domniemany)

Audyt pipeline na 40 wideo (`scripts/debug/audit_pipeline.py`, wynik w `data/audit_pipeline.json`)
oraz analiza zbioru 2826 klatek peak (`data/dataset_final/`) pokazały cztery wady.

### 1.1. Kąt `pitch` mierzy anatomię, nie pozę

`_compute_pitch()` liczy kąt między nosem a linią oczu. U psa nos **zawsze** jest znacznie
poniżej oczu — to budowa pyska, nie pochylenie głowy.

Weryfikacja na **ręcznie anotowanych** landmarkach DogFLW (480 obrazów testowych):

| Kąt | Mediana | Wartości dodatnie | \|kąt\| > 30° |
|-----|---------|-------------------|---------------|
| yaw | −0.2° | 49.4% | 5.8% |
| **pitch** | **+47.5°** | **100%** | **91.5%** |

Yaw zachowuje się jak prawdziwy kąt obrotu (rozkład wokół zera). Pitch to stałe
przesunięcie anatomiczne. Filtr `abs(pitch) > 30 → odrzuć` (`peak_selector.py:308`)
odrzuciłby 91.5% klatek zbioru referencyjnego.

Koszt na naszych danych: **432 klatki** (15.3% wszystkich peaków) odpadły *wyłącznie*
przez pitch — poza tym mają poprawną geometrię, pewność i yaw. Czysty zbiór wynosiłby
**981 zamiast 549**. Dodatkowo powstaje skrzywienie rasowe: mediana |pitch| dla mopsa
14°, dla whippeta 33° — rasy krótkopyskie przechodzą filtr, długopyskie nie.

### 1.2. Szum AU przewyższa próg aktywacji 3–5×

Próg aktywacji AU wynosi `ratio > 1.15`, czyli sygnał 0.15. Zmierzone odchylenie
standardowe ratio **w obrębie jednego wideo** (mediana po 33 wideo):

| AU | σ ratio | Krotność progu |
|----|---------|----------------|
| AU109 | 0.764 | ×5.1 |
| AU116 | 0.750 | ×5.0 |
| AU26 | 0.734 | ×4.9 |
| EAD103 | 0.667 | ×4.4 |
| AU12 | 0.499 | ×3.3 |
| AD19 (najlepszy) | 0.351 | ×2.3 |

AU zapalają się od drgania keypoints, nie od mimiki. Potwierdzenie pośrednie:
17.3% wartości w klamrze, 32.9% poniżej progu wiarygodności.

### 1.3. Na wideo z wieloma psami baza AU może należeć do innego psa

`process_video_for_dataset` bierze `detections[0]` z komentarzem „największy bbox",
ale `bbox.py:235` sortuje detekcje po **confidence**, nie po powierzchni. Gdy w kadrze
jest więcej psów, między klatkami może zostać wybrany inny pies — a delta AU liczy się
względem klatki neutralnej **jednego** psa. Audyt: **12.5% klatek zawiera więcej niż
jednego psa**.

### 1.4. Reguły emocji stoją na najbardziej zaszumionym sygnale

6 z 9 reguł wymaga deskryptorów uszu (EAD101/102/103). EAD103 trafia w klamrę w 21.8%
klatek (470 razy dolna, 148 górna z 2826). Osiągalność reguł na rzeczywistym rozkładzie:
happy 2.2%, angry 3.6%, fearful 1.9%, submission 1.7%, surprise 1.0%, pain 0.5%
(w zbiorze **zero** klatek pain). Stąd 66% neutral i 26% relaxed.

### 1.5. Lej strat (33 wideo, 672 klatki)

| Etap | Strata |
|------|--------|
| Detekcja psa | 26.5% klatek bez psa |
| Keypoints | +12.6% bez punktów; 47.6% z pewnością < 0.5 |
| Poza głowy | **75.8% poza limitem 30°** (głównie przez 1.1) |
| Klatka neutralna | 7 z 40 wideo (17.5%) kończy się `ValueError` |
| Wynik | **3 klatki peak z 33 wideo** |

---

## 2. Cel

Przygotować pipeline do przetwarzania wielu psów na wideo i wytwarzać dane, na których
da się **wytrenować własną sieć** do AU (Sprint 16), a później do emocji.

Kluczowa obserwacja projektowa: **trwała tożsamość psa w czasie rozwiązuje trzy problemy
jednym mechanizmem.** Gdy mamy trek, mamy własny układ odniesienia psa — dla klatki
neutralnej, dla wygładzania szumu i dla anatomii pyska.

### Poza zakresem

- Trenowanie sieci AU (Sprint 16 — wymaga danych zweryfikowanych ręcznie).
- Zmiana modelu keypoints (HRNet-W48 zostaje).
- Import COCO do webapp i eksport CSV (Sprint 15, osobna praca).
- Zmiana progów reguł emocji — reguły pozostają wyłącznie jako **pre-etykieta**.

---

## 3. Decyzje uzgodnione z zespołem

| Pytanie | Decyzja |
|---------|---------|
| Zakres wielu psów | Wszystkie psy, ale z progiem jakości mordy na trek |
| Skład danych | Peaki **plus** klatka neutralna każdego treku |
| Szum AU | Wygładzanie czasowe w obrębie treku |
| Trekowanie | Własny moduł: IoU + histogram koloru |

---

## 4. Weryfikacja podejść w literaturze

Metody sprawdzone przed projektowaniem — szczególnie prace zespołu, który stworzył
DogFLW (nasz zbiór keypoints).

### 4.1. Normalizacja kształtu: Generalised Procrustes Analysis

Martvel i Riemer (Scientific Reports 2025) analizują mimikę psów metodą **geometrycznej
morfometrii**: superpozycja Prokrustesa usuwa przesunięcie, obrót i skalę, następnie PCA
na współrzędnych Prokrustesa. To standard dla porównywania kształtów przy dużej
zmienności morfologicznej — dokładnie nasz przypadek (120 ras).

Nasze obecne dzielenie odległości przez rozstaw oczu jest doraźnym zamiennikiem: usuwa
skalę, ale nie obrót, i sprzęga wszystkie AU z pozą głowy (przy obrocie rozstaw oczu
maleje perspektywicznie, więc **wszystkie** ratio rosną).

### 4.2. Metryki pozy głowy

Ta sama praca mierzy pozę tak:
- **przechylenie** — kąt między linią łączącą wewnętrzne kąciki oczu a osią X;
- **obrót** — różnica odległości od wewnętrznych kącików oczu do środka nosa.

Obie są zerowe dla frontalnej mordy **niezależnie od długości pyska**. Miary „nos poniżej
oczu" w opublikowanym podejściu nie ma — potwierdza to diagnozę z 1.1.

Odsiew: pewność landmarków < 0.6 odrzucona (zostaje 15% klatek), odstające pozy ucinane
**95. percentylem**, między wybranymi klatkami wymagana **≥1 sekunda** odstępu.
Nasza surowość odsiewu (19%) jest zgodna z praktyką — błędne jest kryterium, nie próg.

### 4.3. Stan sztuki w automatycznych AU u psów

Martvel i in. (Nature Sci Reports 2025, PMC12218811): rozpoznawanie DogFACS AU
z landmarków — **średni F1 = 0.292** (precyzja 0.224, czułość 0.512), LSTM-autoenkoder
traktujący AU jako anomalię w szeregu czasowym. Klasyfikacja emocji: 76% trafności,
ale na 29 labradorach (248 wideo).

Wniosek: automatyczne AU u psów to **problem otwarty**. Nasze reguły geometryczne nie
mogą być źródłem prawdy — mogą być wyłącznie pre-etykietą do potwierdzenia przez człowieka.

### 4.4. Wygładzanie czasowe: filtr One Euro

Filtr 1€ (Casiez i in. 2012) jest standardem stabilizacji landmarków twarzy — stosuje go
MediaPipe. Jest **adaptacyjny**: tłumi drgania przy wolnym ruchu i przepuszcza ruch szybki.
To istotne, bo zwykła mediana po oknie zatarłaby sam szczyt mimiki, którego szukamy.

Warunek: przetwarzanie z gęstością ≥5 fps (obecne `batch_annotate --fps 5` spełnia).
Przy 1 fps wygładzanie czasowe nie ma sensu — sąsiednie klatki są niezależne.

### 4.5. Trekowanie zwierząt

Przegląd metod (arXiv 2509.11873): ByteTrack/BoT-SORT **bez re-identyfikacji** spadają
do ID-F1 ≈ 0.15 na długich nagraniach zwierząt; dla zwierząt z nagłą zmianą kierunku
zalecane są cechy wyglądu. Nasz IoU + histogram to uproszczony DeepSORT — zgodne z zaleceniem.
Odrzucone: gotowy tracker ultralytics wymaga ciągłego wideo (na 5 fps predykcja Kalmana
jest bezwartościowa, a YOLO na każdej klatce to 25–30× koszt na CPU).

---

## 5. Architektura

```
klatki (5 fps)
  → detekcja psów (YOLOv8m, wszystkie psy)
  → DogTracker: IoU + histogram → track_id
  → dla KAŻDEGO treku niezależnie:
      → detektor mordy + keypoints (HRNet-W48) na kadrach treku
      → filtr One Euro na keypoints (w układzie mordy)
      → poza głowy (roll z linii oczu, yaw z asymetrii nos↔kąciki oczu)
      → Procrustes: kanoniczny kształt
      → klatka neutralna TEGO treku (mediana okna)
      → delta AU vs własna neutralna
      → TFM → peaki (odstęp ≥1 s)
  → COCO: anotacja na (klatka, trek)
```

### 5.1. Nowy moduł `packages/pipeline/dog_tracker.py`

```python
DogTracker(
    iou_weight=0.6,
    appearance_weight=0.4,
    max_gap_frames=3,
    min_match_score=0.35,
)
tracker.update(frame, detections) -> list[int]   # track_id dla każdej detekcji
```

Koszt dopasowania: `w1·(1−IoU) + w2·(1−podobieństwo histogramu HSV kropu)`.
Przypisanie zachłanne, malejąco po jakości dopasowania. Gdy najlepszy koszt jest gorszy
niż `min_match_score` — **nowy trek**. Trek wygasa po `max_gap_frames` klatkach bez
dopasowania.

Zasada nadrzędna: **lepiej rozerwać trek niż zmieszać dwa psy**. Zmieszanie psów psuje
bazę AU po cichu; rozerwany trek najwyżej skróci serię.

Moduł nie zależy od modeli — testowalny na syntetycznych bboxach i obrazach.

### 5.2. Poza głowy — przepisanie `packages/models/head_pose.py`

```python
roll  = kąt(linia wewnętrznych kącików oczu, oś X)
yaw   = (d(lewy_kącik, nos) − d(prawy_kącik, nos)) / (d(lewy_kącik, nos) + d(prawy_kącik, nos))
```

`yaw` staje się bezwymiarową asymetrią w zakresie [−1, 1] (0 = front), niezależną od
długości pyska. Pole `pitch` **znika** z `HeadPose` — nie da się go wiarygodnie zmierzyć
z 2D bez modelu 3D, a jego obecność wprowadzała w błąd.

Progi odsiewu: percentylowe na poziomie treku (95. percentyl rozkładu w tym treku),
z twardym limitem bezpieczeństwa `|yaw| ≤ 0.35` (asymetria 35% — morda wyraźnie
odwrócona) niezależnie od percentyla. Limit ma sens tylko jako zabezpieczenie przed
trekiem, w którym *wszystkie* klatki są profilowe.

### 5.3. Wygładzanie: `packages/pipeline/landmark_smoothing.py`

Filtr One Euro na każdą współrzędną każdego keypointu, osobny stan na trek.
Wygładzanie działa **we współrzędnych względem boksu mordy** (0–1), nie w pikselach
kadru — inaczej ruch psa po kadrze rozmyłby punkty. Po filtracji współrzędne wracają
do układu obrazu.

### 5.4. Procrustes: `packages/models/shape_normalization.py`

Superpozycja Prokrustesa (usunięcie przesunięcia, skali, obrotu) względem kształtu
referencyjnego wyliczonego raz ze zbioru DogFLW (`data/kp_template.json` już istnieje
jako uśredniony szablon — do przeliczenia metodą GPA).

Wynik trafia do anotacji jako `procrustes_keypoints` — to **wejście dla przyszłej sieci**.
Współrzędne pikselowe zostają dla weryfikacji ręcznej w webapp.

### 5.5. Próg jakości treku

Trek trafia do zbioru, gdy spełnia wszystkie warunki:

| Warunek | Wartość |
|---------|---------|
| Liczba klatek z wykrytą mordą | ≥ 3 |
| Mediana rozmiaru mordy | ≥ 64 px |
| Mediana pewności keypoints | ≥ 0.4 |

Treki odrzucone **są logowane z powodem** (nie znikają po cichu) — inaczej znów nie
będziemy wiedzieć, gdzie tracimy dane.

---

## 6. Schemat COCO — pola dodane

| Pole | Typ | Znaczenie |
|------|-----|-----------|
| `track_id` | int | Tożsamość psa w obrębie wideo |
| `frame_role` | str | `peak` albo `neutral` |
| `neutral_frame_id` | int | `image_id` klatki neutralnej tego treku |
| `label_source` | str | `auto_rules` teraz, `human_verified` po Sprincie 15 |
| `au_noise` | dict | Zmierzone σ ratio na trek, per AU |
| `procrustes_keypoints` | list[float] | Kształt po superpozycji Prokrustesa |

`au_noise` jest kluczowe dla treningu: daje **wagę wiarygodności na próbkę**. Klatka
z rozdygotanego treku nie może ważyć tyle samo, co ze stabilnego.

Pole `au_analysis` pozostaje w formacie `{ratio, is_active, confidence}` (już wdrożone),
odczyt starych zbiorów przez `packages.data.coco.au_ratio()`.

---

## 7. Wpływ na istniejący kod

| Miejsce | Zmiana |
|---------|--------|
| `packages/pipeline/inference.py` | `process_video_for_dataset` zwraca listę treków zamiast jednej listy peaków |
| `scripts/annotation/batch_annotate.py:336` | Anotacja na (klatka, trek), nie jedna na klatkę |
| `apps/webapp/backend/main.py:333` | Sesja dostaje listę psów; front ma już `DogSelector.tsx` |
| `packages/pipeline/peak_selector.py` | Odstęp w sekundach zamiast klatek; filtr pitch usunięty |
| `packages/models/head_pose.py` | Nowe metryki, `pitch` usunięty |
| `packages/pipeline/neutral_frame.py` | `_frontal_factor` liczony bez `pitch`; wybór neutralnej per trek |
| `scripts/annotation/tag_dataset_quality.py` | Tiery liczone na nowych metrykach pozy |

---

## 8. Przypadki brzegowe

| Sytuacja | Zachowanie |
|----------|------------|
| Brak treków spełniających próg | Wideo pominięte z zapisanym powodem (dziś: `ValueError` na 17.5% wideo) |
| Trek bez klatki neutralnej | Trek pominięty — **nigdy** nie bierze neutralnej innego psa |
| Psy przecinają się w kadrze | Trek rozrywa się na dwa; świadomy kompromis (5.1) |
| Jeden pies na wideo | Ścieżka identyczna, jeden trek — bez gałęzi specjalnej |
| Wideo bez wykrytego psa | Pominięte z powodem, bez wyjątku |

---

## 9. Testy (TDD — test przed implementacją)

**Tracker** (syntetyczne bboxy, bez modeli): dwa psy zachowują różne id; pies znika na
1 klatkę i wraca do tego samego treku; pies znika na `max_gap+1` — nowy trek; dwa psy
o podobnym położeniu, różnym kolorze — nie zlewają się.

**Poza głowy** (referencja DogFLW): nowy `yaw` ma medianę bliską zera na 480 obrazach
testowych (obecny `pitch` ma +47.5°); obrót w lewo i w prawo dają przeciwne znaki;
metryka niezależna od skali obrazu.

**Wygładzanie**: sztuczna trajektoria ze znanym szumem — σ po filtracji musi spaść;
skokowa zmiana (peak) musi zostać przepuszczona, nie zatarta.

**Procrustes**: obrót/przesunięcie/przeskalowanie tego samego kształtu daje identyczne
współrzędne wyjściowe.

**Jakość treku**: trek poniżej każdego z trzech progów odrzucony z właściwym powodem.

**Integracja**: dwupsie wideo daje dwie serie anotacji z różnymi `track_id`, każda
z własnym `neutral_frame_id`.

---

## 10. Kryteria akceptacji

1. Wideo z dwoma psami daje anotacje dla obu, każdy z własną klatką neutralną.
2. Nowy `yaw` na referencji DogFLW ma medianę |yaw| odpowiadającą frontalnej pozie
   (rozkład wokół zera, nie stałe przesunięcie).
3. σ ratio AU po wygładzeniu spada mierzalnie względem 0.35–0.76 (ponowny audyt
   tym samym skryptem, te same 40 wideo).
4. Udział klatek przechodzących filtr pozy rośnie względem obecnych 24.2%.
5. Anotacje zawierają `track_id`, `frame_role`, `label_source`, `au_noise`,
   `procrustes_keypoints`.
6. Testy zielone, `ruff check .` czysty.

---

## 11. Źródła

- Martvel G., Riemer S. *Automated analysis of emotional expressions in dogs based on
  geometric morphometrics*. Scientific Reports 2025.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12405473/
- Martvel G. i in. *Dog facial landmarks detection and its applications for facial analysis*.
  Scientific Reports 2025 (AU F1 0.292). https://pmc.ncbi.nlm.nih.gov/articles/PMC12218811/
- DogFLW dataset: https://github.com/martvelge/DogFLW
- *Multi-animal tracking in Transition: Comparative Insights into Established and Emerging
  Methods*. arXiv 2509.11873. https://arxiv.org/html/2509.11873v1
- Casiez G., Roussel N., Vogel D. *1€ Filter: A Simple Speed-based Low-pass Filter for
  Noisy Input in Interactive Systems*. CHI 2012.
  https://github.com/MKSharaf/OneEuroFilterExplained
- Audyt własny: `scripts/debug/audit_pipeline.py`, wyniki `data/audit_pipeline.json`.
