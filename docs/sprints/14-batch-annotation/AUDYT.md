# Audyt pipeline'u — przed i po przejściu na treki

Dokument porównuje pipeline sprzed gałęzi `feature/pipeline-audit` z wersją po niej.
Oba przebiegi: `scripts/debug/audit_pipeline.py --limit 40 --fps 1.0 --max-frames 20
--num-peaks 3 --device cpu`, ten sam katalog wideo (`data/drive_dogs/DOGS`), ta sama
kolejność plików. Surowe wyniki zacommitowane w `docs/sprints/14-batch-annotation/dane/`
(`audit_before.json`, `audit_after.json`), żeby dało się je sprawdzić po sklonowaniu repo.

**Czego NIE da się odtworzyć.** Materiał źródłowy (1505 plików wideo) jest w `.gitignore`
— za duży na repozytorium. Przebiegu „przed" nie odtworzy też bieżąca wersja skryptu:
powstała w tej samej gałęzi i produkuje inne klucze wyjściowe (`etap_0_treki`,
`yaw_asymmetry_abs`) niż wersja, która wygenerowała plik „przed" (`pitch_abs`, bez etapu 0).
Plik „przed" jest więc zapisem historycznym, nie wynikiem powtarzalnego polecenia.

**Jak czytać ten dokument.** Część liczb to porównanie tej samej wielkości przed i po —
te wolno czytać jako poprawę albo pogorszenie. Część to **zamiana miary**: stara metryka
została uznana za błędną i zastąpiona inną, więc liczby nie są porównywalne i zaznaczono
to wprost. Tam, gdzie wielkość ma rozrzut, podany jest rozrzut, a nie jedna cyfra.

---

## 1. Przetwarzanie wideo

| Wielkość | Przed | Po |
|---|---|---|
| Wideo przetworzone | 33 / 40 | **40 / 40** |
| Wideo nieudane (wyjątek) | 7 | **0** |
| Czas na wideo | 40,1 s | 24,7 s |

Siedem wideo kończyło się wyjątkiem, który przerywał całe nagranie. Po zmianie błąd
jednego psa nie kosztuje pozostałych: awaria treku zamienia się w odrzucenie z powodem
(`_TRACK_FAILURES` w `packages/pipeline/inference.py`), a wideo bez godnych treków
zwraca pusty wynik zamiast rzucać.

Skrócenie czasu bierze się stąd, że detektor psa biegnie raz zamiast dwa razy oraz że
keypoints liczone są tylko na klatkach należących do treku.

---

## 2. Lejek etapami

### Etap 0 — treki (nowy)

Etapu nie było: pipeline brał jedną detekcję o najwyższej pewności na klatkę.

| Wielkość | Po |
|---|---|
| Treki przyjęte | 26 |
| Treki odrzucone | 32 |
| — powód: za mało klatek z mordą | 22 |
| — powód: za mała morda | 10 |
| Klatki odsiane progiem pewności keypoints | 320 |
| Treki krótkie **dopiero** przez ten filtr | 11 |

Ostatni wiersz to koszt filtru pewności dodanego w tej gałęzi: 11 z 22 treków odrzuconych
za długość przeszłoby próg, gdyby liczyć także klatki o niepewnych keypoints. Zmiana jest
zamierzona — takie klatki i tak nie trafiłyby do zbioru jako peaki, a wcześniej
współtworzyły medianową bazę AU treku i zawyżały zmierzony szum.

### Etap 1 — detekcja psa

| Wielkość | Przed | Po |
|---|---|---|
| Klatki | 672 | 672 |
| Bez psa | 26,5 % | 26,5 % |
| Więcej niż jeden pies | 12,5 % | 12,5 % |
| Pewność detekcji (mediana) | 0,904 | 0,904 |

Bez zmian — detektor psa nie był ruszany. Wiersz „więcej niż jeden pies" jest tu
uzasadnieniem całej gałęzi: **12,5 % klatek zawiera więcej niż jednego psa**, a przed
zmianą wszystkie one dostawały bazę AU wziętą z detekcji o najwyższej pewności, czyli
potencjalnie od innego psa niż klatka docelowa.

### Etap 2 — keypoints

| Wielkość | Przed | Po |
|---|---|---|
| Detekcje bez keypoints | 12,6 % | 0,0 % |
| Pewność keypoints (mediana) | 0,541 | 0,818 |
| Pewność poniżej 0,5 | 47,6 % | 0,0 % |
| Klatki wchodzące do AU | 454 | 224 |

**To nie jest poprawa modelu keypoints — model jest ten sam.** Zmieniło się to, że klatki
o niepewnym pomiarze są teraz odsiewane, zanim wejdą do treku. Dlatego mediana pewności
rośnie, a liczba klatek spada z 454 do 224. Właściwy odczyt: pipeline przestał liczyć AU
na pomiarach, którym sam nie ufa.

### Etap 3 — poza głowy (ZAMIANA MIARY, nie porównanie)

| Miara | Przed | Po |
|---|---|---|
| Metryka obrotu | `yaw` w stopniach | `yaw_asymmetry` (bezwymiarowa) |
| Metryka pochylenia | `pitch` w stopniach | — (usunięta) |
| Metryka przechylenia | — | `roll` w stopniach |
| Mediana metryki głównej | pitch 43,0° | yaw_asymmetry 0,081 |
| Poza poza limitem | 75,8 % | 9,4 % |

Liczb w ostatnim wierszu **nie wolno czytać jako poprawy** — mierzą co innego.

Stara miara `pitch` (kąt nos–linia oczu) mierzyła anatomię pyska, nie pozę głowy: na 480
ręcznie anotowanych obrazach DogFLW jej mediana wynosiła **+47,5°**, wszystkie wartości
były dodatnie, a **91,5 % przekraczało próg 30°**, mimo że mordy są frontalne. Miara
odrzucała długopyskie rasy za sam kształt czaszki (mediana |pitch| 14° u mopsa wobec 33°
u whippeta). Kosztowało to 432 klatki (15,3 % zbioru) odrzucone **wyłącznie** przez pitch.

Zastąpiona asymetrią odległości kącik oka ↔ nos, która jest zerowa dla mordy frontalnej
niezależnie od długości pyska, oraz przechyleniem z linii oczu. Miary wzorowane na pracy
Martvela i Riemer, *Automated analysis of emotional expressions in dogs based on geometric
morphometrics* (Sci Rep 2025), która stosuje je do landmarków DogFLW.

### Etap 4 — klatka neutralna

| Wielkość | Przed | Po |
|---|---|---|
| Klatek neutralnych | 33 (jedna na wideo) | 26 (jedna na **trek**) |
| Odchylenie od frontalności (mediana) | 55,86 (suma kątów) | 0,34 (suma znormalizowana) |

Jednostki inne (patrz etap 3), więc porównywalna jest wyłącznie zmiana jakościowa: klatka
neutralna należy teraz do konkretnego psa. Wcześniej przy dwóch psach obaj dostawali tę
samą bazę.

### Etap 5 — Action Units

| Wielkość | Przed | Po |
|---|---|---|
| Klatki z policzonymi AU | 454 | 224 |
| Wartości klamrowane (średnio) | 17,3 % | **4,6 %** |
| Najgorsze klamrowanie | AU25 34,6 % | EAD103 19,6 % |

Klamrowanie to pomiar dobity do granicy zakresu `[0.2, 3.0]` — wartość, która wygląda jak
silna aktywacja, choć nic nie mierzy. Spadek z 17,3 % do 4,6 % jest efektem wygładzania
punktów i odsiewu niepewnych klatek.

---

## 3. Szum AU — najważniejsza liczba audytu

Próg aktywacji AU to `ratio > 1.15`, czyli **sygnał 0,15**. Zmierzony szum to odchylenie
standardowe ratio w obrębie treku, osobno dla każdego AU (546 par trek–AU).

| Wielkość | Przed | Po |
|---|---|---|
| Podłoga szumu, najgorsze AU | AU109 0,764 · AU116 0,750 · AU26 0,734 | AU116 0,412 · AU25 0,404 · AD35 0,404 |
| Podłoga szumu, najlepsze AU | — | AU145 0,128 · AU101 0,132 · EAD102 0,145 |
| Mediana po parach trek–AU | — | **0,232** |
| Rozrzut (p10–p90) | — | 0,055 – 0,801 |
| Liczba prób na pomiar (mediana) | — | 8 (p10 = 4, p90 = 16) |

**Szum spadł o połowę i nadal przewyższa sygnał.**

- **68,9 %** wszystkich par trek–AU ma szum **większy niż sygnał aktywacji 0,15**.
- **17 z 21 AU** ma medianę szumu powyżej sygnału.

Wniosek jest ten sam, co przed zmianą, tylko wsparty lepszymi danymi: **reguły AU nadają
się wyłącznie na pre-etykiety, nie na cel treningowy.** Dla większości jednostek flaga
`is_active` zapala się częściej od drgania punktów niż od mimiki psa.

Dźwignia nie leży w filtrowaniu — leży w jakości keypoints. Pomiar pomocniczy: przy
błędzie naszego modelu (NME_iod 0,091) mediana szumu na danych syntetycznych wynosi
**0,17 ± 0,03** (zakres 0,121–0,211 po 20 realizacjach szumu). Realny materiał daje 0,232,
czyli nieco gorzej niż synteza — a to znaczy, że synteza nie była pesymistyczna.

Ponieważ NME jest normalizowane rozstawem oczu, stosunek błędu do sygnału **nie zależy od
rozdzielczości kadru**. Dla AU o krótkiej bazie pomiarowej błąd lokalizacji przewyższa
sygnał wielokrotnie: AU143 (otwarcie oka) 7,1×, AU25/AD35 (szczelina warg) 5,4×, AU116
3,1×, AU101 2,4×, AU26 1,9×, AU118 1,4×. Podnoszenie progu rozmiaru mordy tego nie
naprawi.

**Co z tego wynika dla zbioru.** Każda anotacja niesie teraz `au_noise` (zmierzony szum
tego AU w tym treku) i `au_sample_count` (z ilu klatek policzony), a `au_analysis` dokłada
per AU pola `noise` i `snr` (`|ratio − 1| / noise`). Trening w Sprincie 16 ma czym ważyć
próbki i czym odsiać aktywacje pochodzące z szumu. Brak zmierzonego szumu jest
rozróżnialny od szumu zerowego (`None` zamiast `0.0`) — to nie to samo i nie wolno tego
mylić.

---

## 4. Peaki i emocje

| Wielkość | Przed | Po |
|---|---|---|
| Peaki łącznie | **3** | **39** (dziś 38 — patrz niżej) |
| TFM peaka (mediana) | — | 2,78 |
| TFM peaka (p10) | — | 0,723 |
| Rozkład emocji | neutral 2 · relaxed 1 | neutral 26 · relaxed 6 · sad 2 · happy 2 · angry 2 · submission 1 |

Trzy peaki z 33 wideo oznaczały, że pipeline praktycznie nie produkował materiału.
Trzynastokrotny wzrost bierze się z trzech rzeczy naraz: wideo przestały się wywalać
(7 → 0), peaki liczone są **na psa**, a nie na wideo, i zniknął filtr pitch, który odrzucał
frontalne mordy długopyskich ras.

### Fallback TFM — decyzja na liczbach, już wykonana

> **Uwaga o dacie pomiaru.** Cały audyt „po" biegł na commicie `ba887a5`, czyli **przed**
> `e497bff`, który usunął opisany niżej mechanizm. Wszystkie pozostałe liczby w tym
> dokumencie zmiana nie dotyczy, ale **liczba peaków jest dziś 38, nie 39.**

`PeakFrameSelector` miał zastany mechanizm: gdy silnych kandydatów jest mniej niż
`num_peaks`, dobierał kadry **poniżej progu TFM**, czyli klatki bez mimiki. Audyt zmierzył
oba warianty na tym samym materiale:

| Wariant | Peaki |
|---|---|
| Z fallbackiem (stan w chwili pomiaru, commit `ba887a5`) | 39 |
| Bez fallbacku (**stan obecny**, od commita `e497bff`) | **38** |
| Peaki pochodzące spod progu | 1 (2,6 %) |

**Decyzja: fallback usunięty** (`e497bff`). Kosztował jeden peak na 39, a wprowadzał do
zbioru kadr, o którym z definicji wiadomo, że nie ma mimiki wartej anotacji. Analogiczny
mechanizm przy separacji peaków został usunięty wcześniej w tej samej gałęzi — tam
produkował sąsiadujące klatki, czyli duplikaty.

Usunięcie ujawniło lukę w testach: atrapa psa w testach integracyjnych nie ruszała mordą
wcale, więc pięć testów peaków opierało się właśnie na dobieraniu kadrów spod progu.
Naprawione przez danie atrapie realnej mimiki, nie przez obniżenie progu.

Skutek uboczny dla samego narzędzia: sekcja `audyt_fallbacku_tfm` w
`scripts/debug/audit_pipeline.py` mierzy od tej zmiany wielkość, która z definicji wynosi
zero. Zostawiona świadomie jako strażnik — gdyby dobieranie spod progu kiedyś wróciło
(np. przez „poprawkę" zwiększającą yield), audyt pokaże to natychmiast.

---

## 5. Co ten audyt zmierzył, a czego nie

Zmierzone: lejek etapami, szum AU na realnym materiale, koszt filtru pewności, udział
fallbacku TFM.

**Niezmierzone i wciąż otwarte:**

- **Trafność AU i emocji.** Brak danych z ręczną weryfikacją (GT), więc żadna liczba w tym
  dokumencie nie mówi, czy AU są *poprawne* — mówią wyłącznie, czy są *stabilne*. Niski
  szum nie dowodzi trafności.
- **Próg `snr ≥ 1`** przyjęty przy zapisie do COCO jest konwencją, nie pomiarem. Przy
  medianie szumu 0,232 oznacza realny próg wykrywalności rzędu 0,23 zamiast 0,15, czyli
  większość dzisiejszych `is_active` odpadnie. To wygląda na prawidłową diagnozę, ale
  wymaga potwierdzenia przed treningiem.
- **`MIN_TRACK_FRAMES = 3`.** Mediana liczby prób na pomiar szumu wynosi 8, ale p10 to 4 —
  a sigma z 3–4 próbek ma ~11 % obciążenia i ~50 % rozrzutu własnego. Podniesienie progu do
  5 zmniejszyłoby zbiór; decyzja wymaga zestawienia z docelową liczebnością.
- **Rasa** klasyfikowana jest z boksu psa (przywrócone w tej gałęzi po tym, jak przez jeden
  commit liczyła się z kadru mordy), ale jej trafności ten audyt nie sprawdza.
