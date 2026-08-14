# Instrukcja weryfikacji AU

Dokument dla osób anotujących. Czyta się w pięć minut, potem można zaczynać.

## Po co to robimy

Etykiety, które pipeline nadaje automatycznie, **nie nadają się do zbioru**.
Audyt na 2786 parach pokazał, że reguły bywają odwrotnością tego, co widać:
dyszący pies z otwartym pyskiem i wywieszonym językiem dostawał `neutral`,
a spokojnie siedzący spaniel z zamkniętym pyskiem — `surprise`. Powód jest
mechaniczny: każdy pomiar AU dzieli się przez rozstaw oczu, a ten skraca się
przy obrocie głowy, więc **sam obrót o 30° podnosi wszystkie AU powyżej progu
aktywacji**. U 30.5% par to wystarczało do „aktywacji" bez żadnego ruchu mięśni.

Dlatego werdykt człowieka jest jedynym źródłem prawdy w tym zbiorze.
Automatyczne etykiety zostają obok, jako materiał porównawczy.

## Uruchomienie

```bash
python scripts/run_annotation.py
```

Skrypt sam przygotuje zbiór, podniesie API i frontend. Potem otwórz
http://localhost:5173 i wybierz zakładkę **Weryfikacja AU**. `Ctrl+C` kończy.

Wymagania: `.venv` z `pip install -e .`, `npm` w PATH.

## Jak wygląda praca

Na ekranie są **dwa kadry tego samego psa**:

| lewy — **neutralny** | prawy — **szczytowy** |
|---|---|
| pysk w spoczynku, baza pomiaru | kadr, który oceniasz |

AU to z definicji **różnica względem kadru neutralnego**. Nie oceniaj prawego
kadru samego w sobie — pytaj, *co się zmieniło* względem lewego.

## Klawiatura

| klawisz | działanie |
|---|---|
| `1`–`8` | oznacz AU jako **aktywne** (drugie naciśnięcie cofa) |
| `Shift`+`1`–`8` | oznacz AU jako **niewidoczne** |
| `Enter` | zapisz parę i przejdź dalej |
| `←` `→` | nawigacja bez zapisu |
| `Spacja` | pomiń parę |

**Zaznaczasz tylko to, co widzisz.** Wszystko, czego nie zaznaczysz, zapisze się
jako *nieaktywne* dopiero w chwili naciśnięcia `Enter` — to naciśnięcie znaczy
„obejrzałem wszystkie osiem". Przy typowych 2–3 aktywnych AU wychodzą około
cztery klawisze na parę.

## Osiem AU do weryfikacji

| klawisz | kod | czego szukać |
|---|---|---|
| `1` | AU25 | wargi rozchylone, widać zęby lub szparę |
| `2` | AU26 | żuchwa opuszczona, pysk otwarty |
| `3` | AU27 | pysk otwarty szeroko (ziewanie, dyszenie) |
| `4` | AD19 | język widoczny poza wargami |
| `5` | EAD103 | uszy przyciśnięte do głowy |
| `6` | AU143 | powieka napięta, oko zmrużone |
| `7` | AU101 | brew uniesiona, fałda nad okiem |
| `8` | AU12 | kąciki ust cofnięte do tyłu |

### Dlaczego osiem, a nie 21

Pipeline nadal liczy wszystkie 21 AU DogFACS — spec się nie zmienia. Ale
etykietę człowieka zbieramy tylko dla tych, które **da się orzec na stop-klatce**.

Pozostałe dzielą się na dwie grupy. Część wymaga zbliżenia, jakiego na materiale
stockowym nie ma (marszczenie nosa, ruchy pojedynczej wargi). Część to
**ruchy, a nie stany**: oblizywanie nosa, oblizywanie wargi, mruganie — na
jednej klatce nie da się ich odróżnić od zwykłego zamkniętego oka czy
wysuniętego języka. Wpisanie tam zgadywanki byłoby gorsze niż brak etykiety.

## `niewidoczne` to nie `nieaktywne`

Najważniejsza zasada całej pracy.

- Ucho **przyciśnięte do głowy** → `EAD103` aktywne (`5`).
- Ucho **schowane za głową, nie widać go** → `EAD103` niewidoczne (`Shift+5`).

To nie jest ta sama informacja. Pierwsze znaczy „mięsień zadziałał", drugie
„nie wiem". Wpisanie zera tam, gdzie nie widać, nauczyłoby sieć, że niewidoczne
znaczy spoczynkowe — czyli nauczyłoby ją nieprawdy.

Jeśli w kadrze nie widać połowy mordy, spokojnie oznacz `Shift` na wszystkim,
czego nie widzisz. Lepiej mieć parę z trzema pewnymi etykietami niż osiem
zgadywanych.

## Czego NIE robisz

- **Nie poprawiasz emocji.** Emocja jest warstwą wtórną, liczoną z AU.
- **Nie poprawiasz rasy.** Nie wpływa na AU, a klasyfikator na tym materiale
  ma medianę pewności 0.33 — poprawianie go zajęłoby więcej czasu niż daje.
- **Nie poprawiasz keypoints** w tym trybie. Jeśli punkty leżą ewidentnie nie na
  mordzie, pomiń parę spacją.

## Podpowiedź „reguła: tak"

Przy niektórych AU zobaczysz szary napis `reguła: tak`. To znaczy tylko tyle, że
automat uznał to AU za aktywne. **Nie jest to sugestia, co zaznaczyć.** Nie
patrz na nią, dopóki nie podejmiesz własnej decyzji — po to nic nie jest
zaznaczone z góry. Rozbieżność między tobą a regułą jest cenną informacją
i trafia do raportu jako miara jakości automatu.

## Zgodność między anotatorami

Pierwsze **150 par robią dwie osoby niezależnie**. Z tego liczymy współczynnik
zgodności (Cohena κ). To dwie godziny pracy, które odróżniają zbiór danych od
zestawu obrazków — i najmocniejszy argument, jaki będziemy mieli przed komisją.

Nie konsultujcie się przy tych 150 parach. Zgodność wymuszona rozmową nic nie
mierzy.

## Kolejność par nie jest przypadkowa

Pary idą **na przemian z różnych nagrań**, a w obrębie nagrania najpierw te,
w których pomiar leży blisko progu decyzyjnego. Dzięki temu praca przerwana
w połowie daje przekrój materiału, a nie pięćdziesiąt klatek jednego psa.

Nie przeskakuj do przodu w poszukiwaniu „ładniejszych" kadrów — popsuje to
własność, dla której ta kolejność powstała.

## Eksport

Przycisk **Eksportuj COCO** w prawym górnym rogu zapisuje sesję. Zweryfikowane
anotacje mają `label_source = human_verified` i to one pójdą do treningu.
