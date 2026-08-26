# Pobieranie nagrań z Google Drive

Nagrania źródłowe leżą w folderze na Dysku Google i **nigdy nie wchodzą do
gita**. Trzeba je pobrać na maszynę, na której uruchamiamy anotację wsadową.

Są dwie drogi i różnią się jedną rzeczą — limitem.

| | bez klucza (`sync_drive_gdown.py`) | z kluczem API (`download_drive_folder.py`) |
|---|---|---|
| konfiguracja | żadna | jednorazowo klucz, ~5 minut |
| ile pobierze | **40-60 plików na dobę** | całość za jednym razem |
| do czego | dociągnięcie kilku nowych nagrań | pierwsze pobranie, regularne dociąganie |

Zmierzone 26.08.2026: anonimowe pobieranie zatrzymało się po 38 plikach,
wcześniej po 57 i po 48. Drive odpowiada wtedy na KAŻDY kolejny plik tym samym
`Cannot retrieve the public link (...) or have had many accesses`. Limit jest
dzienny i zwalnia się po kilkunastu godzinach — obejść go bez klucza się nie da.

---

## Zakładanie klucza Google Drive API

Klucz jest **darmowy** i nie wymaga karty płatniczej. Limit żądań (tysiące na
dobę) jest dla nas nieosiągalny.

1. **Konsola Google Cloud** — https://console.cloud.google.com/
   Zaloguj się kontem Google. Przy pierwszym wejściu trzeba zaakceptować
   warunki.

2. **Załóż projekt** — https://console.cloud.google.com/projectcreate
   Nazwa dowolna, np. `dogfacs-drive`. Organizacji nie trzeba ustawiać.
   Poczekaj, aż projekt się utworzy, i upewnij się, że jest wybrany
   w przełączniku u góry strony.

3. **Włącz Google Drive API** —
   https://console.cloud.google.com/apis/library/drive.googleapis.com
   Przycisk **Enable** / **Włącz**. Bez tego kroku klucz istnieje, ale każde
   żądanie wraca z błędem `accessNotConfigured`.

4. **Utwórz klucz** — https://console.cloud.google.com/apis/credentials
   **Create credentials** → **API key**. Klucz pojawi się od razu; skopiuj go.

5. **Ogranicz klucz** (nieobowiązkowe, ale zalecane)
   W tym samym miejscu **Edit API key** → *API restrictions* → *Restrict key*
   → zaznacz wyłącznie **Google Drive API**. Klucz przestaje wtedy być
   przydatny do czegokolwiek innego, gdyby wyciekł.

6. **Zapisz klucz u siebie** — plik `drive_key.txt` w korzeniu repozytorium:

   ```bash
   echo "AIzaSy...twój-klucz..." > drive_key.txt
   ```

   Albo przez zmienną środowiskową, jeśli wolisz nie trzymać go w pliku:

   ```bash
   export GDRIVE_API_KEY="AIzaSy...twój-klucz..."
   ```

> `drive_key.txt` jest w `.gitignore` i **nie może** trafić do repozytorium.
>
> Uwaga na `key.txt` w korzeniu — to klucz **Kaggle**, nie Google. Skrypt
> celowo go nie czyta: wcześniejsza wersja brała go w zastępstwie i wysyłała
> do Google, które odpowiadało nieczytelnym błędem uwierzytelnienia.

---

## Pobieranie

```bash
python scripts/download/download_drive_folder.py \
    https://drive.google.com/drive/folders/1jxUaN3Mq1ge8lFcPzwnN2ISl0E9k9mfQ
```

Domyślnie ląduje w `data/drive_dogs/`. Skrypt:

* schodzi **rekurencyjnie** do podfolderów i zachowuje ich strukturę — nazwa
  katalogu jest etykietą źródłową (`batch_annotate.py` czyta ją z
  `video_path.parent.name`), więc spłaszczenie zgubiłoby informację;
* jest **wznawialny** — pomija nazwy, które już leżą na dysku, gdziekolwiek
  w drzewie. Przerwanie i ponowne uruchomienie nie kosztuje nic poza
  listowaniem;
* **kasuje niedokończony plik** po błędzie — inaczej przy następnym przebiegu
  wyglądałby na pobrany i nagranie zniknęłoby ze zbioru po cichu.

Folder musi być udostępniony **„każdy z linkiem"**. Klucz API nie daje dostępu
do prywatnych plików — na takich zwróci `404`.

---

## Co dalej z pobranym materiałem

Etykiety z nazw katalogów (`angry/`, `happy/`) **nie są prawdą** — to tagi
autora podborki. Zmierzone: zgadzają się z regułami w 46% przypadków,
a w katalogu `angry/` leżą pliki `dogsmile_*`, `dogtailwag_*` i
`youtube_happy dog smiling_*`. Zapisujemy je do `emotion_label` dla
prześledzenia pochodzenia, ale nie używamy jako etykiety.

Dalszy ciąg — anotacja wsadowa i złożenie kolejki — w `CLAUDE.md`,
sekcja „Komendy Deweloperskie".
