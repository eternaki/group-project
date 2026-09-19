# Инструкция для агента-помощника: параллельная обработка кадров Dog FACS

Этот документ — для **агента Claude другого участника команды**, у которого есть
локальный GPU. Задача: помочь обработать часть датасета параллельно с основной
машиной, **не обрабатывая повторно** то, что уже сделано.

Прочитай целиком, затем выполняй по шагам. Все команды — из корня репозитория.

---

## 1. Что вообще происходит

Проект **Dog FACS** (Politechnika Gdańska, WETI) — датасет эмоций собак в формате COCO.
Люди вручную разметили в видео **момент начала и конца эмоции** (2 таймкода на видео).
Скрипт обработки для каждого размеченного видео:

1. берёт кадр в секунде «начало» и в секунде «конец»,
2. прогоняет через модели: **bbox (YOLOv8) → keypoints (46 точек HRNet) → порода (EfficientNet) → AU (21 Action Unit, дельта относительно нейтрального кадра)**,
3. эмоция берётся из ручной разметки,
4. пишет COCO: `annotations.json` + по классам + кадры (jpg) + `licenses.csv` + `frames.csv`.

Скрипт обработки: **`scripts/annotation/build_colab.py`** (несмотря на имя, отлично
работает и локально на GPU — просто читает видео из локальных папок).

---

## 2. Как делится работа (ГЛАВНОЕ — чтобы не дублировать)

Обработка разбита **по эмоциям**. Основная машина (Colab) считает **`happy` и `sad`**.
**Твоя задача — остальные:** `neutral`, `angry`, `surprise`, `fearful`.

Три уровня защиты от двойной обработки:
1. **Разные эмоции** — ты берёшь только свои (`COLAB_EMOTIONS`), пересечения с happy/sad нет.
2. **Чекпоинт `done_keys`** — внутри твоего прогона уже обработанные видео пропускаются
   (можно прерывать и продолжать, чекпоинт лежит в твоей выходной папке).
3. **Мердж с дедупом** — в конце всё объединяется скриптом `merge_releases.py`, который
   выкидывает дубли по `source_video`. Даже если случайно посчитаешь чужое видео —
   при объединении дубль отсеется, ничего страшного.

Итог твоей работы — **отдельная папка** `release_participant/`, которую ты отдашь
основной команде для мерджа. Ты НЕ трогаешь их папку `release_colab`.

---

## 3. Предусловия (это настраивает человек, не агент)

Тебе от участника нужны две вещи с Google Drive (агент сам их не достанет):

- **Веса моделей** (4 файла, ~393 МБ): `yolov8m.pt`, `breed.pt`, `keypoints_dogflw.pt`,
  `dogface_yolo.pt`. Лежат на общем Drive в папке `DOGS/dogfacs_colab/`.
- **Видео твоих эмоций** — папки с Drive `DOGS/` (см. таблицу ниже). Их надо
  **скачать локально** (через веб-интерфейс Drive «Скачать», `rclone` или Google Drive
  для десктопа). Метки разметки качать не нужно — они в git.

Если у участника нет доступа к папке `DOGS` — пусть попросит владельца (Маша) расшарить.

### Какие папки скачать под какие эмоции

| эмоция | папки на Drive (`DOGS/…`) |
|--------|---------------------------|
| `neutral`  | `DataSet_neutral`, `neutral_dog_masha` |
| `angry`    | `new_angry_dogs`, `angry_dogs_2`, `angry_dogs_3`, `angry_dogs_masha` |
| `surprise` | `new_surprised_dogs`, `envato_surprise`, `surprised_dogs_mafin` |
| `fearful`  | `envato_fearful` |

Скачай их в одну локальную директорию, например `~/dogfacs_videos/`, чтобы получилось
`~/dogfacs_videos/DataSet_neutral/…`, `~/dogfacs_videos/new_angry_dogs/…` и т.д.

---

## 4. Установка окружения

```bash
# 1. код + метки разметки (репозиторий публичный)
git clone https://github.com/eternaki/group-project.git
cd group-project

# 2. зависимости (Python 3.10+). Если есть CUDA — поставь torch под свою версию CUDA
pip install torch torchvision ultralytics timm opencv-python numpy pycocotools

# 3. веса моделей — положи 4 файла в ./models/
mkdir -p models
# скопируй сюда yolov8m.pt, breed.pt, keypoints_dogflw.pt, dogface_yolo.pt
ls models/   # должно быть 4 .pt файла
```

Проверь, что видит GPU:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
Должно быть `CUDA: True`. Если `False` — обработка пойдёт на CPU (медленнее, но тоже
сработает; скрипт сам это определит).

---

## 5. Запуск обработки

Скрипт управляется переменными окружения. Подставь свои пути:

```bash
export COLAB_VIDEO_DIRS="$HOME/dogfacs_videos/DataSet_neutral;$HOME/dogfacs_videos/neutral_dog_masha;$HOME/dogfacs_videos/new_angry_dogs;$HOME/dogfacs_videos/angry_dogs_2;$HOME/dogfacs_videos/angry_dogs_3;$HOME/dogfacs_videos/angry_dogs_masha;$HOME/dogfacs_videos/new_surprised_dogs;$HOME/dogfacs_videos/envato_surprise;$HOME/dogfacs_videos/surprised_dogs_mafin;$HOME/dogfacs_videos/envato_fearful"
export COLAB_LABELS_DIR="data/labels/dataset_final"
export COLAB_WORK="release_participant"          # рабочая папка (тут копится результат)
export COLAB_OUTPUT="release_participant"         # финальная папка = та же, локально
export COLAB_DEVICE="cuda"                         # сам упадёт на cpu, если GPU нет
export COLAB_EMOTIONS="neutral,angry,surprise,fearful"
export PYTHONPATH="."

python -m scripts.annotation.build_colab
```

Что увидишь:
- `Etykiet (angry,fearful,neutral,surprise): N | plików wideo w folderach: M`
- загрузку моделей, `Modele załadowane (cuda)`
- прогресс каждые 10 видео: `... 50/900 | kadrów 90 | {'ok': 45, ...}`
- чекпоинт пишется каждые 50 видео в `release_participant/_checkpoint.json`

**Можно прерывать в любой момент** (Ctrl+C, выключение). При повторном запуске той же
команды продолжит с чекпоинта — уже обработанные пропустятся.

### Что значат счётчики
- `ok` — успех, полные данные (bbox+keypoints+порода+AU+эмоция)
- `brak_psa` — модель не нашла собаку в кадре (вернуть нельзя, битых данных нет)
- `brak_neutralnej` — нет нейтрального кадра → нельзя посчитать AU (видео пропущено)
- `brak_odczytu` — файл не читается/битый
- `brak_w_indeksie` — видео из метки нет в скачанных папках (проверь, что скачал все папки из таблицы)

> Замечание про AV1: часть роликов (`o8zoo…`) закодированы в AV1. На локальном Linux с
> нормальным ffmpeg (`ffmpeg -decoders | grep av1` показывает `libdav1d`) они обработаются.
> Если декодера нет — попадут в `brak_odczytu`, тогда доставь `ffmpeg` с av1.

---

## 6. Результат и передача

По завершении в `release_participant/` будет:
- `annotations.json` + `annotations_neutral.json` / `annotations_angry.json` / …
- папки `neutral/`, `angry/`, `surprise/`, `fearful/` с кадрами (jpg)
- `licenses.csv`, `frames.csv`

**Отдай всю папку `release_participant/`** основной команде — загрузи её на общий Drive
(например в `DOGS/dogfacs_colab/release_participant/`) или передай архивом.

Основная команда объединит твой результат со своим одной командой:
```bash
python -m scripts.annotation.merge_releases release_final release_colab release_participant
```
`merge_releases.py` склеит оба в `release_final/`, убрав дубли по `source_video`
(при дубле побеждает вариант с посчитанным AU). Перенумерует id и соберёт единый COCO.

---

## 7. Если что-то идёт не так

- **Много `brak_w_indeksie`** — не все папки из таблицы (раздел 3) скачаны локально или
  пути в `COLAB_VIDEO_DIRS` неверные. Проверь `ls` по каждому пути.
- **`Modele załadowane (cpu)` вместо cuda** — torch не видит GPU. Проверь установку CUDA-сборки
  torch. На CPU тоже сработает, просто медленнее (~1 видео/сек против ~7/сек на GPU).
- **`brak_neutralnej` у многих** — норма для сложных видео (нет спокойного кадра пса);
  такие видео пропускаются, это ожидаемо.
- **Скрипт упал на середине** — просто запусти ту же команду снова, продолжит с чекпоинта.

Не меняй папку `happy`/`sad` и не запускай с `COLAB_EMOTIONS=happy,sad` — это делает
основная машина, чтобы не дублировать.
