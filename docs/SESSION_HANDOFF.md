# Session Handoff — Dog FACS pipeline (keypoints + breed + AU + emotion)

> Заметки для продолжения в другой сессии. Дата: 2026-06-18.
> Цель проекта: качественный датасет эмоций собак (DogFACS): bbox → порода → keypoints → AU → эмоции → COCO.

---

## 0. TL;DR статус 4 моделей
| Модель | Было | Стало / статус |
|--------|------|----------------|
| **Keypoints** | ResNet34, точки «не там» (баг flip), conf ~0.37 | ✅ **HRNet-W48** + DARK + корректный flip + детектор морды + фильтр качества. NME_iod 0.118→**0.091**, PCK **0.748**, conf ~0.9. Смержено в main/develop. |
| **Порода** | плохо | ✅ Тиммейт: **EfficientNet-B4 @380, Top-1 91.5%** (в main, мы подтянули мержем). |
| **AU** | формулы | ⚠️ **Вариант A сделан** (правки формул, ветка `feature/au-improvements`, НЕ слита). Вариант C (нейросеть) — план есть, данных нет. |
| **Эмоции** | rule-based (DogFACS правила на AU) | ❌ ещё не трогали. Тренер эмоций (`train_emotion_keypoints.py`) учится на СЛУЧАЙНОМ ШУМЕ по умолчанию — переделать. |

---

## 1. Git — ветки и состояние
- **main / develop** — синхронизированы на коммите `6d28904`: keypoints W48 + детектор морды + фильтр качества + breed-фикс тиммейта. Запушены. Тесты 161 passed.
- **feature/keypoints-hrnet** — `6d28904`, запушена (история keypoints-работы). Можно удалить после подтверждения.
- **feature/au-improvements** — `d4bc734`, НЕ слита, НЕ запушена(?). Вариант A для AU. **TODO: смержить в develop/main как делали с keypoints** (тесты 164 passed, ruff чисто).
- Workflow проекта: `feature/* → develop → main` (мерж, не прямой коммит). origin: `eternaki/group-project` (public).

## 2. Веса и Git LFS
`.pt` хранятся через **Git LFS** (`.gitattributes`: `*.pt filter=lfs`). После клона/пулла тиммейтам:
```bash
git lfs install && git lfs pull
```
Активные веса в `models/`: `keypoints_dogflw.pt` (HRNet-W48, ~270MB), `dogface_yolo.pt` (детектор морды, 18MB), `breed.pt` (EffNet-B4), `yolov8m.pt` (bbox).
Бэкап-веса (`keypoints_hrnet*.pt`, `_resnet_OLD`) и `data/`-артефакты — в `.gitignore`, НЕ в репо (лежат локально).

## 3. Как запустить локально
```bash
# окружение: .venv (Python 3.12), torch CPU; ffmpeg/node для видео
.venv/Scripts/python.exe -m uvicorn apps.webapp.backend.main:app --host 127.0.0.1 --port 8000   # бэкенд :8000
cd apps/webapp/frontend && npm run dev    # фронтенд :5173 (проксирует /api на :8000)
# открыть http://localhost:5173
pytest -q           # тесты (164 passed)
ruff check .        # линтер
```
Бэкенд при старте грузит пайплайн; в логе видно `backbone: hrnet_w48` + `Detektor mordy`. Health: `GET /api/health`.
ВАЖНО: старые сессии в `apps/webapp/backend/sessions/` хранят СТАРЫЕ аннотации — чтобы увидеть новую модель, обрабатывать видео ЗАНОВО.

## 4. Что сделано по keypoints (детали)
- Корневая причина «точки не там» = `HorizontalFlip` в обучении БЕЗ перестановки лево/право пар (НЕ порядок каналов — он верный). Подтверждено эмпирически.
- Решение: HRNet-W48, вход 320→heatmap 80, AdaptiveWing loss, корректный flip (FLIP_PAIRS), DARK-сабпиксель, метрика NME_iod (внешние уголки глаз, как в статье).
- Прогресс: 256→0.139, 320→0.126, 320+sigma1.5→0.118, **W48+вес ушей→0.091 / PCK 0.748** (нормировка статьи; эталон ELD 0.0652).
- **Детектор морды** (YOLOv8n на DogFLW, mAP50 0.99): чинит кадрирование на полнокадровых собаках. `scripts/training/train_dogface_detector.py` (локальный CPU). В пайплайне: `_detect_face` + `_keypoints_on_region` в `inference.py`, фоллбэк на two-pass.
- **Фильтр качества кадров** (peak-селектор) — в датасет идут только: conf≥0.5, угол головы ≤30°, морда НЕ обрезана краем (margin 3%), резкость (Laplacian var ≥60). См. `packages/pipeline/peak_selector.py` + `process_video_for_dataset`.
- STAR loss пробовали — расходится (NaN), отброшен. Профиль/висячие уши остаются трудными (ограничение модели; чинится дообучением на профиле + GPU).
- Файлы: `packages/models/keypoints.py`, `scripts/training/train_keypoints_dogflw.py`, `scripts/debug/diagnose_kp_*.py`, `verify_keypoints_pipeline.py`. Размеченные примеры: `data/kp_results/`.

## 5. Что сделано по AU — вариант A (ветка feature/au-improvements)
Файл `packages/models/delta_action_units.py` (21 AU = ratio расстояний keypoints vs нейтральный кадр):
- **AD33≠AD35**: была идентичная формула. AD33(Blow)→ширина морды (MUZZLE_LEFT/RIGHT), AD35(LipBite)→щель губ.
- **AD19(TongueShow)**→через `KP.TONGUE_TIP=45` (видимость+позиция ниже губы).
- **EAD104**: задокументировано как ограничение 2D.
- AU уже выигрывает от лучших keypoints + гейтинг `MIN_AU_CONFIDENCE=0.30`.
- Тесты: `tests/test_models/test_delta_action_units.py` (+3, 164 passed).
- НЕ сделано (остаточные рычаги): median-baseline вместо 1 нейтрального кадра; пер-AU пороги (нужна калибровка на данных).

## 6. Следующие шаги (приоритеты)
1. **Смержить `feature/au-improvements`** в develop/main (как keypoints).
2. **AU вариант C — нейросеть** (нужны данные, см. ниже).
3. **Эмоции** — заменить rule-based на ML; сейчас `scripts/training/train_emotion_keypoints.py` по умолчанию учится на `np.random` (мусор). Аналогично: разобрать → исследовать → дизайн.
4. Опц.: дообучить keypoints на профиль/висячие уши (когда вернётся GPU-квота).

## 7. AU вариант C — план сбора данных (ключевое)
Контекст (исследование): автоматический dog-AU — открытая задача, SOTA F1≈0.29 (LSTM-автоэнкодер на временных рядах landmark'ов, Nature 2025). AU-размеченных данных нет (DogFLW = только landmarks; Bremhorst 248 видео лабрадоров — недоступен). Supervised AU-модель обучить нечем БЕЗ сбора меток.
**План бутстрапа меток через webapp:**
1. webapp уже умеет ручную правку AU (PATCH aus, recompute) → правки = метки.
2. Pre-fill формулой (теперь хороша на уверенных фронтальных кадрах) → человек подтверждает/правит.
3. Бинарные AU (active/нет), только уверенные фронтальные кадры.
4. Экспорт verified → CSV (`138 keypoints → 21 AU`).
5. Обучить MLP 138→21 (multi-label), ~300–800 кадров хватит (вход низкоразмерный, обучается на CPU). Это Sprint 16 (зависит от Sprint 15 = ручная разметка).
**TODO для C:** построить инструментарий — AU-редактор с pre-fill+тоглами, endpoint экспорта CSV, тренировочный скрипт MLP.

## 8. Kaggle (как обучать на GPU)
- CLI: `pip install kaggle`. Токен: формат `KGAT_...` → файл `~/.kaggle/access_token` (НЕ kaggle.json для новых CLI). **⚠️ ТОКЕН ЗАСВЕТИЛСЯ В ЧАТЕ ПРОШЛОЙ СЕССИИ — ОТОЗВАТЬ на kaggle.com/settings и создать новый.**
- Датасет: **`georgemartvel/dogflw`** (полные сцены + landmarks + bounding_boxes; НЕ lovodkin).
- **GPU обязательно T4**: `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4`. P100 (sm_60) несовместим с образным torch (CUDA error: no kernel image).
- **Квота 30ч/нед** — была исчерпана (W48-прогоны). Поэтому детектор морды обучали локально на CPU.
- Kernel'ы: `antonshkrebela/dogflw-keypoints-train`, `dogface-detector`, `dogflw-inspect`. Файлы kernel'ов: `C:\Users\Anton\.kaggle_kernels\`.
- Гочи: ASCII в kernel.py (иначе CLI падает на кодировке Windows); kernel пишет `run.log` в /kaggle/working; output только в конце (mid-run недоступен); клонировать код в /tmp (не /kaggle/working).

## 9. Ключевые гочи проекта
- Import shadowing: вложенная `Dog-Emotion-Classification/` может перехватывать import `packages` → `pip install -e .` после изменений структуры.
- `pyproject`: `pythonpath=[".", "apps/webapp/backend"]`.
- Тесты API: httpx `AsyncClient(transport=ASGITransport(app))` + `@pytest.mark.anyio` (не starlette TestClient).
- breeds.json — нейминг ImageNet (нижний регистр, "golden retriever"); тест сделан case-insensitive.

## 10. Источники (AU/keypoints)
- DogFLW dataset: arXiv 2405.11501; repo github.com/martvelge/DogFLW; Kaggle georgemartvel/dogflw.
- Dog landmarks+applications (AU, эмоции): Nature Sci Reports 2025, PMC12218811 (ELD NME 6.52; AU F1 0.29 LSTM-autoencoder; Bremhorst data).
- Springer 2024 (dog emotion, AU vs end-to-end): s00521-024-10042-3.
- py-feat (люди, AU): arXiv 2104.03509. HRNet: 1908.07919. STAR loss: 2306.02763. SuperAnimal: Nature Comms 2024 s41467-024-48792-2.
