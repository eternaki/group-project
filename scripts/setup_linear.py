"""
Скрипт для создания задач в Linear.

Использование:
    1. Получи API ключ: https://linear.app/settings/api → Personal API keys
    2. export LINEAR_API_KEY="lin_api_xxxxxxxxxx"
    3. python scripts/setup_linear.py

Или укажи ключ напрямую в переменной LINEAR_API_KEY ниже.
"""

import json
import os
import urllib.error
import urllib.request

LINEAR_API_KEY = os.environ.get("LINEAR_API_KEY", "")  # ← или вставь сюда
LINEAR_TEAM_KEY = "DOG"
LINEAR_API_URL = "https://api.linear.app/graphql"


def graphql(query: str, variables: dict | None = None) -> dict:
    """Выполняет GraphQL запрос к Linear API."""
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        LINEAR_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": LINEAR_API_KEY,
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_team_id() -> str:
    """Получает ID команды DOG."""
    query = """
    query {
      teams { nodes { id key name } }
    }
    """
    data = graphql(query)
    teams = data["data"]["teams"]["nodes"]
    for team in teams:
        if team["key"] == LINEAR_TEAM_KEY:
            return team["id"]
    raise ValueError(f"Команда {LINEAR_TEAM_KEY} не найдена. Доступные: {[t['key'] for t in teams]}")


def get_workflow_states(team_id: str) -> dict[str, str]:
    """Получает состояния задач (Todo, In Progress, Done)."""
    query = """
    query($teamId: String!) {
      workflowStates(filter: {team: {id: {eq: $teamId}}}) {
        nodes { id name type }
      }
    }
    """
    data = graphql(query, {"teamId": team_id})
    states = data["data"]["workflowStates"]["nodes"]
    return {s["name"]: s["id"] for s in states}


def create_cycle(team_id: str, name: str, description: str) -> str:
    """Создаёт спринт (Cycle) в Linear."""
    from datetime import datetime, timedelta
    # Спринты по 2 недели
    start = datetime.now()
    end = start + timedelta(weeks=2)
    query = """
    mutation($teamId: String!, $name: String!, $description: String!,
             $startsAt: DateTime!, $endsAt: DateTime!) {
      cycleCreate(input: {
        teamId: $teamId
        name: $name
        description: $description
        startsAt: $startsAt
        endsAt: $endsAt
      }) {
        cycle { id name }
        success
      }
    }
    """
    data = graphql(query, {
        "teamId": team_id,
        "name": name,
        "description": description,
        "startsAt": start.isoformat() + "Z",
        "endsAt": end.isoformat() + "Z",
    })
    return data["data"]["cycleCreate"]["cycle"]["id"]


def get_label_ids(team_id: str, label_names: list[str]) -> list[str]:
    """Получает ID лейблов по именам."""
    query = """
    query($teamId: String!) {
      issueLabels(filter: {team: {id: {eq: $teamId}}}) {
        nodes { id name }
      }
    }
    """
    data = graphql(query, {"teamId": team_id})
    labels = {lab["name"]: lab["id"] for lab in data["data"]["issueLabels"]["nodes"]}
    return [labels[n] for n in label_names if n in labels]


def create_issue(team_id: str, title: str, description: str,
                 priority: int, state_id: str,
                 cycle_id: str | None = None,
                 label_ids: list[str] | None = None) -> str:
    """Создаёт задачу в Linear."""
    # Priority: 0=No priority, 1=Urgent, 2=High, 3=Medium, 4=Low
    query = """
    mutation($teamId: String!, $title: String!, $description: String!,
             $priority: Int!, $stateId: String!,
             $cycleId: String, $labelIds: [String!]) {
      issueCreate(input: {
        teamId: $teamId
        title: $title
        description: $description
        priority: $priority
        stateId: $stateId
        cycleId: $cycleId
        labelIds: $labelIds
      }) {
        issue { id identifier title }
        success
      }
    }
    """
    data = graphql(query, {
        "teamId": team_id,
        "title": title,
        "description": description,
        "priority": priority,
        "stateId": state_id,
        "cycleId": cycle_id,
        "labelIds": label_ids or [],
    })
    issue = data["data"]["issueCreate"]["issue"]
    return issue["identifier"]


# ============================================================
# ЗАДАЧИ ПО СПРИНТАМ
# ============================================================

SPRINTS = [
    {
        "name": "Sprint 8 — Научная основа (DogFACS)",
        "description": "Привести AU коды и keypoints в соответствие с официальным DogFACS. 46 keypoints, 21 AU, 9 эмоций.",
        "issues": [
            {
                "title": "[S8] Расширить keypoints с 20 до 46 (DogFLW) в schemas.py",
                "description": "Обновить `packages/data/schemas.py`:\n- `NUM_KEYPOINTS = 46`\n- 46 названий по DogFLW анатомии (глаза×8, брови×6, уши×8, нос×4, рот×8, морда×4, прочее×8)\n- Обновить `SKELETON_CONNECTIONS`\n- Маппинг 1:1 (убрать урезание 46→20)\n- Обновить цвета keypoints",
                "priority": 1,  # Urgent
                "labels": ["model", "data"],
            },
            {
                "title": "[S8] Исправить AU коды в delta_action_units.py (21 официальных DogFACS)",
                "description": "Файл: `packages/models/delta_action_units.py`\n\nУДАЛИТЬ (не существуют в DogFACS): AU102, AU115, AU117, AU121\n\nИСПРАВИТЬ:\n- AU116 → Lower Lip Depressor (не веко!)\n- EAD102 → Ears Adductor (не Forward!)\n- AD37 → Lip Wipe\n- AD137 → Nose Lick\n\nДОБАВИТЬ: AU143, AU145, AU109, AU110, AU118, AU25, AU27, AD33, AD35, EAD101, EAD104, EAD105\n\nПереписать геометрические вычисления под 46 keypoints.",
                "priority": 1,
                "labels": ["model"],
            },
            {
                "title": "[S8] Добавить 3 новые эмоции (surprise, pain, submission) в emotion.py",
                "description": "Файл: `packages/models/emotion.py`\n\n```python\nEMOTION_CLASSES = ['happy','sad','angry','fearful','relaxed','neutral','surprise','pain','submission']\n```\n\nДобавить EmotionRule для:\n- surprise: AU101 + EAD101 + AU25 (рот приоткрыт)\n- pain: AU101 + EAD105 + AU143 + напряжение щёк\n- submission: EAD103 + AD137 (nose lick)\n\nИсточник: Mota-Rojas et al. 2021",
                "priority": 2,
                "labels": ["model"],
            },
            {
                "title": "[S8] Исправить emotion rules под правильные AU коды",
                "description": "Обновить все 6 существующих EmotionRule в `emotion.py`:\n- Убрать несуществующие AU102, AU117\n- AU116 теперь Lower Lip Depressor\n- EAD102 теперь Ears Adductor\n- AD37 теперь Lip Wipe, AD137 — Nose Lick\n- Обновить EAD101 (Forward) в happy rule",
                "priority": 2,
                "labels": ["model"],
            },
            {
                "title": "[S8] Починить координатный баг в inference.py (keypoints offset)",
                "description": "Файл: `packages/pipeline/inference.py`, строки ~487-488\n\nПроблема: при добавлении bbox offset к keypoints могут затрагиваться visibility значения:\n```python\nkeypoints_flat[0::3] += x  # OK\nkeypoints_flat[1::3] += y  # OK\n# visibility на индексах 2::3 не должна меняться\n```\nПроверить и добавить тест.",
                "priority": 2,
                "labels": ["pipeline"],
            },
            {
                "title": "[S8] Убрать дубликат HeadPoseEstimator",
                "description": "Два разных реализации с разными порогами:\n- `packages/models/head_pose.py` (основная)\n- `packages/pipeline/neutral_frame.py` (дубликат)\n\nОставить только основную, обновить импорты в neutral_frame.py.",
                "priority": 3,
                "labels": ["pipeline"],
            },
            {
                "title": "[S8] Обновить peak_selector.py под правильные AU веса",
                "description": "Файл: `packages/pipeline/peak_selector.py`\n\nУбрать веса для AU102, AU115, AU117, AU121.\nДобавить веса для AU143, AU145, EAD101, EAD104, EAD105.",
                "priority": 3,
                "labels": ["pipeline"],
            },
            {
                "title": "[S8] Обновить тесты под 46 keypoints и 21 AU",
                "description": "Обновить:\n- `tests/test_models/test_delta_action_units.py` — 21 AU\n- `tests/test_models/test_emotion_rules.py` — 9 эмоций\n- `tests/test_pipeline/test_neutral_frame.py` — 46 keypoints\n- `test_runner.py` — проверки NUM_KEYPOINTS=46, 21 AU",
                "priority": 3,
                "labels": ["pipeline"],
            },
        ],
    },
    {
        "name": "Sprint 9 — Backend Annotation API",
        "description": "REST API для полного редактирования аннотаций: keypoints, AU, эмоции, породы.",
        "issues": [
            {
                "title": "[S9] Реализовать SessionStore — хранение аннотаций",
                "description": "Класс `SessionStore` в `apps/webapp/backend/main.py`:\n\nСтруктура на диске:\n```\napps/webapp/backend/sessions/{session_id}/\n    metadata.json\n    annotations.json  # расширенный COCO\n    frames/           # JPEG кадры\n```\n\nМетоды: load_session(), save_session(), get_frame(), update_frame()",
                "priority": 1,
                "labels": ["demo"],
            },
            {
                "title": "[S9] PATCH /keypoints — обновить keypoints кадра",
                "description": "Endpoint: `PATCH /api/sessions/{session_id}/frames/{frame_idx}/keypoints`\n\nBody: `{keypoints: [{name, x, y, visibility}, ...], dog_index: 0}`\n\nВалидация: 46 точек, координаты в пределах изображения.",
                "priority": 1,
                "labels": ["demo"],
            },
            {
                "title": "[S9] PATCH /aus — обновить AU значения кадра",
                "description": "Endpoint: `PATCH /api/sessions/{session_id}/frames/{frame_idx}/aus`\n\nBody: `{aus: {AU101: 0.8, EAD103: 0.5, ...}, dog_index: 0}`\n\nВалидация: только официальные 21 DogFACS код, значения 0.0-5.0",
                "priority": 1,
                "labels": ["demo"],
            },
            {
                "title": "[S9] PATCH /emotion — обновить эмоцию кадра",
                "description": "Endpoint: `PATCH /api/sessions/{session_id}/frames/{frame_idx}/emotion`\n\nBody: `{emotion: 'happy', confidence: 0.95, source: 'manual', dog_index: 0}`\n\nПосле обновления: сохранить source='manual', не перезаписывать при recompute.",
                "priority": 1,
                "labels": ["demo"],
            },
            {
                "title": "[S9] PATCH /breed — обновить породу",
                "description": "Endpoint: `PATCH /api/sessions/{session_id}/frames/{frame_idx}/breed`\n\nBody: `{breed: 'Labrador Retriever', confidence: 1.0, dog_index: 0}`",
                "priority": 2,
                "labels": ["demo"],
            },
            {
                "title": "[S9] POST /recompute_aus — пересчитать AU из keypoints",
                "description": "Endpoint: `POST /api/sessions/{session_id}/frames/{frame_idx}/recompute_aus`\n\nПересчитывает AU геометрически из текущих keypoints.\nВозвращает обновлённые AU значения.\nНе перезаписывает AU с source='manual'.",
                "priority": 2,
                "labels": ["demo"],
            },
            {
                "title": "[S9] POST /recompute_emotion — пересчитать эмоцию из AU",
                "description": "Endpoint: `POST /api/sessions/{session_id}/frames/{frame_idx}/recompute_emotion`\n\nПрименяет DogFACS emotion rules к текущим AU.\nВозвращает обновлённую эмоцию.\nНе перезаписывает если emotion.source='manual'.",
                "priority": 2,
                "labels": ["demo"],
            },
            {
                "title": "[S9] GET /frames — список всех кадров сессии",
                "description": "Endpoint: `GET /api/sessions/{session_id}/frames`\n\nВозвращает:\n```json\n{frames: [{frame_idx, image_url, annotation_status, emotion, num_dogs, ...}]}\n```\n\nПоддержка фильтров: ?status=auto, ?emotion=happy",
                "priority": 2,
                "labels": ["demo"],
            },
            {
                "title": "[S9] Реализовать полный COCO export",
                "description": "Файл: `apps/webapp/backend/main.py`, endpoint `POST /api/export_coco`\n\nСейчас — заглушка (TODO). Реализовать через `packages/data/coco.py`.\n\nВключать:\n- bbox, 46 keypoints\n- 21 AU (intensity + source)\n- breed, emotion (9 классов)\n- annotation_status, neutral_frame_id\n- Метаданные видео",
                "priority": 2,
                "labels": ["data"],
            },
        ],
    },
    {
        "name": "Sprint 10 — Annotation Editor Frontend",
        "description": "Полноценный редактор аннотаций в React: keypoints drag&drop, AU слайдеры, выбор эмоции и породы.",
        "issues": [
            {
                "title": "[S10] Обновить TypeScript типы (46 keypoints, 21 AU, 9 эмоций)",
                "description": "Файл: `apps/webapp/frontend/src/types/index.ts`\n\nДобавить:\n```typescript\ninterface Keypoint { name: string; x: number; y: number; visibility: number }\ninterface ActionUnit { ratio: number; intensity: number; is_active: boolean; source: 'ai'|'manual' }\ninterface DogAnnotation { dog_index: number; bbox: []; breed: string|null; keypoints: Keypoint[]; aus: Record<string, ActionUnit>; emotion: string; annotation_status: 'auto'|'reviewed'|'verified' }\n```",
                "priority": 1,
                "labels": ["demo"],
            },
            {
                "title": "[S10] KeypointEditor — canvas drag&drop редактор 46 точек",
                "description": "Компонент: `apps/webapp/frontend/src/components/KeypointEditor.tsx`\n\nФункционал:\n- `<canvas>` поверх изображения собаки\n- 46 точек цветными кружками (по анатомическим группам)\n- Drag&drop для перемещения (mousedown + mousemove)\n- ПКМ — toggle visibility\n- Skeleton connections\n- Кнопка 'Пересчитать AU' → POST /recompute_aus\n\nЦвета групп: глаза=зелёный, брови=фиолетовый, уши=оранжевый, нос=синий, рот=жёлтый, морда=розовый",
                "priority": 1,
                "labels": ["demo"],
            },
            {
                "title": "[S10] AUPanel — слайдеры 0-5 для 21 AU с группировкой",
                "description": "Компонент: `apps/webapp/frontend/src/components/AUPanel.tsx`\n\nГруппы:\n1. Верхнее лицо: AU101 (Inner Brow Raiser), AU143 (Eye Closure), AU145 (Blink)\n2. Нижнее лицо: AU109, AU110, AU12, AU116, AU118, AU25, AU26, AU27\n3. Дескрипторы: AD19, AD33, AD35, AD37, AD137\n4. Уши: EAD101, EAD102, EAD103, EAD104, EAD105\n\nКаждый AU: название на русском, код, слайдер intensity 0-5, checkbox active, badge AI/Manual\nКнопка 'Пересчитать эмоцию' → POST /recompute_emotion",
                "priority": 1,
                "labels": ["demo"],
            },
            {
                "title": "[S10] EmotionSelector — выбор из 9 эмоций",
                "description": "Компонент: `apps/webapp/frontend/src/components/EmotionSelector.tsx`\n\nФункционал:\n- Dropdown с 9 эмоциями и описаниями\n- Ползунок confidence\n- Badge AI/Manual (source)\n- Подсказки: какие AU соответствуют эмоции\n- После выбора → PATCH /emotion",
                "priority": 2,
                "labels": ["demo"],
            },
            {
                "title": "[S10] BreedPicker — поиск породы из 120+ вариантов",
                "description": "Компонент: `apps/webapp/frontend/src/components/BreedPicker.tsx`\n\nФункционал:\n- Searchable dropdown из breeds.json\n- Top-5 AI предсказаний с confidence\n- После выбора → PATCH /breed",
                "priority": 2,
                "labels": ["demo"],
            },
            {
                "title": "[S10] DogSelector — переключение между собаками в кадре",
                "description": "Компонент: `apps/webapp/frontend/src/components/DogSelector.tsx`\n\nФункционал:\n- Tabs/кнопки Dog #1, Dog #2, ...\n- Показывает bbox каждой собаки на миниатюре\n- Синхронизирует все редакторы с выбранной собакой\n- Кнопка 'Добавить собаку' для ручной разметки",
                "priority": 2,
                "labels": ["demo"],
            },
            {
                "title": "[S10] Обновить PeakFrameCard — вкладки редактора",
                "description": "Файл: `apps/webapp/frontend/src/components/PeakFrameCard.tsx`\n\nТекущая карточка только просмотр.\nНовая структура:\n```\n[Keypoints] [Action Units] [Emotion] [Breed] [Info]\n```\nКаждая вкладка — соответствующий компонент-редактор.",
                "priority": 2,
                "labels": ["demo"],
            },
            {
                "title": "[S10] Обновить Zustand store — sync с backend",
                "description": "Файл: `apps/webapp/frontend/src/store/useStore.ts`\n\nДобавить actions с optimistic updates:\n```typescript\nupdateKeypoints(frameIdx, dogIdx, keypoints) → PATCH\nupdateAU(frameIdx, dogIdx, auName, value) → PATCH\nupdateEmotion(frameIdx, dogIdx, emotion) → PATCH\nupdateBreed(frameIdx, dogIdx, breed) → PATCH\nrecomputeAUs(frameIdx, dogIdx) → POST → store update\nrecomputeEmotion(frameIdx, dogIdx) → POST → store update\n```",
                "priority": 1,
                "labels": ["demo"],
            },
        ],
    },
    {
        "name": "Sprint 11 — Multi-dog Pipeline",
        "description": "Исправить pipeline для полноценной обработки всех собак в кадре.",
        "issues": [
            {
                "title": "[S11] Проверить и исправить multi-dog в inference.py",
                "description": "Файл: `packages/pipeline/inference.py`\n\nПроверить что pipeline выполняет keypoints+breed+AU+emotion для КАЖДОЙ обнаруженной собаки.\nДобавить dog_index к каждой аннотации.\nДобавить фильтр min_bbox_area (убрать мелкие ложные детекции).",
                "priority": 1,
                "labels": ["pipeline"],
            },
            {
                "title": "[S11] E2E тест: видео с несколькими собаками",
                "description": "Создать интеграционный тест:\n- Входное изображение/видео с 2+ собаками\n- Проверить что все собаки детектированы\n- Проверить что каждая имеет keypoints, breed, AU, emotion\n- Проверить dog_index корректен",
                "priority": 2,
                "labels": ["pipeline"],
            },
            {
                "title": "[S11] Обновить COCO export для multi-dog",
                "description": "Убедиться что COCO JSON корректно сохраняет несколько аннотаций на один image_id (разные dog_index).",
                "priority": 2,
                "labels": ["data"],
            },
        ],
    },
]


def main() -> None:
    """Создаёт все спринты и задачи в Linear."""
    if not LINEAR_API_KEY:
        print("ERROR: Нет LINEAR_API_KEY!")
        print("Получи ключ: https://linear.app/settings/api → Personal API keys")
        print('Затем: export LINEAR_API_KEY="lin_api_xxxxxxxx"')
        return

    print("Подключение к Linear API...")
    team_id = get_team_id()
    print(f"Команда: {LINEAR_TEAM_KEY} (ID: {team_id})")

    states = get_workflow_states(team_id)
    print(f"Состояния: {list(states.keys())}")

    todo_state_id = states.get("Todo") or states.get("Backlog") or list(states.values())[0]

    for sprint_data in SPRINTS:
        print(f"\nСоздаю спринт: {sprint_data['name']}")
        cycle_id = create_cycle(team_id, sprint_data["name"], sprint_data["description"])
        print(f"  Спринт создан: {cycle_id}")

        for issue_data in sprint_data["issues"]:
            label_ids = get_label_ids(team_id, issue_data.get("labels", []))
            priority = issue_data.get("priority", 3)

            identifier = create_issue(
                team_id=team_id,
                title=issue_data["title"],
                description=issue_data["description"],
                priority=priority,
                state_id=todo_state_id,
                cycle_id=cycle_id,
                label_ids=label_ids,
            )
            print(f"  ✓ {identifier}: {issue_data['title'][:60]}")

    print("\nГотово! Все задачи созданы в Linear.")


if __name__ == "__main__":
    main()
