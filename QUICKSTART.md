# Quick Start Guide - DogFACS Dataset Generator

## Крок 1: Установка залежностей

### Backend (FastAPI)
```bash
pip install -r apps/webapp/backend/requirements.txt
```

### Frontend (React)
Потрібен Node.js та npm. Завантажте з: https://nodejs.org/

```bash
cd apps/webapp/frontend
npm install
```

## Крок 2: Запуск тестів

```bash
python test_runner.py
```

Очікуваний результат: **6/6 тестів пройдено** (або 4-5/6 якщо є дрібні проблеми)

## Крок 3: Запуск застосунків

### Термінал 1: Backend
```bash
cd apps/webapp/backend
python main.py
```

Backend буде на: `http://localhost:8000`

Перевірка: `curl http://localhost:8000/api/health`

### Термінал 2: Frontend
```bash
cd apps/webapp/frontend
npm run dev
```

Frontend буде на: `http://localhost:5173`

Відкрийте в браузері: http://localhost:5173

## Крок 4: Тестування E2E

1. Завантажте тестове відео (20s, MP4)
2. Натисніть "Choose Video"
3. Натисніть "Process Video"
4. Дочекайтеся обробки (30-60 сек)
5. Перегляньте 10 peak frames
6. Перемкніть кілька AU checkboxів
7. Натисніть "Export COCO Dataset"
8. Перевірте JSON файл

## Troubleshooting

### ModuleNotFoundError: No module named 'fastapi'
```bash
pip install -r apps/webapp/backend/requirements.txt
```

### npm: command not found
Встановіть Node.js з https://nodejs.org/

### Port 8000 already in use
```bash
# Знайдіть процес
netstat -ano | findstr :8000
# Вбийте процес
taskkill /PID <PID> /F
```

### Port 5173 already in use
```bash
# У apps/webapp/frontend/vite.config.ts змініть port
server: {
  port: 5174,  // Інший port
}
```

## Докладніше

Дивіться повну документацію:
- [TESTING.md](TESTING.md) - Детальні інструкції з тестування
- [apps/webapp/README.md](apps/webapp/README.md) - Документація webapp

---

*Updated: January 2026 (Sprint 7)*
