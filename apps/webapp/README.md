# DogFACS Dataset Generator - Web Application

Aplikacja webowa do generowania datasetu z wideo psów z wykorzystaniem rule-based DogFACS emotion classification.

## Architektura

```
apps/webapp/
├── backend/          # FastAPI backend
│   ├── main.py      # API endpoints
│   └── static/      # Zapisane klatki
└── frontend/        # React + Vite + Tailwind
    ├── src/
    │   ├── components/  # React components
    │   ├── store/       # Zustand state management
    │   ├── types/       # TypeScript types
    │   └── utils/       # API utilities
    └── package.json
```

## Instalacja

### Backend (FastAPI)

```bash
cd apps/webapp/backend

# Utwórz virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom serwer
python main.py
```

Backend będzie dostępny na `http://localhost:8000`

### Frontend (React + Vite)

```bash
cd apps/webapp/frontend

# Zainstaluj zależności
npm install

# Uruchom dev server
npm run dev
```

Frontend będzie dostępny na `http://localhost:5173`

## Użycie

1. **Upload Video**: Wybierz wideo psa (20s zalecane)
2. **Process**: Kliknij "Process Video" - pipeline automatycznie:
   - Wykryje keypoints dla wszystkich klatek
   - Auto-detekcja neutral frame (najstabilniejsza klatka)
   - Obliczy delta Action Units dla każdej klatki
   - Wybierze 10 peak frames (najwyższa TFM)
   - Sklasyfikuje emocje używając DogFACS rules
3. **Review**: Sprawdź peak frames i ich Action Units
4. **Toggle AUs**: Możesz ręcznie włączyć/wyłączyć AU dla każdej klatki
5. **Export**: Eksportuj dataset do formatu COCO JSON

## API Endpoints

### `POST /api/process_video`

Przetwarza wideo i zwraca peak frames.

**Parameters:**
- `file`: Plik wideo (multipart/form-data)
- `num_peaks`: Liczba peak frames (default: 10)
- `neutral_idx`: Opcjonalny indeks neutral frame (auto-detect jeśli null)
- `min_separation_frames`: Minimalna separacja między peaks (default: 30)

**Response:**
```json
{
  "session_id": "abc123",
  "video_filename": "dog.mp4",
  "neutral_frame_idx": 42,
  "neutral_frame_url": "/static/abc123/frame_0042.jpg",
  "peak_frames": [
    {
      "frame_idx": 100,
      "image_url": "/static/abc123/frame_0100.jpg",
      "aus": {
        "AU101": {
          "ratio": 1.15,
          "delta": 0.15,
          "is_active": true,
          "confidence": 0.9
        },
        ...
      },
      "emotion": "happy",
      "emotion_confidence": 0.85,
      "emotion_rule_applied": "happy_priority_100",
      "tfm_score": 2.345
    },
    ...
  ],
  "total_frames": 600
}
```

### `POST /api/export_coco`

Eksportuje dataset do formatu COCO JSON.

**Request Body:**
```json
{
  "peak_frames": [...],
  "neutral_frame_idx": 42,
  "video_filename": "dog.mp4"
}
```

**Response:**
Plik `dogfacs_dataset_dog.mp4.json` (download)

### `GET /api/health`

Health check.

**Response:**
```json
{
  "status": "ok",
  "pipeline_loaded": true
}
```

## Technologie

### Backend
- **FastAPI**: Web framework
- **OpenCV**: Video processing
- **NumPy**: Numerical operations
- **Packages**: Custom pipeline (YOLOv8, HRNet, DogFACS rules)

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **Zustand**: State management
- **Axios**: HTTP client

## Development

### Backend

```bash
# Linter
ruff check apps/webapp/backend/

# Type checking
mypy apps/webapp/backend/
```

### Frontend

```bash
# Linter
npm run lint

# Build production
npm run build

# Preview production build
npm run preview
```

## Troubleshooting

### Backend nie może załadować modeli

Upewnij się, że masz pobrane wagi modeli:
- `models/yolov8m.pt`
- `models/breed.pt`
- `models/keypoints_dogflw.pt`

### Frontend nie łączy się z backend

Sprawdź czy backend działa na `http://localhost:8000`:
```bash
curl http://localhost:8000/api/health
```

Vite proxy powinien automatycznie przekierować `/api` i `/static` do backend.

### CORS errors

Upewnij się, że frontend działa na `localhost:5173` (Vite default port).
CORS jest skonfigurowany dla `http://localhost:5173` i `http://localhost:3000`.
