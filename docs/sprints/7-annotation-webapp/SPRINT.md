# Sprint 7: Annotation Webapp

**Sprint Goal:** Zbudować aplikację webową do ręcznej anotacji emocji psów (FastAPI + React).

**Duration:** Tygodnie 10-14
**Semester:** 1
**Phase:** Narzędzia anotacji

---

## Overview

Ten sprint tworzy aplikację webową umożliwiającą przeglądanie klatek wideo, edycję keypoints (drag & drop), ustawianie AU, wybór emocji i eksport do formatu COCO. Zastąpił pierwotnie planowaną aplikację Streamlit.

**Rzeczywisty stos technologiczny:**
- **Backend:** FastAPI (Python) z SessionStore i 10 endpointami REST
- **Frontend:** React + TypeScript (Vite, Zustand, Tailwind CSS)
- **Funkcje:** VideoUpload, Timeline, KeypointEditor, AUPanel, EmotionSelector, ExportPanel

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 7.1 App Scaffold | U1 (Danylo L.) | Medium |
| 7.2 Image/Video Upload | U1 (Danylo L.) | Medium |
| 7.3 Video Upload | U1 (Danylo L.) | Medium |
| 7.4 Results Visualization | U1 (Danylo L.) | High |
| 7.5 Export Functionality | U1 (Danylo L.) | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [7.1](stories/7.1-app-scaffold.md) | App Scaffold (FastAPI + React) | Done |
| [7.2](stories/7.2-image-upload.md) | Image Upload | Done |
| [7.3](stories/7.3-video-upload.md) | Video Upload | Done |
| [7.4](stories/7.4-results-visualization.md) | Annotation Editor (keypoints, AU, emocje) | Done |
| [7.5](stories/7.5-export-functionality.md) | Export Functionality (COCO JSON) | Done |

---

## Success Criteria

- [x] Upload wideo → wyodrębnianie klatek szczytowych
- [x] Edycja keypoints (drag & drop) na klatce
- [x] Panel AU z suwakami
- [x] Wybór emocji na psie
- [x] Eksport anotacji do COCO JSON
- [x] 143 testy przechodzą (pytest)

---

## Deliverables

- [x] Backend FastAPI (`apps/webapp/backend/`)
- [x] Frontend React (`apps/webapp/frontend/`)
- [x] SessionStore + 10 endpointów REST
- [x] Annotation Editor Modal
- [x] Export do COCO JSON

---

## Dependencies

- Sprint 6 (Pipeline) — do inference

---

## Technical Notes

**Backend:** `uvicorn apps.webapp.backend.main:app --reload`
**Frontend:** `cd apps/webapp/frontend && npm run dev`
**Testy:** `pytest tests/test_backend/`
**Opis:** `apps/webapp/README.md`
