# Sprint 7: Demo Application

**Sprint Goal:** Build Streamlit app for visualization and university presentation.

**Duration:** Weeks 10-14
**Semester:** 1
**Phase:** AI Development (Demo)

---

## Overview

This sprint creates a Streamlit demo application that allows uploading images/videos and visualizing the annotation results.

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 7.1 App Scaffold | U1 (Danylo L.) | Medium |
| 7.2 Image Upload | U1 (Danylo L.) | Medium |
| 7.3 Video Upload | U1 (Danylo L.) | Medium |
| 7.4 Results Visualization | U1 (Danylo L.) | High |
| 7.5 Export Functionality | U1 (Danylo L.) | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [7.1](stories/7.1-app-scaffold.md) | App Scaffold | To Do |
| [7.2](stories/7.2-image-upload.md) | Image Upload | To Do |
| [7.3](stories/7.3-video-upload.md) | Video Upload | To Do |
| [7.4](stories/7.4-results-visualization.md) | Results Visualization | To Do |
| [7.5](stories/7.5-export-functionality.md) | Export Functionality | To Do |

---

## Success Criteria

- Upload image → see annotated results
- Upload video → see frame-by-frame results
- Visual overlays (bbox, keypoints, labels)
- Export annotations as JSON

---

## Deliverables

- [ ] Working Streamlit app
- [ ] Image/video upload functionality
- [ ] Visualization with overlays
- [ ] JSON export
- [ ] README with demo instructions

---

## Dependencies

- Sprint 6 (Pipeline) - for inference

---

## Technical Notes

**Framework:** Streamlit
**Run command:** `streamlit run apps/demo/app.py`
**Features:** Upload, visualize, export
