# Sprint 6: Inference Pipeline

**Sprint Goal:** Create unified pipeline that runs all 4 models and exports COCO format.

**Duration:** Weeks 13-14
**Semester:** 1
**Phase:** AI Development (Integration)

---

## Overview

This sprint integrates all trained models into a single inference pipeline that processes images/videos and outputs COCO format annotations.

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 6.1 Pipeline Architecture | U3 (Danylo Z.) | High |
| 6.2 Single Frame Inference | U3 (Danylo Z.) | High |
| 6.3 Video Processing | U3 (Danylo Z.) | High |
| 6.4 COCO Export | U3 (Danylo Z.) | High |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [6.1](stories/6.1-pipeline-architecture.md) | Pipeline Architecture | To Do |
| [6.2](stories/6.2-single-frame-inference.md) | Single Frame Inference | To Do |
| [6.3](stories/6.3-video-processing.md) | Video Processing | To Do |
| [6.4](stories/6.4-coco-export.md) | COCO Export | To Do |

---

## Success Criteria

- Pipeline processes image → all annotations
- Video frame extraction at configurable FPS
- COCO JSON export validated with pycocotools
- End-to-end test passing

---

## Deliverables

- [ ] InferencePipeline class
- [ ] VideoProcessor class
- [ ] COCODataset class
- [ ] Integration tests
- [ ] Documentation

---

## Dependencies

- Sprints 2-5 completed (all models trained)

---

## Technical Notes

**Pipeline Flow:**
1. Load image/extract frame
2. Run BBox detection
3. For each detection: crop → breed + keypoints
4. Run emotion on keypoints
5. Combine into annotation
6. Export as COCO JSON
