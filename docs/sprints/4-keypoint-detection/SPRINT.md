# Sprint 4: Keypoint Detection Model

**Sprint Goal:** Train HRNet for facial keypoints with >75% PCK@0.1.

**Duration:** Weeks 7-10
**Semester:** 1
**Phase:** AI Development

---

## Overview

This sprint focuses on training HRNet to detect 20 facial keypoints on dogs. The keypoints are essential for the emotion classification model.

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 4.1 Keypoint Data Preparation | U2 (Anton) | High |
| 4.2 Keypoint Schema Definition | U2 (Anton) | High |
| 4.3 HRNet Training | U2 (Anton) | High |
| 4.4 Keypoint Model Evaluation | U2 (Anton) | High |
| 4.5 Keypoint Model Integration | U2 (Anton) | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [4.1](stories/4.1-keypoint-data-preparation.md) | Keypoint Data Preparation | To Do |
| [4.2](stories/4.2-keypoint-schema-definition.md) | Keypoint Schema Definition | To Do |
| [4.3](stories/4.3-hrnet-training.md) | HRNet Training | To Do |
| [4.4](stories/4.4-model-evaluation.md) | Model Evaluation | To Do |
| [4.5](stories/4.5-model-integration.md) | Model Integration | To Do |

---

## Success Criteria

- PCK@0.1 > 75% on test set
- 20 keypoints detected per dog face
- Model integrated into packages/models/keypoints.py

---

## Deliverables

- [ ] Keypoint schema documented
- [ ] Training dataset prepared
- [ ] Trained HRNet weights
- [ ] Evaluation report
- [ ] KeypointsModel class

---

## Dependencies

- Sprint 2 (BBox model) - for cropping
- Kaggle DogFLW dataset

---

## Technical Notes

**Model:** HRNet-W32
**Input:** 256x256 RGB cropped dog image
**Output:** 20 keypoints with (x, y, visibility)
**Training Data:** Kaggle DogFLW
