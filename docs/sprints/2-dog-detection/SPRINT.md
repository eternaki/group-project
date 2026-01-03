# Sprint 2: Dog Detection Model

**Sprint Goal:** Train YOLOv8 to detect dogs in images with >85% mAP.

**Duration:** Weeks 5-7
**Semester:** 1
**Phase:** AI Development

---

## Overview

This sprint focuses on training and integrating the dog detection model. The model will use YOLOv8m architecture fine-tuned on dog images to output bounding boxes.

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 2.1 Training Data Preparation | U3 (Danylo Z.) | High |
| 2.2 YOLOv8 Fine-tuning | U3 (Danylo Z.) | High |
| 2.3 Model Evaluation | U3 (Danylo Z.) | High |
| 2.4 Model Integration | U3 (Danylo Z.) | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [2.1](stories/2.1-training-data-preparation.md) | Training Data Preparation | To Do |
| [2.2](stories/2.2-yolov8-finetuning.md) | YOLOv8 Fine-tuning | To Do |
| [2.3](stories/2.3-model-evaluation.md) | Model Evaluation | To Do |
| [2.4](stories/2.4-model-integration.md) | Model Integration | To Do |

---

## Success Criteria

- mAP > 85% on test set
- Inference time < 50ms per image on GPU
- Model integrated into packages/models/bbox.py

---

## Deliverables

- [ ] Training dataset in YOLO format
- [ ] Trained model weights (bbox.pt)
- [ ] Evaluation report with metrics
- [ ] BBoxModel class in codebase
- [ ] Unit tests for model

---

## Dependencies

- Sprint 1 completed (tech stack finalized)
- GPU access available
- Training data downloaded

---

## Technical Notes

**Model:** YOLOv8m
**Input:** 640x640 RGB image
**Output:** List of bounding boxes [x, y, w, h] with confidence scores
**Training Data:** Stanford Dogs + Open Images dog subset
