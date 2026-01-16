# Sprint 3: Breed Classification Model

**Sprint Goal:** Train breed classifier with >80% Top-5 accuracy.

**Duration:** Weeks 6-8
**Semester:** 1
**Phase:** AI Development

---

## Overview

This sprint focuses on training and integrating a breed classification model using ViT or EfficientNet architecture, fine-tuned on Stanford Dogs dataset.

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 3.1 Breed Dataset Preparation | U3 (Danylo Z.) | High |
| 3.2 ViT/EfficientNet Fine-tuning | U3 (Danylo Z.) | High |
| 3.3 Breed Model Evaluation | U3 (Danylo Z.) | High |
| 3.4 Breed Model Integration | U3 (Danylo Z.) | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [3.1](stories/3.1-breed-dataset-preparation.md) | Breed Dataset Preparation | Done |
| [3.2](stories/3.2-classifier-finetuning.md) | Classifier Fine-tuning | Done |
| [3.3](stories/3.3-model-evaluation.md) | Model Evaluation | Done |
| [3.4](stories/3.4-model-integration.md) | Model Integration | Done |

---

## Success Criteria

- Top-5 accuracy > 80% on test set
- Top-1 accuracy > 60% on test set
- Support for 50+ breed classes
- Model integrated into packages/models/breed.py

---

## Deliverables

- [x] Training dataset prepared
- [x] Trained model weights (breed.pt)
- [x] Evaluation report with per-class metrics
- [x] BreedModel class in codebase
- [x] Breed labels mapping file

---

## Dependencies

- Sprint 2 (BBox model) - need cropped dog images
- GPU access

---

## Technical Notes

**Model:** EfficientNet-B4 or ViT-B/16
**Input:** 224x224 RGB cropped dog image
**Output:** Class probabilities for 50+ breeds
**Training Data:** Stanford Dogs Dataset (~20,000 images)
