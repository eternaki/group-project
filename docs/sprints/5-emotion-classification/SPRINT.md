# Sprint 5: Emotion Classification Model

**Sprint Goal:** Train emotion classifier with >70% accuracy using DogFACS.

**Duration:** Weeks 9-13
**Semester:** 1
**Phase:** AI Development

---

## Overview

This sprint focuses on training an emotion classifier that uses keypoint features to predict dog emotions based on DogFACS methodology.

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 5.1 Emotion Data Preparation | U2 (Anton) | High |
| 5.2 DogFACS Mapping | U2 (Anton) | High |
| 5.3 Emotion Classifier Training | U2 (Anton) | High |
| 5.4 Emotion Model Evaluation | U2 (Anton) | High |
| 5.5 Emotion Model Integration | U2 (Anton) | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [5.1](stories/5.1-emotion-data-preparation.md) | Emotion Data Preparation | To Do |
| [5.2](stories/5.2-dogfacs-mapping.md) | DogFACS Mapping | To Do |
| [5.3](stories/5.3-classifier-training.md) | Classifier Training | To Do |
| [5.4](stories/5.4-model-evaluation.md) | Model Evaluation | To Do |
| [5.5](stories/5.5-model-integration.md) | Model Integration | To Do |

---

## Success Criteria

- Accuracy > 70% on test set
- 6 emotion classes supported
- Model uses keypoint features as input
- Model integrated into packages/models/emotion.py

---

## Deliverables

- [ ] DogFACS to emotion mapping
- [ ] Training data with emotion labels
- [ ] Trained emotion classifier
- [ ] Evaluation report
- [ ] EmotionModel class

---

## Dependencies

- Sprint 4 (Keypoints model) - provides input features

---

## Technical Notes

**Model:** MLP (3 layers) or small CNN
**Input:** Keypoint features (flattened 20 points × 3 values)
**Output:** 6 emotion class probabilities
**Classes:** happy, sad, angry, fearful, relaxed, neutral
