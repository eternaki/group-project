# Epics and User Stories

**Project:** Dog FACS Dataset
**Version:** 1.0
**Date:** 2025-01-03

---

## Team Reference

| ID | Member | Role |
|----|--------|------|
| U1 | Danylo Lohachov | Coordinator / Documentation / QA / Frontend |
| U2 | Anton Shkrebela | AI/ML (Keypoints & DogFACS models) |
| U3 | Danylo Zherzdiev | Backend (BBox & Breed models, Pipeline, COCO) |
| U4 | Mariia Volkova | Data Engineer (Collection & Verification) |

---

## Epic Overview

| Epic ID | Title | Owner | Stories |
|---------|-------|-------|---------|
| E1 | Project Setup & Research | All | 5 |
| E2 | Dog Detection Model | U3 | 4 |
| E3 | Breed Classification Model | U3 | 4 |
| E4 | Keypoint Detection Model | U2 | 5 |
| E5 | Emotion Classification Model | U2 | 5 |
| E6 | Inference Pipeline | U3 | 4 |
| E7 | Demo Application | U1 | 5 |
| E8 | Data Collection | U4 | 4 |
| E9 | Batch Annotation | U3, U2 | 3 |
| E10 | Manual Verification | U4, U1 | 3 |
| E11 | Dataset Finalization | U3 | 3 |
| E12 | Statistics & Reporting | U1, U4 | 4 |

---

## Epic 1: Project Setup & Research

**Goal:** Set up development environment and research existing solutions.

### Story 1.1: Repository Setup
**Assignee:** U1
**Priority:** High

**Description:**
As a developer, I want the repository structure created so that the team can start development.

**Acceptance Criteria:**
- [x] Monorepo structure created
- [x] pyproject.toml with dependencies
- [x] .gitignore configured
- [x] CI/CD workflow (GitHub Actions)
- [x] README.md with setup instructions

---

### Story 1.2: DogFACS & COCO Research
**Assignee:** U1, U3
**Priority:** High

**Description:**
As a researcher, I want to understand DogFACS methodology and COCO format so that annotations follow standards.

**Acceptance Criteria:**
- [ ] DogFACS paper summarized in docs/
- [ ] COCO format specification documented
- [ ] Keypoint schema defined (20+ points)
- [ ] Emotion categories finalized

---

### Story 1.3: Dataset Analysis
**Assignee:** U2, U4
**Priority:** High

**Description:**
As a data scientist, I want to analyze existing datasets so that I know what training data is available.

**Acceptance Criteria:**
- [ ] HuggingFace datasets evaluated
- [ ] Kaggle datasets evaluated
- [ ] Data quality report created
- [ ] Training data sources selected

---

### Story 1.4: Model Architecture Research
**Assignee:** U2, U3
**Priority:** High

**Description:**
As an ML engineer, I want to research SOTA models so that I choose the best architectures.

**Acceptance Criteria:**
- [ ] YOLOv8 capabilities documented
- [ ] HRNet for keypoints evaluated
- [ ] ViT/EfficientNet compared
- [ ] Final architecture decisions documented

---

### Story 1.5: Tech Stack Finalization
**Assignee:** All
**Priority:** High

**Description:**
As a team, we want to finalize the technology stack so that everyone uses the same tools.

**Acceptance Criteria:**
- [ ] Python version confirmed (3.10+)
- [ ] PyTorch version confirmed
- [ ] Training infrastructure decided (local/cloud)
- [ ] Development tools configured (ruff, mypy)

---

## Epic 2: Dog Detection Model

**Goal:** Train YOLOv8 to detect dogs with >85% mAP.

### Story 2.1: Training Data Preparation
**Assignee:** U3
**Priority:** High

**Description:**
As an ML engineer, I want to prepare training data for dog detection.

**Acceptance Criteria:**
- [ ] Dog images collected from Open Images/Stanford Dogs
- [ ] Data converted to YOLO format
- [ ] Train/val/test split created (80/10/10)
- [ ] Data augmentation pipeline configured

---

### Story 2.2: YOLOv8 Fine-tuning
**Assignee:** U3
**Priority:** High

**Description:**
As an ML engineer, I want to fine-tune YOLOv8 on dog detection.

**Acceptance Criteria:**
- [ ] YOLOv8m base model loaded
- [ ] Training script created (scripts/training/train_bbox.py)
- [ ] Model trained for sufficient epochs
- [ ] Checkpoints saved

---

### Story 2.3: Model Evaluation
**Assignee:** U3
**Priority:** High

**Description:**
As an ML engineer, I want to evaluate the detection model.

**Acceptance Criteria:**
- [ ] mAP calculated on test set
- [ ] mAP > 85% achieved
- [ ] Inference speed measured
- [ ] Results documented

---

### Story 2.4: Model Integration
**Assignee:** U3
**Priority:** Medium

**Description:**
As a developer, I want to integrate the bbox model into the codebase.

**Acceptance Criteria:**
- [ ] BBoxModel class implemented (packages/models/bbox.py)
- [ ] Model weights stored properly
- [ ] Unit tests written
- [ ] Documentation updated

---

## Epic 3: Breed Classification Model

**Goal:** Train breed classifier with >80% Top-5 accuracy.

### Story 3.1: Breed Dataset Preparation
**Assignee:** U3
**Priority:** High

**Description:**
As an ML engineer, I want to prepare breed classification data.

**Acceptance Criteria:**
- [ ] Stanford Dogs Dataset downloaded
- [ ] At least 50 breeds selected
- [ ] Class balance analyzed
- [ ] Train/val/test split created

---

### Story 3.2: ViT/EfficientNet Fine-tuning
**Assignee:** U3
**Priority:** High

**Description:**
As an ML engineer, I want to train a breed classifier.

**Acceptance Criteria:**
- [ ] Architecture selected (ViT or EfficientNet)
- [ ] Training script created (scripts/training/train_breed.py)
- [ ] Model trained
- [ ] Best checkpoint selected

---

### Story 3.3: Breed Model Evaluation
**Assignee:** U3
**Priority:** High

**Description:**
As an ML engineer, I want to evaluate breed classification accuracy.

**Acceptance Criteria:**
- [ ] Top-1 and Top-5 accuracy calculated
- [ ] Top-5 accuracy > 80%
- [ ] Confusion matrix generated
- [ ] Per-class accuracy analyzed

---

### Story 3.4: Breed Model Integration
**Assignee:** U3
**Priority:** Medium

**Description:**
As a developer, I want to integrate the breed model into the codebase.

**Acceptance Criteria:**
- [ ] BreedModel class implemented (packages/models/breed.py)
- [ ] Breed labels mapping created
- [ ] Unit tests written
- [ ] Documentation updated

---

## Epic 4: Keypoint Detection Model

**Goal:** Train HRNet for facial keypoints with >75% PCK@0.1.

### Story 4.1: Keypoint Data Preparation
**Assignee:** U2
**Priority:** High

**Description:**
As an ML engineer, I want to prepare keypoint training data.

**Acceptance Criteria:**
- [ ] Kaggle DogFLW dataset downloaded
- [ ] Keypoint schema defined (20 points)
- [ ] Data format standardized
- [ ] Augmentations configured

---

### Story 4.2: Keypoint Schema Definition
**Assignee:** U2
**Priority:** High

**Description:**
As a researcher, I want to define the keypoint schema.

**Acceptance Criteria:**
- [ ] 20 facial keypoints defined
- [ ] Keypoint names documented
- [ ] Skeleton connections defined
- [ ] Schema added to packages/data/schemas.py

---

### Story 4.3: HRNet Training
**Assignee:** U2
**Priority:** High

**Description:**
As an ML engineer, I want to train HRNet for dog facial keypoints.

**Acceptance Criteria:**
- [ ] HRNet-W32 base model configured
- [ ] Training script created (scripts/training/train_keypoints.py)
- [ ] Model trained
- [ ] Best checkpoint selected

---

### Story 4.4: Keypoint Model Evaluation
**Assignee:** U2
**Priority:** High

**Description:**
As an ML engineer, I want to evaluate keypoint detection accuracy.

**Acceptance Criteria:**
- [ ] PCK@0.1 calculated
- [ ] PCK@0.1 > 75%
- [ ] Per-keypoint accuracy analyzed
- [ ] Visualization of predictions created

---

### Story 4.5: Keypoint Model Integration
**Assignee:** U2
**Priority:** Medium

**Description:**
As a developer, I want to integrate the keypoint model.

**Acceptance Criteria:**
- [ ] KeypointsModel class implemented (packages/models/keypoints.py)
- [ ] Keypoint postprocessing added
- [ ] Unit tests written
- [ ] Documentation updated

---

## Epic 5: Emotion Classification Model

**Goal:** Train emotion classifier with >70% accuracy using DogFACS.

### Story 5.1: Emotion Data Preparation
**Assignee:** U2
**Priority:** High

**Description:**
As an ML engineer, I want to prepare emotion training data.

**Acceptance Criteria:**
- [ ] HuggingFace emotion datasets downloaded
- [ ] Emotion categories defined (6 classes)
- [ ] Data labeled and cleaned
- [ ] Class imbalance addressed

---

### Story 5.2: DogFACS Mapping
**Assignee:** U2
**Priority:** High

**Description:**
As a researcher, I want to map emotions to DogFACS action units.

**Acceptance Criteria:**
- [ ] DogFACS action units documented
- [ ] Emotion-to-AU mapping created
- [ ] Mapping validated with examples
- [ ] Schema added to packages/models/emotion.py

---

### Story 5.3: Emotion Classifier Training
**Assignee:** U2
**Priority:** High

**Description:**
As an ML engineer, I want to train the emotion classifier.

**Acceptance Criteria:**
- [ ] Architecture defined (MLP on keypoint features)
- [ ] Training script created (scripts/training/train_emotion.py)
- [ ] Model trained
- [ ] Best checkpoint selected

---

### Story 5.4: Emotion Model Evaluation
**Assignee:** U2
**Priority:** High

**Description:**
As an ML engineer, I want to evaluate emotion classification.

**Acceptance Criteria:**
- [ ] Accuracy calculated
- [ ] Accuracy > 70%
- [ ] Per-class metrics analyzed
- [ ] Confusion matrix generated

---

### Story 5.5: Emotion Model Integration
**Assignee:** U2
**Priority:** Medium

**Description:**
As a developer, I want to integrate the emotion model.

**Acceptance Criteria:**
- [ ] EmotionModel class implemented (packages/models/emotion.py)
- [ ] DogFACS labels added
- [ ] Unit tests written
- [ ] Documentation updated

---

## Epic 6: Inference Pipeline

**Goal:** Create unified pipeline that runs all models.

### Story 6.1: Pipeline Architecture
**Assignee:** U3
**Priority:** High

**Description:**
As a developer, I want to design the inference pipeline.

**Acceptance Criteria:**
- [ ] Pipeline class designed
- [ ] Model loading strategy defined
- [ ] Data flow documented
- [ ] Error handling planned

---

### Story 6.2: Single Frame Inference
**Assignee:** U3
**Priority:** High

**Description:**
As a developer, I want to process single frames through all models.

**Acceptance Criteria:**
- [ ] InferencePipeline class implemented
- [ ] process_frame() method works
- [ ] All 4 models called in sequence
- [ ] Annotations returned correctly

---

### Story 6.3: Video Processing
**Assignee:** U3
**Priority:** High

**Description:**
As a developer, I want to extract frames from videos.

**Acceptance Criteria:**
- [ ] VideoProcessor class implemented
- [ ] Frame extraction at configurable FPS
- [ ] Memory-efficient iteration
- [ ] Support for MP4 format

---

### Story 6.4: COCO Export
**Assignee:** U3
**Priority:** High

**Description:**
As a developer, I want to export annotations to COCO format.

**Acceptance Criteria:**
- [ ] COCODataset class implemented
- [ ] All required COCO fields populated
- [ ] Custom fields added (breed_id, emotion_id)
- [ ] JSON export working

---

## Epic 7: Demo Application

**Goal:** Build Streamlit app for visualization.

### Story 7.1: App Scaffold
**Assignee:** U1
**Priority:** Medium

**Description:**
As a developer, I want to create the demo app structure.

**Acceptance Criteria:**
- [ ] Streamlit app initialized
- [ ] Basic layout created
- [ ] Configuration added
- [ ] App runs locally

---

### Story 7.2: Image Upload
**Assignee:** U1
**Priority:** Medium

**Description:**
As a user, I want to upload images for analysis.

**Acceptance Criteria:**
- [ ] File uploader component added
- [ ] Image preview displayed
- [ ] Support for JPG, PNG
- [ ] File size validation

---

### Story 7.3: Video Upload
**Assignee:** U1
**Priority:** Medium

**Description:**
As a user, I want to upload short videos for analysis.

**Acceptance Criteria:**
- [ ] Video upload supported
- [ ] Frame extraction preview
- [ ] Support for MP4
- [ ] Duration limit (max 30 sec)

---

### Story 7.4: Results Visualization
**Assignee:** U1
**Priority:** High

**Description:**
As a user, I want to see annotated results visually.

**Acceptance Criteria:**
- [ ] Bounding box drawn on image
- [ ] Keypoints drawn with skeleton
- [ ] Breed label displayed
- [ ] Emotion displayed with confidence

---

### Story 7.5: Export Functionality
**Assignee:** U1
**Priority:** Medium

**Description:**
As a user, I want to export annotations as JSON.

**Acceptance Criteria:**
- [ ] Download button added
- [ ] COCO format JSON generated
- [ ] Filename includes timestamp
- [ ] Export works for single image

---

## Epic 8: Data Collection

**Goal:** Collect 2,500 YouTube videos of dogs.

### Story 8.1: Video Search Strategy
**Assignee:** U4
**Priority:** High

**Description:**
As a data engineer, I want to define video search criteria.

**Acceptance Criteria:**
- [ ] Search queries defined per emotion
- [ ] Quality criteria established
- [ ] Duration requirements (15-25 sec)
- [ ] Breed diversity requirements

---

### Story 8.2: Download Script
**Assignee:** U4
**Priority:** High

**Description:**
As a data engineer, I want to automate video downloading.

**Acceptance Criteria:**
- [ ] yt-dlp script created (scripts/download/)
- [ ] Video metadata saved
- [ ] Error handling for unavailable videos
- [ ] Progress tracking

---

### Story 8.3: Video Preprocessing
**Assignee:** U4
**Priority:** Medium

**Description:**
As a data engineer, I want to preprocess downloaded videos.

**Acceptance Criteria:**
- [ ] Videos trimmed to 20 sec
- [ ] Resolution standardized
- [ ] Invalid videos filtered
- [ ] Storage organized

---

### Story 8.4: Collection Progress Tracking
**Assignee:** U4
**Priority:** Medium

**Description:**
As a coordinator, I want to track collection progress.

**Acceptance Criteria:**
- [ ] Progress spreadsheet/database
- [ ] Videos per emotion category tracked
- [ ] Breed distribution tracked
- [ ] Target: 2,500 videos

---

## Epic 9: Batch Annotation

**Goal:** Auto-annotate 25,000 frames using the pipeline.

### Story 9.1: Batch Processing Script
**Assignee:** U3
**Priority:** High

**Description:**
As a developer, I want to run batch annotation on all videos.

**Acceptance Criteria:**
- [ ] Batch script created (scripts/annotation/)
- [ ] Processes all videos in directory
- [ ] Extracts frames at 1 FPS
- [ ] Saves annotations progressively

---

### Story 9.2: GPU Optimization
**Assignee:** U2, U3
**Priority:** Medium

**Description:**
As a developer, I want to optimize batch processing speed.

**Acceptance Criteria:**
- [ ] Batch inference implemented
- [ ] GPU memory optimized
- [ ] Processing speed > 10 FPS
- [ ] Progress logging added

---

### Story 9.3: Annotation Quality Monitoring
**Assignee:** U2
**Priority:** Medium

**Description:**
As a QA engineer, I want to monitor annotation quality during batch processing.

**Acceptance Criteria:**
- [ ] Confidence score logging
- [ ] Low-confidence frames flagged
- [ ] Sample visualization exported
- [ ] Quality metrics tracked

---

## Epic 10: Manual Verification

**Goal:** Verify 25% of annotations manually (~6,250 frames).

### Story 10.1: Verification Tool Setup
**Assignee:** U4
**Priority:** High

**Description:**
As an annotator, I want a tool to verify annotations.

**Acceptance Criteria:**
- [ ] Verification interface ready (Label Studio or custom)
- [ ] Annotations loaded for review
- [ ] Correction workflow defined
- [ ] Instructions documented

---

### Story 10.2: Sample Selection
**Assignee:** U1
**Priority:** Medium

**Description:**
As a QA engineer, I want to select a representative sample for verification.

**Acceptance Criteria:**
- [ ] Stratified sampling by emotion
- [ ] Stratified sampling by breed
- [ ] 6,250 frames selected
- [ ] Sample distribution documented

---

### Story 10.3: Verification Execution
**Assignee:** U4, U1
**Priority:** High

**Description:**
As annotators, we want to verify the selected samples.

**Acceptance Criteria:**
- [ ] All 6,250 frames reviewed
- [ ] Corrections recorded
- [ ] Inter-annotator agreement measured
- [ ] Verification completed

---

## Epic 11: Dataset Finalization

**Goal:** Create final COCO dataset with all annotations.

### Story 11.1: Merge Annotations
**Assignee:** U3
**Priority:** High

**Description:**
As a developer, I want to merge auto and manual annotations.

**Acceptance Criteria:**
- [ ] Manual corrections applied
- [ ] Auto annotations retained where not corrected
- [ ] Merged dataset created
- [ ] No duplicate annotations

---

### Story 11.2: COCO Validation
**Assignee:** U3
**Priority:** High

**Description:**
As a developer, I want to validate COCO format compliance.

**Acceptance Criteria:**
- [ ] pycocotools validation passes
- [ ] All required fields present
- [ ] Image-annotation mapping correct
- [ ] No orphan annotations

---

### Story 11.3: Dataset Export
**Assignee:** U3
**Priority:** High

**Description:**
As a developer, I want to export the final dataset.

**Acceptance Criteria:**
- [ ] annotations.json generated
- [ ] Images organized properly
- [ ] Dataset size verified (25,000+ frames)
- [ ] Final dataset committed/uploaded

---

## Epic 12: Statistics & Reporting

**Goal:** Generate dataset statistics and final report.

### Story 12.1: Statistics Notebook
**Assignee:** U1, U4
**Priority:** High

**Description:**
As a researcher, I want to analyze dataset statistics.

**Acceptance Criteria:**
- [ ] Jupyter notebook created
- [ ] Emotion distribution histogram
- [ ] Breed distribution chart
- [ ] Keypoint statistics calculated

---

### Story 12.2: Quality Assessment
**Assignee:** U1
**Priority:** High

**Description:**
As a QA engineer, I want to assess annotation quality.

**Acceptance Criteria:**
- [ ] Auto vs manual agreement calculated
- [ ] Agreement > 85% verified
- [ ] Per-category accuracy reported
- [ ] Quality report generated

---

### Story 12.3: Final Report
**Assignee:** U1, U2
**Priority:** High

**Description:**
As authors, we want to write the final project report.

**Acceptance Criteria:**
- [ ] Methodology section written
- [ ] Results section with statistics
- [ ] Conclusions and recommendations
- [ ] Report saved to docs/reports/

---

### Story 12.4: Presentation
**Assignee:** All
**Priority:** High

**Description:**
As presenters, we want to create the final presentation.

**Acceptance Criteria:**
- [ ] Slides created
- [ ] Demo video prepared
- [ ] Key results highlighted
- [ ] Presentation rehearsed

---

## Story Status Legend

- [ ] Not Started
- [x] Completed

---

## GitHub Labels (Suggested)

| Label | Color | Description |
|-------|-------|-------------|
| `epic` | `#6B4BA1` | Epic-level issue |
| `story` | `#0075CA` | User story |
| `task` | `#D4C5F9` | Implementation task |
| `U1-danylo-l` | `#E99695` | Assigned to U1 |
| `U2-anton` | `#BFDADC` | Assigned to U2 |
| `U3-danylo-z` | `#C2E0C6` | Assigned to U3 |
| `U4-mariia` | `#FEF2C0` | Assigned to U4 |
| `semester-1` | `#1D76DB` | Semester 1 work |
| `semester-2` | `#5319E7` | Semester 2 work |
| `high-priority` | `#D93F0B` | High priority |
| `blocked` | `#B60205` | Blocked by dependency |
