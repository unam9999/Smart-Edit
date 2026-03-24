# SmartSort — AI-Powered Bulk Image Categorization Tool

> **6th Semester Minor Project Workplan**
> Target: Publishable, production-quality desktop/web application

---

## 1. Problem Statement

After photoshoots, photographers and content creators end up with hundreds or thousands of images that need to be manually sorted into categories (e.g., *portraits, landscapes, food, events, products*). This is tedious, time-consuming, and error-prone. **SmartSort** automates this by using a deep learning model to classify images in bulk and organize them into category folders — saving hours of manual labor.

---

## 2. Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **ML Framework** | PyTorch + `torchvision` | Industry-standard, great pretrained models |
| **Base Model** | EfficientNet-B0 (or ResNet-50) | Excellent accuracy-to-speed ratio, transfer learning ready |
| **Backend / API** | FastAPI (Python) | Async, fast, auto-generated Swagger docs |
| **Frontend** | React (Vite + TypeScript) | Modern, fast, publishable quality |
| **Desktop (optional)** | Electron or Tauri wrapper | If you want a downloadable `.exe` |
| **Storage** | Local filesystem | Photos stay on user's machine — privacy first |
| **Deployment** | Docker + GitHub Actions CI/CD | Publishable, reproducible builds |

---

## 3. Architecture Overview

```
┌───────────────────────────────────────────────┐
│                  Frontend (React)              │
│  ┌─────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Upload  │  │ Category │  │  Results &   │  │
│  │  Panel  │  │  Config  │  │  Download    │  │
│  └────┬────┘  └────┬─────┘  └──────┬───────┘  │
│       │            │               │           │
└───────┼────────────┼───────────────┼───────────┘
        │   REST API │               │
┌───────▼────────────▼───────────────▼───────────┐
│               Backend (FastAPI)                 │
│  ┌──────────┐  ┌────────────┐  ┌────────────┐  │
│  │  Upload  │  │ Classifier │  │   File     │  │
│  │  Handler │  │  Service   │  │  Organizer │  │
│  └──────────┘  └─────┬──────┘  └────────────┘  │
│                      │                          │
│              ┌───────▼────────┐                 │
│              │  ML Model       │                 │
│              │  (EfficientNet) │                 │
│              └─────────────────┘                 │
└─────────────────────────────────────────────────┘
```

---

## 4. Detailed Phase Plan

### Phase 1 — Research & Setup (Week 1–2)

- [ ] Finalize category list (start with 10–15 common categories: *portrait, landscape, food, architecture, animal, vehicle, document, screenshot, meme, group-photo, selfie, product, nature, night-shot*)
- [ ] Set up Python virtual environment & project structure
- [ ] Set up Git repository with proper `.gitignore`, README, LICENSE
- [ ] Research pretrained models — benchmark EfficientNet-B0 vs ResNet-50 vs MobileNetV3

**Deliverable:** Working dev environment, category taxonomy document

---

### Phase 2 — Dataset Preparation (Week 2–3)

- [ ] Collect/curate dataset (sources below)
  - [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) (subset)
  - [Unsplash Dataset](https://unsplash.com/data) (real-world photos)
  - [Flickr8k / Flickr30k](https://www.kaggle.com/datasets) (via Kaggle)
  - Custom photos from your own shoots
- [ ] Clean & label data — aim for **500–1000 images per category** minimum
- [ ] Implement data augmentation pipeline (rotation, flip, color jitter, random crop)
- [ ] Create train/val/test split (70/15/15)
- [ ] Write a `DataLoader` with proper transforms

**Deliverable:** Curated dataset, augmentation pipeline, data exploration notebook

---

### Phase 3 — Model Training (Week 3–5)

- [ ] Load pretrained EfficientNet-B0 from `torchvision.models`
- [ ] Replace classifier head for your N categories
- [ ] Implement training loop with:
  - Cross-entropy loss
  - Adam optimizer with learning rate scheduling (CosineAnnealing)
  - Early stopping
  - Mixed precision training (`torch.cuda.amp`)
- [ ] Train on GPU (use Google Colab / Kaggle if no local GPU)
- [ ] Log metrics with TensorBoard or Weights & Biases
- [ ] Evaluate on test set — target **>85% accuracy**
- [ ] Export best model as `.pt` / ONNX for inference
- [ ] Generate confusion matrix, precision/recall/F1 per category

**Deliverable:** Trained model, evaluation report, training notebook

---

### Phase 4 — Backend API (Week 5–7)

- [ ] Set up FastAPI project structure:
  ```
  backend/
  ├── app/
  │   ├── main.py            # FastAPI app entry
  │   ├── routers/
  │   │   ├── classify.py     # /classify endpoint
  │   │   └── health.py       # /health endpoint
  │   ├── services/
  │   │   ├── classifier.py   # ML inference logic
  │   │   └── organizer.py    # File sorting logic
  │   ├── models/
  │   │   └── schemas.py      # Pydantic models
  │   └── utils/
  │       └── image_utils.py  # Preprocessing helpers
  ├── model/                  # Trained .pt files
  ├── tests/
  └── requirements.txt
  ```
- [ ] Implement `/classify` endpoint (accepts batch of images, returns predictions)
- [ ] Implement `/organize` endpoint (moves/copies files into category folders)
- [ ] Add confidence threshold — only auto-sort if confidence > 70%, else flag as "uncertain"
- [ ] Add progress tracking via WebSocket or SSE for bulk operations
- [ ] Write unit tests for classifier service
- [ ] Write integration tests for API endpoints

**Deliverable:** Working REST API with Swagger docs

---

### Phase 5 — Frontend (Week 7–9)

- [ ] Initialize Vite + React + TypeScript project
- [ ] Design & implement UI pages:
  - **Home / Upload Page** — drag-and-drop bulk upload zone
  - **Category Config Page** — user can select/customize target categories
  - **Processing Page** — live progress bar with image thumbnails being classified
  - **Results Dashboard** — grid view of sorted images, confidence scores, category breakdown chart
  - **Download / Export Page** — download sorted folders as ZIP
- [ ] Implement API integration with `axios` or `fetch`
- [ ] Add dark mode, responsive design, smooth animations
- [ ] Add image preview modals with ability to manually re-categorize misclassified images

**Deliverable:** Polished, responsive web frontend

---

### Phase 6 — Integration & Testing (Week 9–10)

- [ ] End-to-end testing: upload 100+ images → classify → verify sorting accuracy
- [ ] Performance benchmarks:
  - Inference time per image
  - Batch processing throughput
  - Memory usage
- [ ] Edge case handling:
  - Corrupted images
  - Non-image files mixed in
  - Very large images (>20MB)
  - Unsupported formats
- [ ] User acceptance testing with real photoshoot data

**Deliverable:** Test report, performance benchmarks

---

### Phase 7 — Documentation & Publishing (Week 10–12)

- [ ] Write comprehensive README with:
  - Project overview, screenshots, demo GIF
  - Installation instructions
  - Usage guide
  - API documentation
- [ ] Add Dockerfile for one-click setup
- [ ] Create a project poster / presentation for college submission
- [ ] Optional: Write a short technical paper (IEEE format) covering:
  - Problem statement
  - Literature survey
  - Methodology (transfer learning approach)
  - Results & discussion
  - Future scope
- [ ] Publish to GitHub with proper tags, releases, and GitHub Pages demo

**Deliverable:** Published repository, documentation, presentation

---

## 5. Project Structure (Full)

```
SmartSort/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/
│   │   ├── services/
│   │   ├── models/
│   │   └── utils/
│   ├── model/
│   │   └── efficientnet_smartsort.pt
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── public/
│   ├── package.json
│   └── Dockerfile
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── augmented/
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── poster.pptx
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── README.md
├── LICENSE
└── .gitignore
```

---

## 6. Key Features for "Publishable" Quality

| Feature | Why It Matters |
|---|---|
| **Transfer Learning** | Shows you understand modern ML, not training from scratch |
| **Confidence Scores** | Transparency — user knows when model is unsure |
| **Manual Override** | Practical UX — user can fix mistakes |
| **Batch Processing** | Core value proposition — handles 1000s of images |
| **Progress Tracking** | Real-time feedback during processing |
| **Dark Mode + Polished UI** | Professional look for portfolio/publication |
| **Docker Support** | One-command setup, reproducibility |
| **API Documentation** | Swagger/OpenAPI auto-generated |
| **Test Coverage** | Shows engineering rigor |
| **CI/CD Pipeline** | Industry best practice |

---

## 7. Stretch Goals (if time permits)

- [ ] **Custom Category Training** — let user upload labeled samples to create custom categories
- [ ] **EXIF-based Sorting** — combine ML predictions with metadata (date, GPS, camera model)
- [ ] **Duplicate Detection** — find and flag near-duplicate images using perceptual hashing
- [ ] **Face Clustering** — group photos by person using face embeddings
- [ ] **Desktop App** — wrap in Tauri/Electron for native `.exe`
- [ ] **Cloud Deployment** — deploy on AWS/GCP with a public demo URL

---

## 8. Verification Plan

### Automated Tests
- **Backend unit tests:** `pytest backend/tests/ -v` — classifier service, image utils, organizer
- **Backend API tests:** `pytest backend/tests/test_api.py -v` — endpoint integration tests
- **Frontend tests:** `npm test` — component rendering, API integration mocking
- **ML evaluation:** Run `notebooks/03_evaluation.ipynb` — confusion matrix, per-class F1

### Manual Verification
- Upload a batch of 50–100 mixed images through the UI → verify correct categorization
- Test with edge cases (corrupted files, non-images, huge files)
- Check that the "uncertain" bucket works for low-confidence predictions
- Verify ZIP download contains correctly organized folder structure

---

## 9. Timeline Summary

| Week | Phase | Key Milestone |
|---|---|---|
| 1–2 | Research & Setup | Environment ready, categories finalized |
| 2–3 | Dataset Preparation | Clean dataset with augmentation |
| 3–5 | Model Training | Trained model with >85% accuracy |
| 5–7 | Backend API | Working classification API |
| 7–9 | Frontend | Polished web UI |
| 9–10 | Integration & Testing | End-to-end validated |
| 10–12 | Documentation & Publishing | GitHub published, presentation ready |

---

> [!TIP]
> **Start with the model first.** If the ML side works well, the rest is "just" engineering. Use Google Colab for free GPU access during training.

> [!IMPORTANT]
> **For publishability**, make sure your README has: clear screenshots, a demo GIF, installation steps, and a "How It Works" section. First impressions matter on GitHub.
