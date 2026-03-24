# VisionSort — AI-Powered Image Categorization & Enhancement System

## Overview

VisionSort is an AI-powered system designed to automate image organization and improve image quality in a single workflow. The system classifies images into meaningful categories and applies intelligent enhancement techniques to improve visual quality.

It is built to handle large batches of images efficiently, reducing manual effort in both sorting and editing tasks.

---

## Problem Statement

Managing and organizing large collections of images is time-consuming and inefficient when done manually. Additionally, basic image improvements such as brightness, contrast, and sharpness adjustments require separate tools, increasing the overall workload.

---

## Proposed Solution

VisionSort combines deep learning-based image classification with image processing techniques to create a unified solution that:

* Automatically categorizes images
* Enhances image quality using adjustable parameters
* Supports bulk processing of images
* Provides confidence-based classification

---

## Key Features

### 🔹 AI-Based Image Classification

* Uses transfer learning with EfficientNet
* Classifies images into categories such as:

  * Portrait
  * Group Photo
  * Landscape
  * Food
  * Event
  * Product
* Provides confidence scores for each prediction

---

### 🔹 Image Enhancement Module

* Brightness adjustment
* Contrast enhancement
* Saturation control
* Sharpness improvement
* Warmth adjustment
* Skin smoothing (face-aware enhancement)

---

### 🔹 Bulk Processing

* Supports large image batches
* Automatically sorts images into category folders

---

### 🔹 Confidence-Based Sorting

* High-confidence images are auto-sorted
* Low-confidence images are marked as **“uncertain”**

---

### 🔹 User Interaction (Frontend)

* Upload images in bulk
* View classification results
* Apply enhancement adjustments
* Preview and download processed images

---

## System Architecture

```text
User Interface (React)
        ↓
Backend API (FastAPI)
        ↓
 ┌─────────────────────────────┐
 │   Processing Layer          │
 │  ┌──────────────┐           │
 │  │ Enhancement  │ (OpenCV)  │
 │  └──────┬───────┘           │
 │         ↓                   │
 │  ┌──────────────┐           │
 │  │ Classification│ (PyTorch)│
 │  └──────────────┘           │
 └─────────────────────────────┘
        ↓
Organized & Enhanced Images
```

---

## Tech Stack

### Machine Learning

* PyTorch
* Torchvision
* EfficientNet (Transfer Learning)

### Image Processing

* OpenCV
* Pillow
* NumPy

### Backend

* FastAPI

### Frontend

* React (TypeScript)

### Deployment

* Docker
* GitHub CI/CD

---

## Workflow

1. User uploads images
2. Images are preprocessed
3. Enhancement operations are optionally applied
4. Images are passed through the trained model
5. Model predicts category with confidence score
6. Images are sorted into category folders
7. Results are displayed to the user

---

## Expected Outcome

* Significant reduction in manual image sorting time
* Improved image quality through automated enhancements
* Efficient handling of large image datasets

---

## Future Scope

* Custom category training by users
* Face-based grouping and clustering
* Duplicate image detection
* Cloud deployment for remote access
* Advanced AI-based enhancement (super-resolution, denoising)

---

## Author

Minor Project — 6th Semester
