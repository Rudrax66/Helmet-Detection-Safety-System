# 🪖 Helmet Detection Safety System
> AI-powered construction site safety monitoring using **YOLOv8 + BERT + GPT-2**
> Built with Python · Streamlit · HuggingFace Transformers · Ultralytics · Pickle

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)
![BERT](https://img.shields.io/badge/BERT-HuggingFace-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎬 Demo & Video

| Type | Link |
|------|------|
| 🎥 **Video Demo** | [Watch Full Demo Video](https://github.com/Rudrax66/Helmet-Detection-Safety-System/blob/main/Helmet%20Detection%20Safety%20System.mp4) |
| 🌐 **Live Demo** | [Launch App → localhost:8501](http://localhost:8501/) |

> 📽️ Click the video link above to watch the full walkthrough of the Helmet Detection Safety System in action.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Features](#features)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [How to Use the Dashboard](#how-to-use-the-dashboard)
- [Risk Level Logic](#risk-level-logic)
- [Fine-tuning YOLO](#fine-tuning-yolo-on-custom-data)
- [Common Errors & Fixes](#common-errors--fixes)
- [Tech Stack](#tech-stack)

---

## 📖 About the Project

The **Helmet Detection Safety System** is an AI-powered safety monitoring tool designed for construction sites, factories, and industrial zones. It uses three pretrained deep learning models working together to:

1. **Detect** whether workers are wearing helmets in images or video frames
2. **Classify** the safety risk level based on detections or incident reports
3. **Generate** professional safety incident reports automatically

All models are saved as **pickle (.pkl)** files for fast reloading — no re-downloading needed after the first setup.

---

## 🏗️ Architecture

```
INPUT IMAGE / INCIDENT TEXT
        │
        ├──── MODULE 1: YOLOv8 ──────────────────────────────
        │      Pretrained on COCO → Fine-tune on helmet data
        │      Input  → Raw Image (any resolution)
        │      Output → Bounding Boxes + Labels + Confidence
        │              [helmet ✅ | no_helmet ❌]
        │      Saved  → models/yolo_helmet.pkl
        │
        ├──── MODULE 2: BERT ─────────────────────────────────
        │      bert-base-uncased → CLS embeddings (768-dim)
        │      + Sklearn LogisticRegression classifier
        │      Input  → Incident text OR auto-generated from YOLO
        │      Output → Risk Level [Low | Medium | High | Critical]
        │      Saved  → models/bert_classifier.pkl
        │
        └──── MODULE 3: GPT-2 ────────────────────────────────
               Pretrained GPT-2 text-generation pipeline
               Input  → Structured prompt (zone + counts + risk)
               Output → Full professional safety incident report
               Saved  → models/gpt_reporter.pkl

OUTPUT
    ├── Annotated image with bounding boxes
    ├── Detection counts (helmet / no-helmet)
    ├── Risk level badge (Low / Medium / High / Critical)
    ├── Alert banner
    └── Downloadable safety report (.txt)
```

---

## 📁 Project Structure

```
D:\Transformer\helmet_detection\
│
├── models\
│   ├── __init__.py              # Makes models a Python package
│   ├── yolo_model.py            # YOLOv8 fine-tune + pickle save/load
│   ├── bert_model.py            # BERT embeddings + sklearn classifier
│   └── gpt_model.py             # GPT-2 report generator
│
├── app\
│   ├── __init__.py              # Makes app a Python package
│   └── dashboard.py             # Main Streamlit UI (v2 - Real YOLO)
│
├── data\
│   └── helmet.yaml              # YOLO dataset config (auto-created)
│
├── setup_models.py              # One-time model download & pickle save
├── requirements.txt             # All Python dependencies
└── README.md                    # This file
```

---

## 🤖 Models Used

| # | Model | Source | Saved As | Role |
|---|-------|--------|----------|------|
| 1 | **YOLOv8n** | `ultralytics` (COCO pretrained) | `yolo_helmet.pkl` | Detect helmet / no_helmet in images |
| 2 | **BERT-base** | `bert-base-uncased` (HuggingFace) | `bert_classifier.pkl` | Classify risk level from incident text |
| 3 | **GPT-2** | `gpt2` (HuggingFace) | `gpt_reporter.pkl` | Generate safety incident reports |

> **Note:** No raw PyTorch used. All models accessed via HuggingFace `pipeline()` API or Ultralytics API.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📷 Real YOLO Inference | Upload any image → real helmet detection (no simulation) |
| 🧠 BERT Risk Analysis | Type incident text → classify as Low / Medium / High / Critical |
| 📝 Auto Report | Professional safety report generated from detection results |
| ⬇️ Download Report | Save report as `.txt` file with one click |
| 🚨 Alert Banners | Color-coded alerts based on risk level |
| 💾 Pickle Storage | All models stored locally — fast reload, no re-download |
| 🎛️ Adjustable Confidence | Slider to tune YOLO detection threshold |
| 🏷️ Zone Selection | Label detections by site zone (Zone-A, Zone-B, etc.) |

---

## 💻 Requirements

- **OS:** Windows 10/11, macOS, Linux
- **Python:** 3.9 – 3.11 (recommended: 3.11)
- **RAM:** 8 GB minimum, 16 GB recommended
- **GPU:** Optional (NVIDIA CUDA for faster inference)
- **Disk:** ~5 GB free (for model downloads)
- **Internet:** Required for first-time model download only

### Python Packages

```
streamlit>=1.32.0
ultralytics>=8.0.0
transformers>=4.38.0
scikit-learn>=1.4.0
pillow>=10.0.0
opencv-python>=4.9.0
numpy>=1.26.0
pandas>=2.0.0
```

---

## ⚙️ Installation & Setup

### Step 1 — Clone or Download the Project

```bash
# Create the folder structure
mkdir D:\Transformer\helmet_detection
mkdir D:\Transformer\helmet_detection\models
mkdir D:\Transformer\helmet_detection\app
mkdir D:\Transformer\helmet_detection\data
```

Place all `.py` files in their correct locations as shown in Project Structure above.

---

### Step 2 — Create Virtual Environment

```bash
cd D:\Transformer
python -m venv venv
venv\Scripts\activate
```

You will see `(venv)` appear in your terminal — this means the virtual environment is active.

---

### Step 3 — Install Dependencies

```bash
pip install streamlit ultralytics transformers scikit-learn pillow opencv-python numpy pandas
```

⏳ This takes **5–10 minutes** depending on internet speed.

---

### Step 4 — Create `__init__.py` Files

```bash
cd D:\Transformer\helmet_detection
type nul > models\__init__.py
type nul > app\__init__.py
```

These files are required — they tell Python that `models` and `app` are packages.

---

### Step 5 — Download & Save Models as Pickle

```bash
cd D:\Transformer\helmet_detection
python setup_models.py
```

This will:
- ✅ Download **YOLOv8n** weights (COCO pretrained)
- ✅ Download **BERT** (bert-base-uncased) from HuggingFace
- ✅ Download **GPT-2** from HuggingFace
- ✅ Save all 3 as `.pkl` files in the `models\` folder

⏳ **First run takes 10–20 minutes.** After that, models load instantly from pickle.

Expected output:
```
==================================================
🚀 HELMET DETECTION SYSTEM - MODEL SETUP
==================================================
[1/3] Setting up YOLO model...
   ✅ YOLO saved: models/yolo_helmet.pkl

[2/3] Setting up BERT classifier...
   ✅ BERT saved: models/bert_classifier.pkl

[3/3] Setting up GPT-2 reporter...
   ✅ GPT-2 saved: models/gpt_reporter.pkl

🎉 Setup complete! Run: streamlit run app/dashboard.py
==================================================
```

---

## 🚀 How to Run

```bash
cd D:\Transformer\helmet_detection
streamlit run app\dashboard.py
```

Your browser will automatically open at:
```
http://localhost:8501
```

---

## 🔁 Every Time You Come Back

```bash
cd D:\Transformer
venv\Scripts\activate
cd helmet_detection
streamlit run app\dashboard.py
```

> You do **NOT** need to run `setup_models.py` again — models are already saved as pickle files!

---

## 🖥️ How to Use the Dashboard

### Tab 1 — 📷 Image Detection

| Step | Action |
|------|--------|
| 1 | Select your **Site Zone** from the sidebar |
| 2 | Adjust **Confidence Threshold** if needed (default: 0.35) |
| 3 | Click **"Upload image"** and select a photo |
| 4 | Optionally type an incident description |
| 5 | Click **🔍 ANALYZE IMAGE** |
| 6 | View annotated image, detection counts, risk level, and report |
| 7 | Click **⬇ Download Report** to save the safety report |

### Tab 2 — 📝 Risk Analysis

| Step | Action |
|------|--------|
| 1 | Select a sample text or type your own incident description |
| 2 | Click **🧠 CLASSIFY RISK** |
| 3 | View risk level, confidence score, and probability bars |

### Tab 3 — 📊 System Info

Shows model file status, working directory, detection mode, and quick start commands.

---

## ⚠️ Risk Level Logic

Risk level is determined automatically from YOLO detection counts:

| Condition | Risk Level | Alert |
|-----------|-----------|-------|
| 0 persons detected OR all have helmets | 🟢 **Low** | All compliant — no action |
| 1 person without helmet (≤3 total) | 🟡 **Medium** | Verbal warning required |
| 2+ persons without helmets | 🟠 **High** | Stop work + safety briefing |
| 3+ persons without helmets | 🔴 **Critical** | Emergency halt + evacuate |

If you type incident text in the description box, **BERT** classifies the risk from the text instead.

---

## 🏋️ Fine-tuning YOLO on Custom Data

To improve detection accuracy with your own helmet dataset:

### 1. Prepare Dataset in YOLO Format

```
data/
├── train/
│   ├── images/   ← training images (.jpg)
│   └── labels/   ← YOLO format .txt labels
├── val/
│   ├── images/
│   └── labels/
└── helmet.yaml
```

### 2. helmet.yaml Format

```yaml
path: ./data
train: train/images
val: val/images

nc: 2
names:
  0: helmet
  1: no_helmet
```

### 3. Run Training

```bash
python models\yolo_model.py
```

This fine-tunes YOLOv8 and saves `best.pt`. Update `setup_models.py` to point to `best.pt` and re-run it.

### 4. Free Helmet Datasets

| Dataset | Link |
|---------|------|
| Hard Hat Universe | https://universe.roboflow.com/roboflow-universe-projects/hard-hat-universe |
| Safety Helmet Detection | https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection |

---

## 🐛 Common Errors & Fixes

### ❌ `ModuleNotFoundError: No module named 'models'`
```bash
# Always run from inside helmet_detection folder
cd D:\Transformer\helmet_detection
streamlit run app\dashboard.py
```

### ❌ `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### ❌ `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics
```

### ❌ `models/yolo_helmet.pkl not found`
```bash
python setup_models.py
```

### ❌ `streamlit: command not found`
```bash
python -m streamlit run app\dashboard.py
```

### ❌ No detections on image
- Lower the **Confidence Threshold** slider to `0.10` or `0.15`
- Use images with clear, close-up views of people
- Note: YOLOv8n is pretrained on COCO — fine-tune on helmet data for best results

### ❌ Browser doesn't open automatically
- Manually open: `http://localhost:8501`

### ❌ App running slowly
- Normal on CPU — BERT and GPT-2 are large models
- GPU (NVIDIA CUDA) will speed things up significantly

---

## 🧰 Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Object Detection | YOLOv8 (Ultralytics) | 8.4+ |
| Text Classification | BERT (HuggingFace) | bert-base-uncased |
| Report Generation | GPT-2 (HuggingFace) | gpt2 |
| ML Pipeline | Scikit-learn | 1.4+ |
| UI Framework | Streamlit | 1.32+ |
| Image Processing | Pillow + OpenCV | 10+ / 4.9+ |
| Model Storage | Pickle (.pkl) | Built-in |
| Language | Python | 3.11 |

---

## 📊 Target Performance

| Model | Metric | Target |
|-------|--------|--------|
| YOLOv8 (fine-tuned) | mAP@0.5 | > 85% |
| BERT Classifier | Accuracy | > 90% |
| System | FPS (GPU) | > 30 FPS |

---

## 🗺️ Real-World Use Cases

- 🏗️ **Construction Sites** — Monitor workers for helmet compliance
- 🏭 **Factories & Warehouses** — Automated PPE enforcement
- 🛣️ **Road Construction** — Traffic safety worker monitoring
- ⛏️ **Mining Zones** — High-risk area safety auditing
- 📹 **CCTV Integration** — Real-time video stream analysis

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 👨‍💻 Author

Built using:
- 🤗 [HuggingFace Transformers](https://huggingface.co/transformers)
- 🎯 [Ultralytics YOLOv8](https://ultralytics.com)
- 🌊 [Streamlit](https://streamlit.io)

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🎥 Video Demo | [Watch Demo](https://github.com/Rudrax66/Helmet-Detection-Safety-System/blob/main/Helmet%20Detection%20Safety%20System.mp4) |
| 🌐 Live Demo | [localhost:8501](http://localhost:8501/) |
| 📦 GitHub Repo | [Helmet-Detection-Safety-System](https://github.com/Rudrax66/Helmet-Detection-Safety-System) |

---

*Last updated: February 2026*
