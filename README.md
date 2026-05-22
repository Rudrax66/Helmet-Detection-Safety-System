# 🔍 Visual Question Answering (VQA) System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Ask any question about any image — powered by BLIP VQA + YOLOv8**

[Demo](#-demo) · [Features](#-features) · [Installation](#-installation) · [Usage](#-usage) · [Architecture](#-architecture) · [FAQ](#-faq)

</div>

---

## 📸 Demo

Upload an image, type a question, and get an instant AI-powered answer with confidence scores.

```
Image: Street scene with cars and people
Q: "How many people are in this image?"   →  A: 7  (97.0% confidence) [YOLOv8]
Q: "What is happening in this image?"     →  A: people walking  (85.0%) [BLIP]
Q: "Is this indoors or outdoors?"         →  A: outdoors  (100.0%) [BLIP]
Q: "What vehicles are present?"           →  A: motorcycle  (85.0%) [BLIP]
```

---

## ✨ Features

- 🧠 **BLIP VQA** — Salesforce's state-of-the-art vision-language model for open-ended questions
- 🎯 **YOLOv8 Counting** — Automatically routes counting questions ("how many…") to YOLOv8 for pixel-accurate object detection
- 🔀 **Smart Router** — Detects question intent and picks the best backend automatically
- 📊 **Top-K Answers** — Shows multiple answer candidates with animated confidence bars
- 🕓 **Session History** — Tracks all Q&A pairs with timestamps and exportable as `.txt`
- 🌑 **Deep Space UI** — Custom dark bioluminescent theme with Orbitron / JetBrains Mono fonts
- 💾 **Local Model Cache** — Model saved locally after first download; no re-download on restart
- ⚡ **CPU + GPU** — Works on CPU out of the box; automatically uses CUDA if available

---

## 🗂 Project Structure

```
vqa_system/
├── dashboard.py        # Streamlit app — main entry point
├── vqa_model.py        # Model load / inference helpers
├── setup_models.py     # One-time model download script
├── requirements.txt    # Python dependencies
├── models/
│   └── blip-vqa-base/  # Auto-created after setup_models.py
└── README.md
```

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/vqa-system.git
cd vqa-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` installation varies by platform. If the above is slow, install PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/) then run `pip install -r requirements.txt`.

### 4. Download the model (one-time, ~1 GB)

```bash
python setup_models.py
```

This downloads `Salesforce/blip-vqa-base` from HuggingFace and saves it to `models/blip-vqa-base/`. You only need to do this once.

> **Skip this step** if you want the app to auto-download on first launch instead.

---

## 🚀 Usage

### Launch the dashboard

```bash
streamlit run dashboard.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Use the Python API directly

```python
from vqa_model import load_vqa_model, answer_question
from PIL import Image

# Load model
model_data = load_vqa_model()

# Ask a question
image = Image.open("street.jpg")
result = answer_question(image, "What vehicles are present?", model_data)

print(result["answer"])      # → "motorcycle"
print(result["confidence"])  # → 0.85
print(result["top_answers"]) # → [{"answer": "motorcycle", "score": 0.85}, ...]
```

### Ask multiple questions about one image

```python
from vqa_model import load_vqa_model, answer_multiple_questions
from PIL import Image

model_data = load_vqa_model()
image      = Image.open("street.jpg")

questions = [
    "Is this indoors or outdoors?",
    "What time of day is it?",
    "Are there any animals?",
]

results = answer_multiple_questions(image, questions, model_data)
```

---

## 🏗 Architecture

```
INPUT IMAGE + QUESTION TEXT
        │
        ▼
  ┌─────────────────────┐
  │   Smart Router      │
  │  (question intent)  │
  └──────┬──────────────┘
         │
   counting?──YES──► YOLOv8n ──► Object Count
         │
        NO
         │
         ▼
  ┌──────────────────────────────────────┐
  │         BLIP VQA Model               │
  │   (Salesforce/blip-vqa-base)         │
  │                                      │
  │  Image ──► Vision Encoder (ViT)      │
  │  Question ──► Text Encoder (BERT)    │
  │          ──► Multimodal Fusion       │
  │          ──► Answer Decoder          │
  └──────────────────────────────────────┘
         │
         ▼
  Top-K Answers + Confidence Scores
         │
         ▼
  Streamlit Dashboard
```

### Models used

| Model | Source | Size | Used For |
|---|---|---|---|
| `blip-vqa-base` | Salesforce / HuggingFace | ~990 MB | All descriptive / yes-no / scene questions |
| `yolov8n.pt` | Ultralytics | ~6 MB | Counting questions ("how many…") |

---

## 🧩 Supported Question Types

| Category | Examples |
|---|---|
| 🔍 Object Detection | "What is the main object?" · "What animals can you see?" |
| 🌍 Scene Understanding | "Where was this taken?" · "Is this indoors or outdoors?" |
| 🎨 Color & Appearance | "What color is the car?" · "What is the person wearing?" |
| ✅ Yes / No | "Is there a person?" · "Is the sky visible?" |
| 🔢 Counting | "How many people are there?" · "How many cars?" |

---

## ⚙️ Configuration

All settings are available in the sidebar of the dashboard:

| Setting | Default | Description |
|---|---|---|
| Top-K Answers | 3 | Number of answer candidates shown with confidence bars |
| Clear History | — | Resets session Q&A history and stats |

---

## 📦 Dependencies

```
streamlit>=1.32.0
transformers>=4.38.0
Pillow>=10.0.0
numpy>=1.26.0
torch>=2.0.0
ultralytics>=8.0.0
```

---

## 🖥 System Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11+ |
| RAM | 4 GB | 8 GB+ |
| Storage | 1.5 GB (model) | 2 GB |
| GPU | Not required | CUDA GPU (faster inference) |

> On CPU, each inference takes ~2–5 seconds. On a CUDA GPU it drops to under 1 second.

---

## 🔧 Troubleshooting

**Model not loading / download fails**
```bash
# Re-run setup with verbose output
python setup_models.py
```

**`ultralytics` not found**
```bash
pip install ultralytics
```
Without it, counting questions fall back to BLIP automatically (no crash).

**`streamlit: command not found`**
```bash
pip install streamlit
python -m streamlit run dashboard.py
```

**CUDA out of memory**
The model runs on CPU by default (`device=-1`). If you force GPU and run out of memory, set `torch.cuda.empty_cache()` or use CPU.

**Port already in use**
```bash
streamlit run dashboard.py --server.port 8502
```

---

## 🗺 Roadmap

- [ ] BLIP-2 support (higher accuracy, needs GPU)
- [ ] Batch image processing
- [ ] REST API endpoint (`FastAPI`)
- [ ] Docker image
- [ ] Drag-and-drop multi-image comparison

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Salesforce BLIP](https://github.com/salesforce/BLIP) — Vision-Language pre-training
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io)

---

<div align="center">
Made with ❤️ using BLIP · YOLOv8 · Streamlit
</div>
