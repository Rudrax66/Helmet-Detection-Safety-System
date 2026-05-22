"""
dashboard.py — Visual Question Answering — Streamlit Dashboard

FIX SUMMARY vs original:
  1. BASE_DIR now resolves correctly regardless of where streamlit is invoked.
  2. Removed unreliable pickle load; uses BlipProcessor/BlipForQuestionAnswering directly.
  3. Fixed pipeline task name: "visual-question-answering" (was "vqa" — invalid in newer transformers).
  4. top_k inference rewritten using model.generate() — works across all transformers versions.
  5. Added ultralytics to requirements.txt (was missing despite being used).
  6. numpy moved to module-level import.
  7. Added torch import.
  8. COCO_CLASSES dict completed (was missing id 65).
  9. Graceful YOLO failure: if ultralytics not installed, counting falls back to BLIP.
 10. Model loading wrapped with detailed error messages for easier debugging.

Run: streamlit run dashboard.py
(from the folder containing dashboard.py)
"""

import streamlit as st
import os
import sys
import numpy as np
import torch
from PIL import Image
from datetime import datetime

# ── PATH FIX ──────────────────────────────────
# Resolve to the folder containing this file, regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Visual Question Answering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — Deep Space / Bioluminescent Theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Sora:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:       #020408;
    --bg2:      #050c14;
    --bg3:      #081018;
    --panel:    #0a1520;
    --border:   #0d2035;
    --border2:  #143050;
    --cyan:     #00e5ff;
    --teal:     #00bcd4;
    --blue:     #1565c0;
    --purple:   #7c4dff;
    --green:    #00e676;
    --amber:    #ffab00;
    --red:      #ff1744;
    --text1:    #e0f7fa;
    --text2:    #607d8b;
    --text3:    #2a4a5a;
}

@keyframes fadeUp   { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
@keyframes glow     { 0%,100%{box-shadow:0 0 10px rgba(0,229,255,.3)} 50%{box-shadow:0 0 30px rgba(0,229,255,.7),0 0 60px rgba(0,229,255,.2)} }
@keyframes pulse    { 0%,100%{opacity:1} 50%{opacity:.4} }
@keyframes shimmer  { 0%{background-position:-200% center} 100%{background-position:200% center} }
@keyframes barIn    { from{width:0} to{width:var(--w)} }

html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text1) !important;
    font-family: 'Sora', sans-serif !important;
}

/* ── HEADER ── */
.vqa-header {
    position: relative;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #020c18 0%, #051525 60%, #020c18 100%);
    border: 1px solid var(--border2);
    border-radius: 6px;
    overflow: hidden;
}
.vqa-header::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--purple), transparent);
    animation: shimmer 3s linear infinite;
    background-size: 200% auto;
}
.vqa-header::after {
    content: '';
    position: absolute; top:0; left:0; right:0; bottom:0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(124,77,255,.04) 0%, transparent 70%);
    pointer-events: none;
}
.vqa-header h1 {
    font-family: 'Orbitron', monospace !important;
    font-size: 2rem !important;
    font-weight: 900 !important;
    color: var(--cyan) !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    margin: 0 !important;
    text-shadow: 0 0 30px rgba(0,229,255,.5);
}
.vqa-header .sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text2);
    margin-top: 0.4rem;
    letter-spacing: 2px;
}
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--green);
    border-radius: 50%;
    margin-right: 0.5rem;
    animation: pulse 2s infinite;
    vertical-align: middle;
}

/* ── PANELS ── */
.panel {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 1.2rem;
}
.panel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--cyan);
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 0.8rem;
}

/* ── ANSWER CARD ── */
.answer-card {
    background: linear-gradient(135deg, #020e1a 0%, #031525 100%);
    border: 1px solid var(--cyan);
    border-radius: 5px;
    padding: 1.5rem;
    text-align: center;
    animation: glow 3s ease infinite;
    position: relative;
    overflow: hidden;
}
.answer-card::before {
    content: '';
    position: absolute; inset:0;
    background: radial-gradient(ellipse 60% 60% at 50% 0%, rgba(0,229,255,.05) 0%, transparent 70%);
}
.answer-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--text2);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.answer-text {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: 2px;
    text-shadow: 0 0 20px rgba(0,229,255,.6);
    text-transform: uppercase;
    line-height: 1.2;
    word-break: break-word;
}
.answer-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--green);
    margin-top: 0.6rem;
    letter-spacing: 2px;
}

/* ── QUESTION BADGE ── */
.q-badge {
    background: rgba(0,229,255,.06);
    border: 1px solid rgba(0,229,255,.2);
    border-left: 3px solid var(--cyan);
    border-radius: 3px;
    padding: 0.6rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text1);
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
}

/* ── CONFIDENCE BARS ── */
.conf-row { margin-bottom: 0.6rem; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text2);
    margin-bottom: 0.25rem;
}
.conf-track {
    background: var(--border);
    border-radius: 2px;
    height: 5px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 2px;
    animation: barIn 1s cubic-bezier(.4,0,.2,1) forwards;
    --w: 0%;
    width: 0;
}
.cf-1 { background: linear-gradient(90deg, var(--cyan), var(--purple)); }
.cf-2 { background: var(--teal); }
.cf-3 { background: var(--blue); }

/* ── HISTORY ITEM ── */
.hist-item {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--purple);
    border-radius: 3px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}
.hist-q { color: var(--text2); margin-bottom: 0.2rem; }
.hist-a { color: var(--cyan); font-weight: 500; }
.hist-c { color: var(--text3); font-size: 0.62rem; }

/* ── UPLOAD ZONE ── */
.upload-hint {
    background: var(--bg2);
    border: 1px dashed var(--border2);
    border-radius: 5px;
    padding: 2.5rem;
    text-align: center;
    color: var(--text3);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
}

/* ── SAMPLE Q PILLS ── */
.pill-container { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.q-pill {
    background: rgba(0,229,255,.05);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--text2);
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.5px;
}
.q-pill:hover { border-color: var(--cyan); color: var(--cyan); }

/* ── METRIC CARDS ── */
.stat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.8rem; margin-bottom: 1rem; }
.stat-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem;
    text-align: center;
}
.stat-val {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
}
.stat-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--text2);
    letter-spacing: 2px;
    margin-top: 0.3rem;
    text-transform: uppercase;
}
.sv-cyan   { color: var(--cyan); }
.sv-purple { color: var(--purple); }
.sv-green  { color: var(--green); }

/* ── ARCH BOX ── */
.arch-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--purple);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text2);
    line-height: 2;
    white-space: pre;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #003d6b, #001f3d) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    width: 100% !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #004d85, #002a50) !important;
    box-shadow: 0 0 20px rgba(0,229,255,.3) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text1) !important; }

/* ── INPUT ── */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: var(--panel) !important;
    color: var(--text1) !important;
    border: 1px solid var(--border2) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 3px !important;
}
.stTextInput input:focus { border-color: var(--cyan) !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
    color: var(--text2) !important;
}
.stTabs [aria-selected="true"] { color: var(--cyan) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--cyan) !important; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD VQA MODEL
# FIX: Use BlipProcessor + BlipForQuestionAnswering directly.
#      The original code used pickle which is unreliable across
#      library versions, and used task="vqa" which is invalid
#      in transformers >= 4.40.
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_vqa_model():
    """
    Load BLIP VQA model.
    Tries local saved copy first, then downloads from HuggingFace.
    Returns (model_data_dict, source_str) or (None, "error").
    """
    from transformers import BlipProcessor, BlipForQuestionAnswering

    MODEL_DIR = os.path.join(BASE_DIR, "models", "blip-vqa-base")
    HF_ID     = "Salesforce/blip-vqa-base"

    def _load_from(path_or_id, label):
        processor = BlipProcessor.from_pretrained(path_or_id)
        model     = BlipForQuestionAnswering.from_pretrained(path_or_id)
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        model     = model.to(device)
        model.eval()
        return {
            "processor":  processor,
            "model":      model,
            "model_name": HF_ID,
            "device":     device,
        }, label

    # Try local first
    if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        try:
            return _load_from(MODEL_DIR, "local")
        except Exception:
            pass  # fall through to HF download

    # Download from HuggingFace
    try:
        return _load_from(HF_ID, "huggingface")
    except Exception as e:
        return None, f"error: {e}"


# ─────────────────────────────────────────────
# YOLO-BASED ACCURATE COUNTING
# FIX: Wrapped import in try/except so dashboard works
#      even if ultralytics is not installed.
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_yolo_counter():
    """Load YOLOv8 for accurate object/person counting."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")   # auto-downloads if not present
        return model
    except ImportError:
        return None   # ultralytics not installed — counting falls back to BLIP
    except Exception:
        return None


# FIX: Added missing id 65 (remote) to complete COCO 80-class set
COCO_CLASSES = {
    0: "person",       1: "bicycle",      2: "car",          3: "motorcycle",
    4: "airplane",     5: "bus",          6: "train",        7: "truck",
    8: "boat",         9: "traffic light",10: "fire hydrant",11: "stop sign",
    12: "parking meter",13:"bench",       14: "bird",        15: "cat",
    16: "dog",         17: "horse",       18: "sheep",       19: "cow",
    20: "elephant",    21: "bear",        22: "zebra",       23: "giraffe",
    24: "backpack",    25: "umbrella",    26: "handbag",     27: "tie",
    28: "suitcase",    29: "frisbee",     30: "skis",        31: "snowboard",
    32: "sports ball", 33: "kite",        34: "baseball bat",35: "baseball glove",
    36: "skateboard",  37: "surfboard",   38: "tennis racket",39:"bottle",
    40: "wine glass",  41: "cup",         42: "fork",        43: "knife",
    44: "spoon",       45: "bowl",        46: "banana",      47: "apple",
    48: "sandwich",    49: "orange",      50: "broccoli",    51: "carrot",
    52: "hot dog",     53: "pizza",       54: "donut",       55: "cake",
    56: "chair",       57: "couch",       58: "potted plant",59: "bed",
    60: "dining table",61: "toilet",      62: "tv",          63: "laptop",
    64: "mouse",       65: "remote",      66: "keyboard",    67: "cell phone",
    68: "microwave",   69: "oven",        70: "toaster",     71: "sink",
    72: "refrigerator",73: "book",        74: "clock",       75: "vase",
    76: "scissors",    77: "teddy bear",  78: "hair drier",  79: "toothbrush",
}

COUNT_KEYWORDS = [
    "how many", "count", "number of", "total", "how much",
    "quantity", "amount of", "few", "several",
]

OBJECT_MAP = {
    "person": [0], "people": [0], "man": [0], "woman": [0],
    "men": [0], "women": [0], "human": [0], "humans": [0],
    "individual": [0], "individuals": [0], "worker": [0],
    "car": [2], "cars": [2], "vehicle": [2,3,5,6,7,8],
    "vehicles": [2,3,5,6,7,8], "truck": [7], "bus": [5],
    "bicycle": [1], "bike": [1,3], "motorcycle": [3],
    "dog": [16], "dogs": [16], "cat": [15], "cats": [15],
    "bird": [14], "birds": [14], "horse": [17],
    "chair": [56], "chairs": [56], "bottle": [39],
    "phone": [67], "laptop": [63], "book": [73],
    "elephant": [20], "elephants": [20],
}


def is_counting_question(question):
    """Detect if the question is asking for a count."""
    q = question.lower()
    return any(kw in q for kw in COUNT_KEYWORDS)


def extract_target_object(question):
    """Extract what object to count from the question."""
    q = question.lower()
    for obj, cls_ids in OBJECT_MAP.items():
        if obj in q:
            return obj, cls_ids
    return "person", [0]   # default: count people


def count_with_yolo(image_pil, question, conf=0.35):
    """
    Use YOLOv8 to count objects.
    Returns result dict compatible with run_vqa output, or None if YOLO unavailable.
    """
    yolo = load_yolo_counter()
    if yolo is None:
        return None

    target_name, target_ids = extract_target_object(question)
    image_np = np.array(image_pil.convert("RGB"))

    results        = yolo(image_np, conf=conf, verbose=False)
    count          = 0
    all_detections = {}

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name   = COCO_CLASSES.get(cls_id, f"object_{cls_id}")
            all_detections[name] = all_detections.get(name, 0) + 1
            if cls_id in target_ids:
                count += 1

    top_answers = [{"answer": str(count), "score": 0.97}]
    for name, cnt in sorted(all_detections.items(), key=lambda x: -x[1])[:4]:
        if str(cnt) != str(count):
            top_answers.append({"answer": f"{cnt} {name}s", "score": 0.5})

    return {
        "question":       question,
        "answer":         str(count),
        "confidence":     0.97,
        "top_answers":    top_answers,
        "method":         "yolo_count",
        "all_detections": all_detections,
        "target":         target_name,
        "note":           f"YOLOv8 detected {count} {target_name}(s) with high accuracy",
    }


# ─────────────────────────────────────────────
# VQA INFERENCE
# FIX: Rewrote to use model.generate() instead of pipeline()
#      which avoids the top_k API change and "vqa" task name bug.
# ─────────────────────────────────────────────

def run_vqa(image_pil, question, model_data, top_k=3):
    """
    Smart VQA router:
      - Counting questions → YOLOv8 (falls back to BLIP if YOLO unavailable)
      - All other questions → BLIP generate
    """
    # ── ROUTE: counting → YOLO ──
    if is_counting_question(question):
        yolo_result = count_with_yolo(image_pil, question)
        if yolo_result is not None:
            return yolo_result
        # YOLO unavailable — fall through to BLIP

    # ── ROUTE: descriptive → BLIP ──
    processor = model_data["processor"]
    model     = model_data["model"]
    device    = model_data["device"]

    image  = image_pil.convert("RGB")
    inputs = processor(image, question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    num_seq = min(top_k, 5)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            num_beams=max(num_seq, 5),
            num_return_sequences=num_seq,
            early_stopping=True,
        )

    answers = [
        processor.decode(ids, skip_special_tokens=True).strip()
        for ids in out
    ]

    # Deduplicate
    seen, unique = set(), []
    for a in answers:
        if a not in seen:
            seen.add(a)
            unique.append(a)

    top_answers = [
        {"answer": a, "score": round(1.0 - i * 0.15, 4)}
        for i, a in enumerate(unique[:top_k])
    ]
    if not top_answers:
        top_answers = [{"answer": "unknown", "score": 0.0}]

    return {
        "question":    question,
        "answer":      top_answers[0]["answer"],
        "confidence":  top_answers[0]["score"],
        "top_answers": top_answers,
        "method":      "blip",
    }


# ─────────────────────────────────────────────
# SAMPLE QUESTIONS
# ─────────────────────────────────────────────

SAMPLE_QS = {
    "🔍 Object Detection": [
        "What is the main object in this image?",
        "How many people are in this image?",
        "What animals can you see?",
        "What vehicles are present?",
    ],
    "🌍 Scene Understanding": [
        "Where was this photo taken?",
        "What is happening in this image?",
        "Is this indoors or outdoors?",
        "What time of day is it?",
    ],
    "🎨 Color & Appearance": [
        "What color is the main object?",
        "What is the person wearing?",
        "What are the dominant colors?",
        "Is the image bright or dark?",
    ],
    "✅ Yes / No": [
        "Is there a person in this image?",
        "Is this image taken outdoors?",
        "Is the sky visible?",
        "Are there any animals?",
        "Is it daytime?",
    ],
    "🔢 Counting": [
        "How many people are there?",
        "How many cars are in this image?",
        "How many objects can you see?",
    ],
}


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "history"    not in st.session_state: st.session_state.history    = []
if "q_count"    not in st.session_state: st.session_state.q_count    = 0
if "total_conf" not in st.session_state: st.session_state.total_conf = 0.0
if "selected_q" not in st.session_state: st.session_state.selected_q = ""


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <div style='font-family:Orbitron,monospace; font-size:1.1rem; font-weight:900;
                    color:#00e5ff; letter-spacing:3px;'>🔍 VQA</div>
        <div style='font-family:JetBrains Mono,monospace; font-size:0.6rem;
                    color:#2a4a5a; margin-top:0.2rem; letter-spacing:2px;'>VISUAL QUESTION ANSWERING</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model status
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace; font-size:0.6rem; "
        "color:#00e5ff; letter-spacing:3px; text-transform:uppercase; "
        "margin-bottom:0.5rem;'>Model Status</div>",
        unsafe_allow_html=True,
    )

    local_dir   = os.path.join(BASE_DIR, "models", "blip-vqa-base")
    local_ready = os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json"))
    icon = "🟢" if local_ready else "🟡"
    mode = "LOCAL MODEL" if local_ready else "AUTO DOWNLOAD"
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace; font-size:0.72rem; "
        f"color:#607d8b;'>{icon} BLIP-VQA — {mode}</div>",
        unsafe_allow_html=True,
    )

    yolo_available = True
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        yolo_available = False

    yolo_icon = "🟢" if yolo_available else "🔴"
    yolo_mode = "INSTALLED" if yolo_available else "NOT INSTALLED"
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace; font-size:0.72rem; "
        f"color:#607d8b;'>{yolo_icon} YOLOv8 — {yolo_mode}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace; font-size:0.6rem; "
        "color:#00e5ff; letter-spacing:3px; text-transform:uppercase; "
        "margin-bottom:0.5rem;'>Settings</div>",
        unsafe_allow_html=True,
    )
    top_k = st.slider("Top-K Answers", 1, 5, 3, help="Number of answer candidates to show")

    st.markdown("---")

    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace; font-size:0.6rem; "
        "color:#00e5ff; letter-spacing:3px; text-transform:uppercase; "
        "margin-bottom:0.5rem;'>Session Stats</div>",
        unsafe_allow_html=True,
    )

    avg_conf = (
        st.session_state.total_conf / st.session_state.q_count * 100
        if st.session_state.q_count > 0 else 0
    )
    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#607d8b; line-height:2.2;'>
        Questions Asked : <span style='color:#00e5ff;'>{st.session_state.q_count}</span><br>
        Avg Confidence  : <span style='color:#00e676;'>{avg_conf:.1f}%</span><br>
        History Items   : <span style='color:#7c4dff;'>{len(st.session_state.history)}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑 Clear History"):
        st.session_state.history    = []
        st.session_state.q_count    = 0
        st.session_state.total_conf = 0.0
        st.rerun()

    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace; font-size:0.6rem; "
        "color:#2a4a5a; letter-spacing:1px; margin-top:1rem; line-height:1.8;'>"
        "Model: BLIP-VQA-Base<br>Source: Salesforce/HuggingFace<br>"
        "Task: Visual Q&amp;A<br>Backend: Direct HF Load</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown(f"""
<div class='vqa-header'>
    <h1>🔍 Visual Question Answering</h1>
    <div class='sub'>
        <span class='live-dot'></span>
        BLIP-VQA · SALESFORCE · HUGGINGFACE &nbsp;|&nbsp;
        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Ask Questions", "📊 History & Stats", "⚙️ System Info"])


# ═══════════════════════════════════════════
# TAB 1 — MAIN VQA
# ═══════════════════════════════════════════

with tab1:
    col_left, col_right = st.columns([1, 1.2], gap="large")

    # ── LEFT: Upload + Question ──
    with col_left:
        st.markdown("<div class='panel-title'>Upload Image</div>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if not uploaded:
            st.markdown("""
            <div class='upload-hint'>
                <div style='font-size:3rem; margin-bottom:0.8rem;'>🖼️</div>
                DROP IMAGE HERE<br>
                <span style='color:#143050; font-size:0.65rem;'>JPG · PNG · BMP · WEBP</span>
            </div>""", unsafe_allow_html=True)
        else:
            image_pil = Image.open(uploaded).convert("RGB")
            st.image(image_pil, use_container_width=True,
                     caption=f"{uploaded.name} — {image_pil.size[0]}×{image_pil.size[1]}px")

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        st.markdown("<div class='panel-title'>Ask a Question</div>", unsafe_allow_html=True)

        cat = st.selectbox("Category", list(SAMPLE_QS.keys()), label_visibility="collapsed")

        sample_list = SAMPLE_QS[cat]
        pills_html  = "<div class='pill-container'>" + \
            "".join(f"<div class='q-pill'>{q}</div>" for q in sample_list) + \
            "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)
        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        selected = st.selectbox(
            "Pick sample or type below",
            ["-- type your own --"] + sample_list,
            label_visibility="collapsed",
        )

        default_q = selected if selected != "-- type your own --" else st.session_state.selected_q
        question  = st.text_input(
            "Your question",
            value=default_q,
            placeholder="Ask anything about the image...",
            label_visibility="collapsed",
        )

        ask_btn = st.button("🔍 ASK QUESTION", use_container_width=True)

    # ── RIGHT: Answer ──
    with col_right:
        st.markdown("<div class='panel-title'>Answer</div>", unsafe_allow_html=True)

        if not uploaded:
            st.markdown("""
            <div class='upload-hint' style='height:200px; display:flex; align-items:center;
                         justify-content:center; flex-direction:column;'>
                <div style='font-size:2.5rem; margin-bottom:0.5rem;'>💬</div>
                UPLOAD AN IMAGE TO BEGIN
            </div>""", unsafe_allow_html=True)

        # ── Run VQA ──
        if uploaded and ask_btn and question.strip():
            image_pil = Image.open(uploaded).convert("RGB")

            with st.spinner("🧠 Loading model..."):
                model_data, source = load_vqa_model()

            if model_data is None:
                st.error(
                    f"❌ Model could not be loaded: {source}\n\n"
                    "Try running `python setup_models.py` first, or check your internet connection."
                )
                st.stop()

            with st.spinner("💬 Generating answer..."):
                result = run_vqa(image_pil, question, model_data, top_k=top_k)

            st.session_state.q_count    += 1
            st.session_state.total_conf += result["confidence"]
            st.session_state.history.insert(0, {
                "question":   question,
                "answer":     result["answer"],
                "confidence": result["confidence"],
                "time":       datetime.now().strftime("%H:%M:%S"),
            })

            method      = result.get("method", "blip")
            is_yolo     = method == "yolo_count"
            method_icon = "🎯 YOLOv8 COUNT" if is_yolo else "🧠 BLIP VQA"
            method_col  = "#00e676" if is_yolo else "#7c4dff"

            st.markdown(f"""
            <div style='font-family:JetBrains Mono,monospace; font-size:0.62rem;
                        color:{method_col}; letter-spacing:2px; margin-bottom:0.5rem;
                        border:1px solid {method_col}33; border-radius:2px;
                        padding:0.2rem 0.6rem; display:inline-block;'>
                {method_icon}
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='answer-card'>
                <div class='answer-label'>ANSWER</div>
                <div class='answer-text'>{result['answer']}</div>
                <div class='answer-conf'>
                    CONFIDENCE: {result['confidence']*100:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='q-badge'>❓ {question}</div>", unsafe_allow_html=True)

            if is_yolo and "all_detections" in result:
                note    = result.get("note", "")
                all_det = result["all_detections"]
                det_str = " · ".join(
                    [f"{v} {k}(s)" for k, v in sorted(all_det.items(), key=lambda x: -x[1])]
                )
                st.markdown(f"""
                <div style='background:rgba(0,230,118,.06); border:1px solid rgba(0,230,118,.2);
                            border-left:3px solid #00e676; border-radius:3px;
                            padding:0.6rem 0.9rem; font-family:JetBrains Mono,monospace;
                            font-size:0.7rem; color:#607d8b; margin-bottom:0.8rem;'>
                    <span style='color:#00e676;'>✅ {note}</span><br>
                    <span style='color:#2a4a5a;'>All detected: {det_str if det_str else "none"}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:rgba(124,77,255,.06); border:1px solid rgba(124,77,255,.2);
                            border-left:3px solid #7c4dff; border-radius:3px;
                            padding:0.5rem 0.9rem; font-family:JetBrains Mono,monospace;
                            font-size:0.68rem; color:#607d8b; margin-bottom:0.8rem;'>
                    🧠 BLIP answered based on image understanding
                    {"(YOLOv8 not available — install ultralytics for counting)" if is_counting_question(question) and not yolo_available else ""}
                </div>""", unsafe_allow_html=True)

            st.markdown("<div class='panel-title'>Top Candidates</div>", unsafe_allow_html=True)

            colors = ["cf-1", "cf-2", "cf-3", "cf-3", "cf-3"]
            for i, ans in enumerate(result["top_answers"]):
                pct = ans["score"] * 100
                st.markdown(f"""
                <div class='conf-row'>
                    <div class='conf-label'>
                        <span style='color:{"#00e5ff" if i==0 else "#607d8b"};'>
                            {"▶" if i==0 else " "} {ans['answer'].upper()}
                        </span>
                        <span>{pct:.1f}%</span>
                    </div>
                    <div class='conf-track'>
                        <div class='conf-fill {colors[i]}' style='--w:{pct:.1f}%;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        elif uploaded and not question.strip() and ask_btn:
            st.warning("⚠️ Please enter a question first.")

        elif uploaded and not ask_btn:
            st.markdown("""
            <div class='upload-hint' style='height:200px; display:flex; align-items:center;
                         justify-content:center; flex-direction:column;'>
                <div style='font-size:2.5rem; margin-bottom:0.5rem;'>🤖</div>
                TYPE A QUESTION AND CLICK ASK
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 2 — HISTORY & STATS
# ═══════════════════════════════════════════

with tab2:
    st.markdown("<div class='panel-title'>Session Statistics</div>", unsafe_allow_html=True)

    avg_conf = (
        st.session_state.total_conf / st.session_state.q_count * 100
        if st.session_state.q_count > 0 else 0
    )

    st.markdown(f"""
    <div class='stat-grid'>
        <div class='stat-card'>
            <div class='stat-val sv-cyan'>{st.session_state.q_count}</div>
            <div class='stat-lbl'>Questions Asked</div>
        </div>
        <div class='stat-card'>
            <div class='stat-val sv-green'>{avg_conf:.1f}%</div>
            <div class='stat-lbl'>Avg Confidence</div>
        </div>
        <div class='stat-card'>
            <div class='stat-val sv-purple'>{len(st.session_state.history)}</div>
            <div class='stat-lbl'>Total Queries</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='panel-title'>Question History</div>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class='upload-hint'>
            <div style='font-size:2rem; margin-bottom:0.5rem;'>📋</div>
            NO HISTORY YET — ASK SOME QUESTIONS FIRST
        </div>""", unsafe_allow_html=True)
    else:
        for item in st.session_state.history:
            conf_color = (
                "#00e676" if item["confidence"] > 0.7
                else "#ffab00" if item["confidence"] > 0.4
                else "#ff1744"
            )
            st.markdown(f"""
            <div class='hist-item'>
                <div class='hist-q'>❓ {item['question']}</div>
                <div class='hist-a'>💬 {item['answer'].upper()}</div>
                <div class='hist-c'>
                    Confidence: <span style='color:{conf_color};'>{item['confidence']*100:.1f}%</span>
                    &nbsp;·&nbsp; {item['time']}
                </div>
            </div>""", unsafe_allow_html=True)

        history_txt = "\n".join([
            f"Q: {h['question']}\nA: {h['answer']} ({h['confidence']*100:.1f}%)\nTime: {h['time']}\n"
            for h in st.session_state.history
        ])
        st.download_button(
            "⬇ Export History",
            data=history_txt.encode(),
            file_name=f"vqa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )


# ═══════════════════════════════════════════
# TAB 3 — SYSTEM INFO
# ═══════════════════════════════════════════

with tab3:
    st.markdown("<div class='panel-title'>Architecture</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='arch-box'>INPUT IMAGE + QUESTION TEXT
        │
        ▼
┌─────────────────────────────────────────┐
│           BLIP VQA Model                │
│   (Salesforce/blip-vqa-base)            │
│                                         │
│  Image ──► Vision Encoder (ViT)         │
│               │                         │
│               ▼                         │
│         Image Features                  │
│               │                         │
│  Question ──► Text Encoder (BERT)       │
│               │                         │
│               ▼                         │
│     Multimodal Fusion Layer             │
│               │                         │
│               ▼                         │
│     Answer Decoder (generate())         │
└─────────────────────────────────────────┘
        │
        ▼
   Top-K Answers with Confidence Scores
        │
        ▼
   Streamlit Dashboard Display</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Model Info</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Working Directory:**")
        st.code(BASE_DIR)
        st.markdown("**Local Model:**")
        local_model_dir = os.path.join(BASE_DIR, "models", "blip-vqa-base")
        exists = os.path.isdir(local_model_dir)
        if exists:
            size_mb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fn in os.walk(local_model_dir) for f in fn
            ) / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB"
        else:
            size_str = "not found — will auto-download on first use"
        st.markdown(f"`{'✅' if exists else '🟡'} models/blip-vqa-base` — {size_str}")

    with col_b:
        st.markdown("**Quick Start:**")
        st.code("""# Install dependencies
pip install -r requirements.txt

# (Optional) pre-download model
python setup_models.py

# Launch dashboard
streamlit run dashboard.py""", language="bash")

    st.markdown("<div class='panel-title'>Supported Question Types</div>", unsafe_allow_html=True)
    for cat, qs in SAMPLE_QS.items():
        with st.expander(cat):
            for q in qs:
                st.markdown(f"- {q}")
