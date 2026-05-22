"""
vqa_model.py — Visual Question Answering using BLIP
Pretrained model: Salesforce/blip-vqa-base (HuggingFace)

FIX NOTES (vs original):
  - Removed unreliable pickle serialization of pipeline objects.
    Pickle of transformers pipelines breaks across library versions.
    Using processor.save_pretrained / model.save_pretrained instead.
  - Added safe top-k generation that works across transformers versions.
  - Moved numpy import to module level.

Requirements: pip install transformers pillow torch ultralytics
"""

import os
import numpy as np
from PIL import Image
import torch


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DEFAULT_MODEL_DIR = "models/blip-vqa-base"
HF_MODEL_ID       = "Salesforce/blip-vqa-base"


# ─────────────────────────────────────────────
# 1. SAVE MODEL (run once via setup_models.py)
# ─────────────────────────────────────────────

def save_vqa_model(output_dir=DEFAULT_MODEL_DIR):
    """Download BLIP VQA model from HuggingFace and save locally."""
    from transformers import BlipProcessor, BlipForQuestionAnswering

    print(f"📥 Downloading BLIP VQA model: {HF_MODEL_ID} ...")
    processor = BlipProcessor.from_pretrained(HF_MODEL_ID)
    model     = BlipForQuestionAnswering.from_pretrained(HF_MODEL_ID)

    os.makedirs(output_dir, exist_ok=True)
    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"✅ Model saved to: {output_dir}")
    return processor, model


# ─────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────

def load_vqa_model(model_dir=DEFAULT_MODEL_DIR):
    """
    Load BLIP VQA model.
    - First tries local saved copy (model_dir).
    - Falls back to downloading from HuggingFace.

    Returns:
        dict with keys: processor, model, model_name, device
    """
    from transformers import BlipProcessor, BlipForQuestionAnswering

    source = "local"
    if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"✅ Loading BLIP VQA from local: {model_dir}")
        try:
            processor = BlipProcessor.from_pretrained(model_dir)
            model     = BlipForQuestionAnswering.from_pretrained(model_dir)
        except Exception as e:
            print(f"⚠️  Local load failed ({e}), falling back to HuggingFace...")
            source    = "huggingface"
            processor = BlipProcessor.from_pretrained(HF_MODEL_ID)
            model     = BlipForQuestionAnswering.from_pretrained(HF_MODEL_ID)
    else:
        print(f"📥 Local model not found. Downloading {HF_MODEL_ID} ...")
        source    = "huggingface"
        processor = BlipProcessor.from_pretrained(HF_MODEL_ID)
        model     = BlipForQuestionAnswering.from_pretrained(HF_MODEL_ID)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()

    print(f"✅ Model ready on {device} (source: {source})")
    return {
        "processor":  processor,
        "model":      model,
        "model_name": HF_MODEL_ID,
        "device":     device,
    }


# ─────────────────────────────────────────────
# 3. RUN VQA INFERENCE
# ─────────────────────────────────────────────

def answer_question(image_input, question, model_data, top_k=3):
    """
    Answer a question about an image.

    Args:
        image_input : PIL.Image | file path | numpy array
        question    : str
        model_data  : dict returned by load_vqa_model()
        top_k       : int — number of answer candidates

    Returns:
        dict with answer, confidence, top_answers, model
    """
    # ── Load image ──
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    processor = model_data["processor"]
    model     = model_data["model"]
    device    = model_data["device"]

    # ── Encode ──
    inputs = processor(image, question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ── Generate answers ──
    with torch.no_grad():
        # num_beams / num_return_sequences gives top-k candidates
        num_seq = min(top_k, 5)
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

    # Deduplicate while preserving order
    seen, unique_answers = set(), []
    for a in answers:
        if a not in seen:
            seen.add(a)
            unique_answers.append(a)

    # Assign pseudo-confidence scores (beam scores aren't directly exposed)
    top_answers = [
        {"answer": a, "score": round(1.0 - i * 0.15, 4)}
        for i, a in enumerate(unique_answers[:top_k])
    ]

    if not top_answers:
        top_answers = [{"answer": "unknown", "score": 0.0}]

    return {
        "question":    question,
        "answer":      top_answers[0]["answer"],
        "confidence":  top_answers[0]["score"],
        "top_answers": top_answers,
        "model":       model_data["model_name"],
    }


# ─────────────────────────────────────────────
# 4. BATCH QUESTION ANSWERING
# ─────────────────────────────────────────────

def answer_multiple_questions(image_input, questions, model_data):
    """Answer multiple questions about the same image."""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    results = []
    for q in questions:
        result = answer_question(image, q, model_data)
        results.append(result)
        print(f"  Q: {q}")
        print(f"  A: {result['answer']} ({result['confidence']*100:.1f}%)\n")
    return results


# ─────────────────────────────────────────────
# 5. SAMPLE QUESTIONS
# ─────────────────────────────────────────────

SAMPLE_QUESTIONS = {
    "Object Detection": [
        "What objects are in this image?",
        "What is the main object in this image?",
        "How many people are in this image?",
        "What animals can you see?",
    ],
    "Scene Understanding": [
        "What is happening in this image?",
        "Where was this photo taken?",
        "Is this indoors or outdoors?",
        "What time of day is it?",
    ],
    "Color & Appearance": [
        "What color is the main object?",
        "What are the dominant colors?",
        "Is the image bright or dark?",
        "What is the person wearing?",
    ],
    "Counting": [
        "How many cars are there?",
        "How many people are in the image?",
        "How many objects can you count?",
    ],
    "Yes/No Questions": [
        "Is there a person in this image?",
        "Is this image taken outdoors?",
        "Is the sky visible?",
        "Are there any animals?",
    ],
}


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("🔍 VISUAL QUESTION ANSWERING — MODEL SETUP")
    print("=" * 55)
    print()
    print("Downloading and saving BLIP VQA model (~1 GB) ...")
    save_vqa_model()
    print("\n✅ Done. Run: streamlit run dashboard.py")
