"""
YOLO Model - Fine-tune YOLOv8 on Helmet Dataset and Save as Pickle
Requirements: pip install ultralytics opencv-python pillow
"""

import pickle
import os
import numpy as np
from PIL import Image
import cv2

# ─────────────────────────────────────────────
# 1. FINE-TUNE YOLO (Run once to train)
# ─────────────────────────────────────────────

def train_yolo(data_yaml="data/helmet.yaml", epochs=50, imgsz=640):
    """
    Fine-tune YOLOv8 on custom helmet dataset.
    Requires: helmet.yaml + dataset in YOLO format
    """
    from ultralytics import YOLO

    # Load pretrained YOLOv8 nano (COCO weights)
    model = YOLO("yolov8n.pt")

    # Fine-tune
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name="helmet_detector",
        patience=10,
        save=True,
    )

    print("✅ Training Complete!")
    print(f"Best weights saved at: runs/detect/helmet_detector/weights/best.pt")
    return model


# ─────────────────────────────────────────────
# 2. SAVE YOLO MODEL AS PICKLE
# ─────────────────────────────────────────────

def save_yolo_pickle(weights_path="yolov8n.pt", output_path="models/yolo_helmet.pkl"):
    """
    Wrap the YOLO model and save as .pkl file.
    Uses pretrained yolov8n.pt if custom weights not available.
    """
    from ultralytics import YOLO

    print(f"📦 Loading YOLO from: {weights_path}")
    model = YOLO(weights_path)

    # Wrap in a serializable dict
    model_data = {
        "model": model,
        "classes": {0: "helmet", 1: "no_helmet"},
        "conf_threshold": 0.4,
        "version": "yolov8n",
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"✅ YOLO model saved as pickle: {output_path}")


# ─────────────────────────────────────────────
# 3. LOAD YOLO FROM PICKLE & PREDICT
# ─────────────────────────────────────────────

def load_yolo_pickle(pkl_path="models/yolo_helmet.pkl"):
    """Load YOLO model from pickle file."""
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    print("✅ YOLO model loaded from pickle.")
    return model_data


def predict_helmet(image_input, model_data):
    """
    Run YOLO inference on an image.

    Args:
        image_input: PIL Image or numpy array or file path
        model_data: dict loaded from pickle

    Returns:
        annotated_image (numpy array), detections (list of dicts)
    """
    model = model_data["model"]
    conf = model_data["conf_threshold"]

    # Convert PIL to numpy if needed
    if isinstance(image_input, Image.Image):
        image_np = np.array(image_input)
    elif isinstance(image_input, str):
        image_np = cv2.imread(image_input)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        image_np = image_input

    # Run inference
    results = model(image_np, conf=conf, verbose=False)

    detections = []
    helmet_count = 0
    no_helmet_count = 0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            label = model_data["classes"].get(cls_id, f"class_{cls_id}")

            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": xyxy,
            })

            if label == "helmet":
                helmet_count += 1
            elif label == "no_helmet":
                no_helmet_count += 1

    # Draw annotations
    annotated = results[0].plot()  # Returns BGR numpy array
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    summary = {
        "total_detections": len(detections),
        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "detections": detections,
    }

    return annotated_rgb, summary


# ─────────────────────────────────────────────
# 4. HELMET YAML TEMPLATE
# ─────────────────────────────────────────────

HELMET_YAML = """
# helmet.yaml - Custom Dataset Config
path: ./data
train: train/images
val: val/images
test: test/images

nc: 2
names:
  0: helmet
  1: no_helmet
"""


def create_yaml():
    os.makedirs("data", exist_ok=True)
    with open("data/helmet.yaml", "w") as f:
        f.write(HELMET_YAML)
    print("✅ helmet.yaml created.")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== YOLO Helmet Model Setup ===")
    print("Step 1: Creating YAML config...")
    create_yaml()

    print("\nStep 2: Saving pretrained YOLO as pickle (use custom weights after training)...")
    save_yolo_pickle(
        weights_path="yolov8n.pt",       # Change to best.pt after fine-tuning
        output_path="models/yolo_helmet.pkl"
    )

    print("\nStep 3: Test loading pickle...")
    model_data = load_yolo_pickle("models/yolo_helmet.pkl")
    print("Model classes:", model_data["classes"])
    print("\n✅ YOLO setup complete!")
