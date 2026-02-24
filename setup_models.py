"""
setup_models.py - Run this ONCE to download and save all models as pickle files.
Requirements: pip install ultralytics transformers scikit-learn numpy pillow opencv-python
"""

import os
import pickle

def setup_all_models():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("=" * 50)
    print("🚀 HELMET DETECTION SYSTEM - MODEL SETUP")
    print("=" * 50)

    # ── 1. YOLO ──────────────────────────────────────
    print("\n[1/3] Setting up YOLO model...")
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")  # Downloads automatically
        model_data = {
            "model": model,
            "classes": {0: "helmet", 1: "no_helmet"},
            "conf_threshold": 0.4,
            "version": "yolov8n",
        }
        with open("models/yolo_helmet.pkl", "wb") as f:
            pickle.dump(model_data, f)
        print("   ✅ YOLO saved: models/yolo_helmet.pkl")
    except Exception as e:
        print(f"   ⚠️  YOLO setup failed: {e}")
        print("   → Make sure ultralytics is installed: pip install ultralytics")

    # ── 2. BERT ──────────────────────────────────────
    print("\n[2/3] Setting up BERT classifier...")
    try:
        from transformers import pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        import numpy as np

        SAMPLE_DATA = [
            ("All workers wearing helmets in Zone A", "Low"),
            ("Safety equipment properly worn by all personnel", "Low"),
            ("Routine inspection passed. No violations found.", "Low"),
            ("Workers following all safety protocols on site", "Low"),
            ("Helmets and vests worn by all workers", "Low"),
            ("One worker not wearing helmet near entry gate", "Medium"),
            ("Minor safety violation observed at loading dock", "Medium"),
            ("Helmet missing from worker during break period", "Medium"),
            ("Two workers found without helmets in parking", "Medium"),
            ("Worker without helmet near heavy machinery", "High"),
            ("No helmet detected on 3 workers near crane", "High"),
            ("Multiple safety violations near active construction", "High"),
            ("Workers without protective gear near electrical zone", "High"),
            ("Worker without helmet injured near active crane", "Critical"),
            ("Emergency: workers without PPE near blast site", "Critical"),
            ("Critical safety breach: No helmets in demolition zone", "Critical"),
            ("Severe violation: workers exposed without helmets", "Critical"),
        ]

        texts = [d[0] for d in SAMPLE_DATA]
        labels = [d[1] for d in SAMPLE_DATA]

        print("   Extracting BERT embeddings (this may take a few minutes)...")
        extractor = pipeline(
            "feature-extraction",
            model="bert-base-uncased",
            truncation=True,
            max_length=128,
        )

        embeddings = []
        for text in texts:
            out = extractor(text)
            embeddings.append(np.array(out[0][0]))
        X = np.array(embeddings)

        le = LabelEncoder()
        y = le.fit_transform(labels)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)

        bert_package = {
            "classifier": clf,
            "label_encoder": le,
            "classes": list(le.classes_),
            "model_name": "bert-base-uncased",
        }

        with open("models/bert_classifier.pkl", "wb") as f:
            pickle.dump(bert_package, f)
        print("   ✅ BERT saved: models/bert_classifier.pkl")

    except Exception as e:
        print(f"   ⚠️  BERT setup failed: {e}")
        print("   → Make sure transformers & sklearn are installed")

    # ── 3. GPT ───────────────────────────────────────
    print("\n[3/3] Setting up GPT-2 reporter...")
    try:
        from transformers import pipeline

        generator = pipeline(
            "text-generation",
            model="gpt2",
            pad_token_id=50256,
        )

        gpt_package = {
            "generator": generator,
            "model_name": "gpt2",
        }

        with open("models/gpt_reporter.pkl", "wb") as f:
            pickle.dump(gpt_package, f)
        print("   ✅ GPT-2 saved: models/gpt_reporter.pkl")

    except Exception as e:
        print(f"   ⚠️  GPT setup failed: {e}")
        print("   → Make sure transformers is installed")

    # ── Summary ──────────────────────────────────────
    print("\n" + "=" * 50)
    print("📁 Model Files Created:")
    for fname in ["yolo_helmet.pkl", "bert_classifier.pkl", "gpt_reporter.pkl"]:
        path = f"models/{fname}"
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"   ✅ {path} ({size:.1f} MB)")
        else:
            print(f"   ❌ {path} (NOT created)")

    print("\n🎉 Setup complete! Run: streamlit run app/dashboard.py")
    print("=" * 50)


if __name__ == "__main__":
    setup_all_models()
