"""
BERT Risk Classifier - Fine-tune BERT on Safety Reports and Save as Pickle
Requirements: pip install transformers scikit-learn pandas numpy
NOTE: No PyTorch used directly - using sklearn pipeline + transformers pipeline API
"""

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ─────────────────────────────────────────────
# 1. SAMPLE SAFETY REPORT DATASET
# ─────────────────────────────────────────────

SAMPLE_DATA = [
    # Low Risk
    ("All workers wearing helmets in Zone A", "Low"),
    ("Safety equipment properly worn by all personnel", "Low"),
    ("Routine inspection passed. No violations found.", "Low"),
    ("Workers following all safety protocols on site", "Low"),
    ("Helmets and vests worn by all workers in Zone B", "Low"),
    ("Safety briefing completed. All gear in place.", "Low"),

    # Medium Risk
    ("One worker not wearing helmet near entry gate", "Medium"),
    ("Minor safety violation observed at loading dock", "Medium"),
    ("Helmet missing from 1 worker during break period", "Medium"),
    ("Partial compliance observed in Zone C workers", "Medium"),
    ("Two workers found without helmets in parking area", "Medium"),

    # High Risk
    ("Worker without helmet near heavy machinery Zone D", "High"),
    ("No helmet detected on 3 workers near crane operation", "High"),
    ("Multiple safety violations near active construction", "High"),
    ("Workers without protective gear near electrical zone", "High"),
    ("Helmet non-compliance detected in high-risk zone", "High"),

    # Critical
    ("Worker without helmet injured near active crane", "Critical"),
    ("Emergency: 5 workers without PPE near blast site", "Critical"),
    ("Critical safety breach: No helmets in demolition zone", "Critical"),
    ("Accident risk imminent: unprotected workers at height", "Critical"),
    ("Severe violation: workers exposed to falling debris without helmets", "Critical"),
]


# ─────────────────────────────────────────────
# 2. EXTRACT BERT EMBEDDINGS (using pipeline)
# ─────────────────────────────────────────────

def get_bert_embeddings(texts):
    """
    Use HuggingFace pipeline to extract BERT CLS embeddings.
    No raw PyTorch needed - uses transformers feature-extraction pipeline.
    """
    from transformers import pipeline

    print("📥 Loading BERT feature extractor...")
    extractor = pipeline(
        "feature-extraction",
        model="bert-base-uncased",
        tokenizer="bert-base-uncased",
        truncation=True,
        max_length=128,
    )

    embeddings = []
    for i, text in enumerate(texts):
        print(f"  Extracting [{i+1}/{len(texts)}]: {text[:50]}...")
        output = extractor(text)
        # output shape: [1, seq_len, 768] - take CLS token (index 0)
        cls_embedding = np.array(output[0][0])  # Shape: (768,)
        embeddings.append(cls_embedding)

    return np.array(embeddings)


# ─────────────────────────────────────────────
# 3. TRAIN BERT + SKLEARN CLASSIFIER
# ─────────────────────────────────────────────

def train_bert_classifier(output_path="models/bert_classifier.pkl"):
    """
    Extract BERT embeddings → train LogisticRegression → save as pickle.
    """
    texts = [item[0] for item in SAMPLE_DATA]
    labels = [item[1] for item in SAMPLE_DATA]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Extract BERT embeddings
    print("\n🔵 Extracting BERT embeddings...")
    X = get_bert_embeddings(texts)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression on top of BERT embeddings
    print("\n🏋️ Training Logistic Regression on BERT embeddings...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save everything as pickle
    model_package = {
        "classifier": clf,
        "label_encoder": le,
        "classes": list(le.classes_),
        "embedding_dim": 768,
        "model_name": "bert-base-uncased",
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model_package, f)

    print(f"\n✅ BERT classifier saved: {output_path}")
    return model_package


# ─────────────────────────────────────────────
# 4. LOAD & PREDICT
# ─────────────────────────────────────────────

def load_bert_classifier(pkl_path="models/bert_classifier.pkl"):
    """Load BERT classifier from pickle."""
    with open(pkl_path, "rb") as f:
        model_package = pickle.load(f)
    print("✅ BERT classifier loaded from pickle.")
    return model_package


def predict_risk(text, model_package):
    """
    Predict risk level from incident text.

    Args:
        text: str - incident description
        model_package: dict loaded from pickle

    Returns:
        dict with predicted label and probabilities
    """
    from transformers import pipeline

    # Extract embedding
    extractor = pipeline(
        "feature-extraction",
        model=model_package["model_name"],
        truncation=True,
        max_length=128,
    )

    output = extractor(text)
    embedding = np.array(output[0][0]).reshape(1, -1)

    # Predict
    clf = model_package["classifier"]
    le = model_package["label_encoder"]

    pred_id = clf.predict(embedding)[0]
    pred_proba = clf.predict_proba(embedding)[0]

    pred_label = le.inverse_transform([pred_id])[0]
    class_probs = {
        cls: round(float(prob), 3)
        for cls, prob in zip(le.classes_, pred_proba)
    }

    return {
        "risk_level": pred_label,
        "confidence": round(float(max(pred_proba)), 3),
        "probabilities": class_probs,
    }


# ─────────────────────────────────────────────
# DEMO PREDICTIONS (without training)
# ─────────────────────────────────────────────

def demo_predict_risk(text):
    """
    Simple rule-based risk prediction for demo purposes.
    Used when BERT model is not yet trained.
    """
    text_lower = text.lower()

    critical_keywords = ["injured", "accident", "emergency", "blast", "demolition", "severe", "imminent"]
    high_keywords = ["machinery", "crane", "electrical", "multiple", "3 workers", "4 workers", "5 workers"]
    low_keywords = ["all workers", "all personnel", "passed", "complete", "properly", "following"]

    if any(k in text_lower for k in critical_keywords):
        return {"risk_level": "Critical", "confidence": 0.91, "probabilities": {"Critical": 0.91, "High": 0.06, "Medium": 0.02, "Low": 0.01}}
    elif any(k in text_lower for k in high_keywords):
        return {"risk_level": "High", "confidence": 0.87, "probabilities": {"Critical": 0.05, "High": 0.87, "Medium": 0.06, "Low": 0.02}}
    elif any(k in text_lower for k in low_keywords):
        return {"risk_level": "Low", "confidence": 0.93, "probabilities": {"Critical": 0.01, "High": 0.02, "Medium": 0.04, "Low": 0.93}}
    else:
        return {"risk_level": "Medium", "confidence": 0.75, "probabilities": {"Critical": 0.05, "High": 0.12, "Medium": 0.75, "Low": 0.08}}


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== BERT Risk Classifier Setup ===")
    print("Training BERT + Sklearn classifier...")
    model_package = train_bert_classifier("models/bert_classifier.pkl")

    # Test prediction
    test_texts = [
        "Worker found without helmet near crane",
        "All workers wearing helmets on site",
        "Critical breach: workers without PPE near blast zone",
    ]

    print("\n🔍 Test Predictions:")
    loaded = load_bert_classifier("models/bert_classifier.pkl")
    for text in test_texts:
        result = predict_risk(text, loaded)
        print(f"  Text: {text[:50]}...")
        print(f"  → Risk: {result['risk_level']} (conf: {result['confidence']})\n")
