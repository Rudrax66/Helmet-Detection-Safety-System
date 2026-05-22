"""
setup_models.py — Download & Save VQA Model
Run this ONCE before launching the dashboard.

Requirements:
    pip install transformers pillow torch streamlit ultralytics

Usage:
    python setup_models.py
"""

import os


def setup_vqa():
    os.makedirs("models", exist_ok=True)

    print("=" * 55)
    print("🚀 VQA SYSTEM — MODEL SETUP")
    print("=" * 55)
    print()
    print("This will download the BLIP VQA model (~1 GB).")
    print("The model is saved directly (no pickle) for reliability.")
    print()

    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        from PIL import Image
        import torch

        SAVE_DIR = "models/blip-vqa-base"

        print("[1/2] Downloading BLIP VQA model from HuggingFace...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model     = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

        print("[2/2] Saving model locally...")
        processor.save_pretrained(SAVE_DIR)
        model.save_pretrained(SAVE_DIR)

        # Quick test
        print("\n🧪 Running quick test...")
        test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        inputs   = processor(test_img, "What color is this image?", return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        print(f"   Test answer: {answer} ✅")

        print()
        print("=" * 55)
        print("📁 Model saved to:", SAVE_DIR)
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fn in os.walk(SAVE_DIR) for f in fn
        ) / (1024 * 1024)
        print(f"   Total size: {size_mb:.1f} MB")
        print()
        print("🎉 Setup complete!")
        print("   Run: streamlit run dashboard.py")
        print("=" * 55)

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have installed:")
        print("  pip install transformers pillow torch streamlit ultralytics")


if __name__ == "__main__":
    setup_vqa()
