"""
GPT Report Generator - Use GPT-2 to generate safety incident reports
Requirements: pip install transformers
NOTE: No PyTorch used directly - uses transformers text-generation pipeline
"""

import pickle
import os
from datetime import datetime


# ─────────────────────────────────────────────
# 1. LOAD GPT-2 AND SAVE AS PICKLE
# ─────────────────────────────────────────────

def save_gpt_pipeline(output_path="models/gpt_reporter.pkl"):
    """
    Load GPT-2 text generation pipeline and save as pickle.
    """
    from transformers import pipeline

    print("📥 Loading GPT-2 text generation pipeline...")
    generator = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.75,
        top_p=0.92,
        repetition_penalty=1.3,
        pad_token_id=50256,
    )

    gpt_package = {
        "generator": generator,
        "model_name": "gpt2",
        "max_new_tokens": 200,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(gpt_package, f)

    print(f"✅ GPT pipeline saved: {output_path}")
    return gpt_package


def load_gpt_pipeline(pkl_path="models/gpt_reporter.pkl"):
    """Load GPT pipeline from pickle."""
    with open(pkl_path, "rb") as f:
        gpt_package = pickle.load(f)
    print("✅ GPT reporter loaded from pickle.")
    return gpt_package


# ─────────────────────────────────────────────
# 2. BUILD PROMPT
# ─────────────────────────────────────────────

def build_prompt(yolo_summary, risk_level, zone="Zone-A", worker_count=None):
    """
    Construct a structured prompt for GPT-2 report generation.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    helmet = yolo_summary.get("helmet_count", 0)
    no_helmet = yolo_summary.get("no_helmet_count", 0)
    total = yolo_summary.get("total_detections", 0)

    prompt = (
        f"SAFETY INCIDENT REPORT\n"
        f"Date: {timestamp}\n"
        f"Location: {zone}\n"
        f"Detected: {total} workers total. "
        f"Helmets worn: {helmet}. "
        f"Without helmets: {no_helmet}.\n"
        f"Risk Assessment: {risk_level}\n"
        f"Recommended Action: "
    )
    return prompt


# ─────────────────────────────────────────────
# 3. GENERATE REPORT
# ─────────────────────────────────────────────

def generate_report(yolo_summary, risk_level, zone="Zone-A", gpt_package=None):
    """
    Generate a safety report using GPT-2.

    Args:
        yolo_summary: dict from YOLO prediction (helmet_count, no_helmet_count etc.)
        risk_level: str - Low / Medium / High / Critical
        zone: str - site zone name
        gpt_package: dict loaded from pickle (optional, loads live if None)

    Returns:
        str - generated safety report text
    """
    prompt = build_prompt(yolo_summary, risk_level, zone)

    if gpt_package is None:
        # Load live pipeline without pickle
        from transformers import pipeline
        generator = pipeline(
            "text-generation",
            model="gpt2",
            pad_token_id=50256,
        )
    else:
        generator = gpt_package["generator"]

    output = generator(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.75,
        top_p=0.92,
        repetition_penalty=1.3,
    )

    generated_text = output[0]["generated_text"]

    # Clean up output - remove repeated prompt prefix
    if "Recommended Action:" in generated_text:
        report = generated_text.split("Recommended Action:")[0]
        action_part = generated_text.split("Recommended Action:")[1]
        # Take first 3 sentences from action
        sentences = action_part.replace("\n", " ").split(".")
        action = ". ".join(s.strip() for s in sentences[:3] if s.strip()) + "."
        final_report = report + f"Recommended Action: {action}"
    else:
        final_report = generated_text

    return final_report.strip()


# ─────────────────────────────────────────────
# 4. FALLBACK TEMPLATE REPORT (No GPU needed)
# ─────────────────────────────────────────────

REPORT_TEMPLATES = {
    "Low": """
✅ SAFETY COMPLIANCE REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: COMPLIANT
All detected workers are wearing proper helmets.
No immediate action required.
Recommendation: Continue regular safety monitoring.
Next Review: Scheduled routine inspection.
""",
    "Medium": """
⚠️ SAFETY ADVISORY REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━
Status: PARTIAL COMPLIANCE
One or more workers detected without helmets.
Action Required: Verbal warning to be issued immediately.
Supervisor must ensure all workers wear helmets before resuming work.
Follow-up inspection required within 2 hours.
""",
    "High": """
🚨 SAFETY VIOLATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: NON-COMPLIANT — HIGH RISK
Multiple workers detected without helmets near hazardous zones.
IMMEDIATE ACTION REQUIRED:
1. Stop work operations in affected zone.
2. Issue helmets to non-compliant workers immediately.
3. Supervisor must conduct safety briefing.
4. Document incident for safety audit.
5. Operations may resume only after full compliance verified.
""",
    "Critical": """
🆘 CRITICAL SAFETY EMERGENCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: CRITICAL BREACH — IMMEDIATE DANGER
Workers detected without helmets in active hazard zone!
EMERGENCY PROTOCOL ACTIVATED:
1. HALT ALL OPERATIONS IMMEDIATELY.
2. Evacuate all non-compliant workers from zone.
3. Alert site safety officer and management NOW.
4. Do NOT resume work until full safety audit completed.
5. Incident must be reported to safety board.
6. Medical standby recommended.
""",
}


def get_template_report(risk_level, yolo_summary, zone="Zone-A"):
    """
    Return a template-based report (no model required).
    Used as fallback or for demo purposes.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template = REPORT_TEMPLATES.get(risk_level, REPORT_TEMPLATES["Medium"])

    header = f"""
HELMET DETECTION SAFETY REPORT
Generated: {timestamp}
Zone: {zone}
Workers Detected: {yolo_summary.get('total_detections', 0)}
├── With Helmet:    {yolo_summary.get('helmet_count', 0)} ✅
└── Without Helmet: {yolo_summary.get('no_helmet_count', 0)} ❌
Risk Level: {risk_level}
"""
    return header + template


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== GPT Report Generator Setup ===")

    # Save GPT pipeline
    gpt_pkg = save_gpt_pipeline("models/gpt_reporter.pkl")

    # Test
    dummy_summary = {
        "total_detections": 3,
        "helmet_count": 1,
        "no_helmet_count": 2,
    }

    print("\n📝 Template Report (no GPU required):")
    report = get_template_report("High", dummy_summary, zone="Zone-C")
    print(report)

    print("\n🤖 GPT-2 Generated Report:")
    loaded = load_gpt_pipeline("models/gpt_reporter.pkl")
    gpt_report = generate_report(dummy_summary, "High", "Zone-C", loaded)
    print(gpt_report)
