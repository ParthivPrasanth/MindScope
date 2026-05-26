import torch
import numpy as np

from nlp_wrapper import NLPModel
from xgboost_wrapper import XGBoostModel
from fusion_model import FusionNetwork, load_fusion_model, DEVICE

# Confidence threshold — if depression probability is in this grey zone
# the system withholds its verdict and asks the user to seek help
CONFIDENCE_THRESHOLD = 0.35

SEVERITY_MAP = {
    0: "Minimal / Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Extremely Severe"
}

HELPLINE = "iCall: 9152987821"


def predict(survey_dict: dict, free_text: str) -> dict:
    """
    End-to-end prediction for one user.

    Args:
        survey_dict : dictionary of survey answers with these keys:
                      age, gender, cgpa, sleep_duration, academic_pressure,
                      study_satisfaction, financial_stress, work_study_hours,
                      dietary_habits, family_history, suicidal_thoughts
        free_text   : the user's written description of how they feel

    Returns:
        dict with result, probabilities, severity, and a message
    """

    # Load models
    xgb_model    = XGBoostModel()
    nlp_model    = NLPModel()
    fusion_model = load_fusion_model()

    # Get probabilities from both branches
    xgb_prob = xgb_model.get_depression_probability(survey_dict)
    nlp_prob = nlp_model.get_depression_probability(free_text)

    print(f"\nXGBoost depression probability : {xgb_prob:.4f}")
    print(f"NLP depression probability     : {nlp_prob:.4f}")

    # Build fusion input
    x = torch.tensor([[xgb_prob, nlp_prob]], dtype=torch.float32).to(DEVICE)

    # Run fusion model
    fusion_model.eval()
    with torch.no_grad():
        binary_logit, severity_logit = fusion_model(x)

    depression_prob = float(torch.sigmoid(binary_logit).item())

    # Confidence gate
    # If the model is hovering near 50% it's not confident enough
    if CONFIDENCE_THRESHOLD < depression_prob < (1 - CONFIDENCE_THRESHOLD):
        return {
            "result":              "uncertain",
            "depression_probability": round(depression_prob, 3),
            "message": (
                "We are not confident enough to assess your risk level based on the "
                "information provided. Please speak with a counsellor or mental health "
                f"professional. {HELPLINE}"
            ),
            "disclaimer": (
                "This tool is not a clinical diagnosis. It is an educational AI project. "
                "If you are experiencing distress, please contact a qualified mental health professional."
            )
        }

    # Not depressed
    if depression_prob < 0.5:
        return {
            "result":                 "not_depressed",
            "depression_probability": round(depression_prob, 3),
            "message":                "No significant indicators of depression detected.",
            "disclaimer": (
                "This tool is not a clinical diagnosis. It is an educational AI project. "
                "If you are experiencing distress, please contact a qualified mental health professional."
            )
        }

    # Depressed — assess severity
    severity_probs = torch.softmax(severity_logit, dim=1).squeeze()
    severity_idx   = int(torch.argmax(severity_probs).item())
    severity_label = SEVERITY_MAP[severity_idx]
    severity_conf  = float(severity_probs[severity_idx].item())

    # Build response
    result = {
        "result":                 "depressed",
        "depression_probability": round(depression_prob, 3),
        "severity":               severity_label,
        "severity_confidence":    round(severity_conf, 3),
        "message": (
            f"Depression indicators detected. "
            f"Assessed severity: {severity_label}."
        ),
        "disclaimer": (
            "This tool is not a clinical diagnosis. It is an educational AI project. "
            "If you are experiencing distress, please contact a qualified mental health professional."
        )
    }

    # Add helpline for severe cases
    if severity_idx >= 3:
        result["support"] = (
            f"Your responses indicate a high level of distress. "
            f"Please reach out for support. {HELPLINE}"
        )

    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Sample 1 — should return depressed, moderate to severe
    sample_survey = {
        "age":                20,
        "gender":             "Male",
        "cgpa":               5.2,
        "sleep_duration":     "Less than 5 hours",
        "academic_pressure":  5,
        "study_satisfaction": 1,
        "financial_stress":   4,
        "work_study_hours":   3,
        "dietary_habits":     "Unhealthy",
        "family_history":     "Yes",
        "suicidal_thoughts":  "Yes",
    }

    sample_text = (
        "I haven't been to class in a week and I can't make myself get up. "
        "Everything feels pointless and heavy. I've stopped eating properly "
        "and I feel like I'm disappearing. I don't see the point in anything anymore."
    )

    print("="*60)
    print("TEST PREDICTION 1 — Expected: Depressed, Severe")
    print("="*60)
    result = predict(sample_survey, sample_text)
    for key, val in result.items():
        print(f"{key}: {val}")

    print()

    # Sample 2 — should return not depressed
    sample_survey_2 = {
        "age":                21,
        "gender":             "Female",
        "cgpa":               8.4,
        "sleep_duration":     "7-8 hours",
        "academic_pressure":  2,
        "study_satisfaction": 4,
        "financial_stress":   1,
        "work_study_hours":   6,
        "dietary_habits":     "Healthy",
        "family_history":     "No",
        "suicidal_thoughts":  "No",
    }

    sample_text_2 = (
        "This week has been pretty good. I submitted my assignment on time "
        "and caught up with friends over the weekend. Feeling a bit tired "
        "but overall in a decent headspace. Looking forward to the holidays."
    )

    print("="*60)
    print("TEST PREDICTION 2 — Expected: Not Depressed")
    print("="*60)
    result2 = predict(sample_survey_2, sample_text_2)
    for key, val in result2.items():
        print(f"{key}: {val}")
