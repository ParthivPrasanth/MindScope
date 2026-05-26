import sys
import os

# Add the xgboost src folder to Python's path
# so it can find config.py, data_utils.py etc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "xgboost"))

import src.config as _cfg
_cfg.MODEL_PATH            = "models/xgboost/artifacts/xgb_model.json"
_cfg.ENCODER_PATH          = "models/xgboost/artifacts/encoder.pkl"
_cfg.FEATURE_COLUMNS_PATH  = "models/xgboost/artifacts/feature_columns.pkl"

from src.predict_model import DepressionPredictor as _BasePredictor


def _build_xgb_input(survey_dict: dict) -> dict:
    """
    Translates your survey's field names and value formats
    into the format the XGBoost model was trained on.

    Your survey collects:
        age, gender, cgpa, sleep_duration, academic_pressure,
        study_satisfaction, financial_stress, work_study_hours,
        dietary_habits, family_history, suicidal_thoughts

    The XGBoost model expects:
        Gender, Age, City, Profession, Academic Pressure,
        Work Pressure, CGPA, Study Satisfaction, Job Satisfaction,
        Sleep Duration, Dietary Habits, Degree,
        Have you ever had suicidal thoughts ?,
        Work/Study Hours, Financial Stress,
        Family History of Mental Illness
    """

    # Financial stress: your survey gives 1-5 number
    # Model expects "Low" / "Medium" / "High" string
    # (data_utils.py then maps those back to 1/2/3)
    financial_stress_raw = survey_dict.get("financial_stress", 3)
    if financial_stress_raw <= 2:
        financial_stress_str = "Low"
    elif financial_stress_raw <= 3:
        financial_stress_str = "Medium"
    else:
        financial_stress_str = "High"

    # Suicidal thoughts: your survey gives Yes/No/Prefer not to answer
    # Model expects Yes/No
    suicidal_raw = survey_dict.get("suicidal_thoughts", "No")
    suicidal_str = "Yes" if suicidal_raw == "Yes" else "No"

    # Family history: your survey gives Yes/No/I'm not sure
    # Model expects Yes/No
    family_raw = survey_dict.get("family_history", "No")
    family_str = "Yes" if family_raw == "Yes" else "No"

    # Gender: pass through directly
    gender = survey_dict.get("gender", "Male")

    return {
        "Gender":                           gender,
        "Age":                              survey_dict.get("age", 20),
        "City":                             "Mumbai",       # dummy — not predictive
        "Profession":                       "Student",      # dummy — all students
        "Academic Pressure":                survey_dict.get("academic_pressure", 3),
        "Work Pressure":                    0,              # always 0 for students
        "CGPA":                             survey_dict.get("cgpa", 7.0),
        "Study Satisfaction":               survey_dict.get("study_satisfaction", 3),
        "Job Satisfaction":                 0,              # always 0 for students
        "Sleep Duration":                   survey_dict.get("sleep_duration", "6-7 hours"),
        "Dietary Habits":                   survey_dict.get("dietary_habits", "Moderate"),
        "Degree":                           "B.Tech",       # dummy — not strongly predictive
        "Have you ever had suicidal thoughts ?": suicidal_str,
        "Work/Study Hours":                 survey_dict.get("work_study_hours", 6),
        "Financial Stress":                 financial_stress_str,
        "Family History of Mental Illness": family_str,
    }


class XGBoostModel:
    def __init__(self):
        self.predictor = _BasePredictor()
        print("XGBoost model loaded")

    def get_depression_probability(self, survey_dict: dict) -> float:
        # Translate survey format to model format
        xgb_input = _build_xgb_input(survey_dict)

        # Run prediction — returns (binary_pred, probability)
        _, prob = self.predictor.predict(xgb_input)
        return float(prob)
