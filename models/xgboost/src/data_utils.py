import pandas as pd

SUSPICIOUS_CITIES = ["Saanvi", "Mira", "Gaurav", "Harsh", "Kibara", "Rashi", "Nalini"]

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    if "City" in df.columns:
        df = df[~df["City"].isin(SUSPICIOUS_CITIES)]

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Academic Pressure" in df.columns and "Study Satisfaction" in df.columns:
        df["Stress_Ratio"] = df["Academic Pressure"] / df["Study Satisfaction"].replace(0, 1)

    if "Work/Study Hours" in df.columns and "Work Pressure" in df.columns:
        df["Work_Study_Total"] = df["Work/Study Hours"] + df["Work Pressure"]

    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Don't create unnecessary copies - work on the dataframe directly
    # Clean the data
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    if "City" in df.columns:
        df = df[~df["City"].isin(SUSPICIOUS_CITIES)]
    
    # Convert Financial Stress to numeric (like it was during training)
    if "Financial Stress" in df.columns:
        stress_mapping = {"Low": 1, "Medium": 2, "High": 3}
        df["Financial Stress"] = df["Financial Stress"].map(stress_mapping).fillna(2)
    
    # Engineer features - this preserves original columns
    if "Academic Pressure" in df.columns and "Study Satisfaction" in df.columns:
        df["Stress_Ratio"] = df["Academic Pressure"] / df["Study Satisfaction"].replace(0, 1)

    if "Work/Study Hours" in df.columns and "Work Pressure" in df.columns:
        df["Work_Study_Total"] = df["Work/Study Hours"] + df["Work Pressure"]
    
    # Fill missing values
    df = df.fillna(0)
    
    return df