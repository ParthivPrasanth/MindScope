import os
import joblib
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from category_encoders import TargetEncoder
import xgboost as xgb

from config import (
    ARTIFACTS_DIR,
    MODEL_PATH,
    ENCODER_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_WEIGHTS_PATH,
    DETAILED_IMPORTANCE_PATH,
    KAGGLE_DATASET,
)
from data_utils import prepare_features

def load_data_from_kaggle():
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded Kaggle dataset.")

    csv_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_path)
    return df

def train():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_data_from_kaggle()
    print("Columns:", df.columns.tolist())

    df = prepare_features(df)

    X = df.drop("Depression", axis=1)
    y = df["Depression"].astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    encoder = TargetEncoder(cols=cat_cols)
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_val_enc = encoder.transform(X_val)
    X_test_enc = encoder.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42
    )

    model.fit(
        X_train_enc,
        y_train,
        eval_set=[(X_val_enc, y_val)],
        verbose=False
    )

    preds = model.predict(X_test_enc)
    probs = model.predict_proba(X_test_enc)[:, 1]

    print(f"\nFinal Test Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Save model artifacts
    model.save_model(MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(X_train_enc.columns.tolist(), FEATURE_COLUMNS_PATH)

    # Save feature importance
    feature_names = X_train_enc.columns.tolist()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Weight": model.feature_importances_
    }).sort_values("Weight", ascending=False)

    importance_df.to_csv(FEATURE_WEIGHTS_PATH, index=False)

    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    cover_scores = booster.get_score(importance_type="cover")

    detailed_importance = pd.DataFrame({
        "Feature": feature_names
    })
    detailed_importance["Gain"] = detailed_importance["Feature"].map(gain_scores).fillna(0)
    detailed_importance["Frequency"] = detailed_importance["Feature"].map(weight_scores).fillna(0)
    detailed_importance["Cover"] = detailed_importance["Feature"].map(cover_scores).fillna(0)
    detailed_importance = detailed_importance.sort_values("Gain", ascending=False)

    detailed_importance.to_csv(DETAILED_IMPORTANCE_PATH, index=False)

    print("\nArtifacts saved successfully.")

if __name__ == "__main__":
    train()