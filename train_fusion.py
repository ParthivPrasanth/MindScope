import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from nlp_wrapper import NLPModel
from xgboost_wrapper import XGBoostModel
from fusion_model import FusionNetwork, DEVICE, FUSION_WEIGHTS_PATH

DATA_PATH = "data/survey_data.xlsx"


# ── Dataset ───────────────────────────────────────────────────────────────────

class FusionDataset(Dataset):
    def __init__(self, xgb_probs, nlp_probs, binary_labels, severity_labels):
        # Stack the two probabilities into a 2-value input vector per sample
        combined = np.stack([xgb_probs, nlp_probs], axis=1).astype(np.float32)

        self.X          = torch.tensor(combined)
        self.y_binary   = torch.tensor(binary_labels, dtype=torch.float32)
        self.y_severity = torch.tensor(severity_labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_binary[idx], self.y_severity[idx]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame, xgb_model: XGBoostModel, nlp_model: NLPModel):
    """
    Runs each row through both models and collects their output probabilities.
    This is the step that converts raw survey data into fusion layer inputs.
    """
    xgb_probs = []
    nlp_probs = []

    print(f"Extracting features from {len(df)} rows...")

    for i, row in df.iterrows():
        # Build survey dict from row
        survey = {
            "age":                row["age"],
            "gender":             row["gender"],
            "cgpa":               row["cgpa"],
            "sleep_duration":     row["sleep_duration"],
            "academic_pressure":  row["academic_pressure"],
            "study_satisfaction": row["study_satisfaction"],
            "financial_stress":   row["financial_stress"],
            "work_study_hours":   row["work_study_hours"],
            "dietary_habits":     row["dietary_habits"],
            "family_history":     row["family_history"],
            "suicidal_thoughts":  row["suicidal_thoughts"],
        }

        xgb_prob = xgb_model.get_depression_probability(survey)
        nlp_prob = nlp_model.get_depression_probability(str(row["free_text"]))

        xgb_probs.append(xgb_prob)
        nlp_probs.append(nlp_prob)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(df)} rows")

    return np.array(xgb_probs), np.array(nlp_probs)


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    # Load survey data
    df = pd.read_excel(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Compute PHQ-9 labels if not already present
    phq_cols = ["phq1","phq2","phq3","phq4","phq5","phq6","phq7","phq8","phq9"]

    if "severity_label" not in df.columns:
        df["phq9_score"] = df[phq_cols].sum(axis=1)

        def phq9_severity(score):
            if score <= 4:    return 0
            elif score <= 9:  return 1
            elif score <= 14: return 2
            elif score <= 19: return 3
            else:             return 4

        df["severity_label"] = df["phq9_score"].apply(phq9_severity)
        df["binary_label"]   = (df["severity_label"] > 0).astype(int)

    binary_labels   = df["binary_label"].values.astype(np.float32)
    severity_labels = df["severity_label"].values.astype(np.int64)

    print("\nSeverity distribution:")
    print(df["severity_label"].value_counts().sort_index())

    # Load both upstream models
    print("\nLoading models...")
    xgb_model = XGBoostModel()
    nlp_model = NLPModel()

    # Extract probabilities from both models for every row
    xgb_probs, nlp_probs = extract_features(df, xgb_model, nlp_model)

    print(f"\nXGB prob range: {xgb_probs.min():.3f} – {xgb_probs.max():.3f}")
    print(f"NLP prob range: {nlp_probs.min():.3f} – {nlp_probs.max():.3f}")

    # Split into train / val / test (80 / 10 / 10)
    idx = np.arange(len(df))
    idx_temp, idx_test = train_test_split(idx, test_size=0.1, random_state=42, stratify=binary_labels)
    idx_train, idx_val = train_test_split(idx_temp, test_size=0.111, random_state=42, stratify=binary_labels[idx_temp])

    def make_dataset(indices):
        return FusionDataset(
            xgb_probs[indices],
            nlp_probs[indices],
            binary_labels[indices],
            severity_labels[indices]
        )

    train_ds = make_dataset(idx_train)
    val_ds   = make_dataset(idx_val)
    test_ds  = make_dataset(idx_test)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Initialise fusion network
    # Input size is 2 — one prob from XGBoost, one from NLP
    model = FusionNetwork(input_size=2, dropout_rate=0.3).to(DEVICE)

    binary_loss_fn   = nn.BCEWithLogitsLoss()
    severity_loss_fn = nn.CrossEntropyLoss()
    optimizer        = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs = 50

    for epoch in range(epochs):

        # Training phase
        model.train()
        total_train_loss = 0

        for X_batch, y_bin, y_sev in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_bin   = y_bin.to(DEVICE)
            y_sev   = y_sev.to(DEVICE)

            optimizer.zero_grad()
            binary_logit, severity_logit = model(X_batch)

            loss_binary = binary_loss_fn(binary_logit.squeeze(), y_bin)

            depressed_mask = y_bin == 1
            if depressed_mask.sum() > 0:
                loss_severity = severity_loss_fn(
                    severity_logit[depressed_mask],
                    y_sev[depressed_mask]
                )
                loss = 0.5 * loss_binary + 0.5 * loss_severity
            else:
                loss = loss_binary

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for X_batch, y_bin, y_sev in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_bin   = y_bin.to(DEVICE)
                y_sev   = y_sev.to(DEVICE)

                binary_logit, severity_logit = model(X_batch)
                loss_binary = binary_loss_fn(binary_logit.squeeze(), y_bin)

                depressed_mask = y_bin == 1
                if depressed_mask.sum() > 0:
                    loss_severity = severity_loss_fn(
                        severity_logit[depressed_mask],
                        y_sev[depressed_mask]
                    )
                    val_loss = 0.5 * loss_binary + 0.5 * loss_severity
                else:
                    val_loss = loss_binary

                total_val_loss += val_loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val   = total_val_loss   / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), FUSION_WEIGHTS_PATH)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:>3}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to {FUSION_WEIGHTS_PATH}")

    # Plot training curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", color="steelblue")
    plt.plot(val_losses,   label="Val Loss",   color="coral")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fusion Model Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()
    print("Training curve saved to training_curve.png")

    # Final evaluation on test set
    model.load_state_dict(torch.load(FUSION_WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    all_bin_preds, all_bin_labels = [], []
    all_sev_preds, all_sev_labels = [], []

    with torch.no_grad():
        for X_batch, y_bin, y_sev in test_loader:
            X_batch = X_batch.to(DEVICE)

            binary_logit, severity_logit = model(X_batch)
            bin_preds = (torch.sigmoid(binary_logit).squeeze() > 0.5).int().cpu().numpy()
            sev_preds = torch.argmax(severity_logit, dim=1).cpu().numpy()

            all_bin_preds.extend(bin_preds)
            all_bin_labels.extend(y_bin.numpy())
            all_sev_preds.extend(sev_preds)
            all_sev_labels.extend(y_sev.numpy())

    all_bin_preds  = np.array(all_bin_preds)
    all_bin_labels = np.array(all_bin_labels)
    all_sev_preds  = np.array(all_sev_preds)
    all_sev_labels = np.array(all_sev_labels)

    print("\n" + "="*50)
    print("STAGE 1 — Binary Classification")
    print("="*50)
    print(classification_report(all_bin_labels, all_bin_preds,
                                 target_names=["Not Depressed", "Depressed"]))

    print("="*50)
    print("STAGE 2 — Severity (depressed samples only)")
    print("="*50)
    depressed_mask = all_bin_labels == 1
    severity_names = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]
    present        = sorted(np.unique(np.concatenate([
        all_sev_labels[depressed_mask],
        all_sev_preds[depressed_mask]
    ])))
    present_names  = [severity_names[i] for i in present]
    print(classification_report(
        all_sev_labels[depressed_mask],
        all_sev_preds[depressed_mask],
        labels=present,
        target_names=present_names
    ))


if __name__ == "__main__":
    train()
