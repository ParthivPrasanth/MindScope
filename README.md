# MindScope — Multi-Modal Student Depression Assessment System

**CSS 2203 — Introduction to Artificial Intelligence | Group Project**

---

## Overview

MindScope is a multi-modal machine learning system that predicts depression risk
in students by combining two independent signals:

1. **Structured survey data** — processed by a fine-tuned XGBoost classifier
2. **Free-text emotional input** — processed by a fine-tuned DistilBERT NLP model

The outputs of both models are passed into a **fusion neural network** (PyTorch)
that produces a two-stage prediction:
- Stage 1: Binary — depressed or not depressed
- Stage 2: Severity level — Minimal, Mild, Moderate, Severe, or Extremely Severe

Severity is assessed using the **PHQ-9** clinical framework.

---

## Team

| Name | Roll No. | Reg No. | Responsibility |
|------|----------|---------|----------------|
| Siddharth Anand | 22 | 240905144 | Fusion layer, data pipeline, Google Form |
| Abhirup Bhattacharya | 57 | 240905556 | Report |
| Anurimaa Pewekar | 58 | 240905582 | XGBoost model, SHAP |
| Parthiv Prasanth | 47 | 240905450 | NLP model (DistilBERT) |
| Arnav Gandotra | 50 | 240905490 | UI, deployment |

---

## Project Structure

```
mindscope_fusion/
│
├── models/
│   ├── nlp_weights/          ← DistilBERT weights (not in repo, share via Drive)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   │
│   └── xgboost/
│       ├── artifacts/        ← XGBoost artifacts (not in repo, share via Drive)
│       │   ├── encoder.pkl
│       │   ├── feature_columns.pkl
│       │   ├── xgb_model.json
│       │   ├── xgb_feature_weights.csv
│       │   └── xgb_detailed_importance.csv
│       └── src/
│           ├── config.py
│           ├── data_utils.py
│           ├── predict_model.py
│           ├── train_model.py
│           └── model_info.py
│
├── data/                     ← Survey data (not in repo, share via Drive)
│   └── survey_data.xlsx
│
├── nlp_wrapper.py            ← Loads DistilBERT, exposes get_depression_probability()
├── xgboost_wrapper.py        ← Loads XGBoost, translates survey input format
├── fusion_model.py           ← FusionNetwork architecture definition
├── train_fusion.py           ← Full training pipeline
├── predict.py                ← End-to-end single prediction
├── generate_synthetic_data.py← Generates synthetic training data
│
├── best_fusion_model.pt      ← Trained fusion weights (not in repo, share via Drive)
├── training_curve.png        ← Generated after training
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mindscope_fusion.git
cd mindscope_fusion
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model weights and data

The following files are **not included in this repository** due to size constraints.
Download them from the shared Google Drive folder and place them as shown:

```
models/nlp_weights/          ← 4 files from NLP model Drive folder
models/xgboost/artifacts/    ← 5 files from XGBoost model Drive folder
data/survey_data.xlsx        ← survey data file
best_fusion_model.pt         ← trained fusion weights
```

### 5. Train the fusion model

```bash
python train_fusion.py
```

### 6. Run a test prediction

```bash
python predict.py
```

---

## Model Architecture

```
User Survey Input ──► XGBoostModel ──► P(depression) ──┐
                                                        ├──► FusionNetwork ──► Binary + Severity
User Free Text   ──► DistilBERT   ──► P(depression) ──┘
```

The fusion network takes two probability values as input (one per branch)
and produces a two-stage classification output.

---

## Datasets

| Dataset | Source | Used For |
|---------|--------|----------|
| Student Depression Dataset | Kaggle (adilshamim8) | XGBoost branch training |
| Sentiment Analysis for Mental Health | Kaggle (suchintikasarkar) | DistilBERT branch training |
| Synthetic paired survey data | Generated (generate_synthetic_data.py) | Fusion layer training |

---

## Ethical Safeguards

- All outputs include a disclaimer that this is not a clinical diagnosis
- Severe and Extremely Severe results display the iCall helpline (9152987821)
- A confidence threshold withholds predictions when the model is uncertain
- No user data is stored or logged

---

## Requirements

See `requirements.txt` for the full dependency list.

---

*This project is for educational purposes only and is not intended for clinical use.*