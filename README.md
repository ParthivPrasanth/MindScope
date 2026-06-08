# MindScope: Multi-Modal Depression Assessment

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00)](https://huggingface.co)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-1798D3)](https://xgboost.readthedocs.io)
[![React](https://img.shields.io/badge/React-Vite-61DAFB)](https://react.dev)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000)](https://flask.palletsprojects.com)

A full-stack ML pipeline that predicts depression risk using a **late-fusion multi-modal architecture** — combining structured clinical survey data with free-form text through a custom PyTorch neural network that outputs both a binary classification and a PHQ-9 calibrated severity score.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only and is not intended for clinical use.

---

## Highlights at a Glance

| | |
|---|---|
| 🧠 **NLP Model** | DistilBERT fine-tuned — **99% test accuracy, macro-F1 = 0.99** on 5,808 samples |
| 📊 **Tabular Model** | XGBoost with engineered features on Kaggle student depression dataset |
| 🔀 **Fusion Layer** | Custom PyTorch MLP, **247,174 parameters**, dual-head (binary + PHQ-9 severity) |
| 🗂️ **Training Data** | 29,040 English mental health text samples + structured clinical survey data |
| 🖥️ **Full-Stack** | Flask API + React/Vite frontend + 22-question PHQ-9 calibrated survey |
| 🛡️ **Ethics-First** | Zero-logging, confidence thresholding, crisis helpline triggers |

---

## What is this?

MindScope is a machine learning pipeline that predicts depression risk by combining traditional survey data with natural language processing.

This initially started as a group project for my Intro to AI class. However, I ended up taking ownership of the NLP models, the neural fusion layer, and the underlying inference architecture. I've spun my contributions out into this standalone repository to highlight the specific systems I built.

**My individual contributions:** NLP branch (full fine-tuning pipeline), fusion architecture design and implementation, end-to-end inference pipeline, modular inference wrappers, ethical safeguards, survey design, and React/Flask frontend integration.

---

## How it Works

Clinical surveys can be rigid, so I wanted to see if we could get better predictions by combining them with free-form text. The system splits the work into two independent branches:

1. **The Tabular Branch:** An XGBoost classifier that handles structured clinical survey data (age, CGPA, sleep, academic pressure, financial stress, etc.)
2. **The NLP Branch:** A DistilBERT model (fine-tuned using HuggingFace) that reads unstructured text input to detect emotional distress markers.

Instead of just averaging their scores, I built a **late-fusion neural network** using PyTorch. It takes the probability outputs from both branches, weighs them against each other, and outputs a two-stage prediction:

- **Stage 1:** Binary classification (Depressed vs. Not Depressed)
- **Stage 2:** Severity Level (Minimal → Mild → Moderate → Severe → Extremely Severe, calibrated using the PHQ-9 clinical framework)

```
User Survey Input  ──►  XGBoost Classifier      ──►  P(tabular)  ──┐
                                                                     ├──►  PyTorch Fusion Net  ──►  Binary + Severity
User Free Text     ──►  DistilBERT (fine-tuned)  ──►  P(text)    ──┘
```

---

## Results

### NLP Branch — DistilBERT

Fine-tuned on 29,040 English-filtered samples from the Kaggle Mental Health Sentiment dataset. Evaluated on a held-out test set of **5,808 samples**.

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 0.99 | 0.99 | 0.99 | 2,778 |
| Depression | 0.99 | 0.99 | 0.99 | 3,030 |
| **Overall (macro)** | **0.99** | **0.99** | **0.99** | **5,808** |

**Test Accuracy: 99%** · **GPU:** Tesla T4 (Kaggle) · **Epochs:** 3 · **Batch Size:** 32

**Training progression:**

| Epoch | Avg. Train Loss | Val Macro-F1 | Saved? |
|---|---|---|---|
| 1 | 0.1087 | 0.9757 | ✅ |
| 2 | 0.0402 | **0.9805** | ✅ **(best)** |
| 3 | 0.0169 | 0.9774 | ❌ |

### Fusion Layer — PyTorch MLP

- **Architecture:** 770→256→128 shared trunk (ReLU + Dropout 0.3), dual output heads
- **Binary head:** 128→64→1 (BCEWithLogitsLoss)
- **Severity head:** 128→64→5 classes (CrossEntropyLoss)
- **Total parameters:** 247,174
- **Loss:** weighted sum — 0.5 × binary + 0.5 × severity
- **Prototype trained on:** 200 synthetic paired probability samples (real-world evaluation pending survey data collection)

> The production fusion (`train_fusion.py`) is structured for a clean 80/10/10 train-val-test split with stratification once real survey responses are collected.

### XGBoost Branch

- 1,000 boosting rounds with early stopping (50-round patience)
- `TargetEncoder` for categorical features; engineered `Stress_Ratio` and `Work_Study_Total` features
- Feature importance tracked across Gain, Frequency, and Cover metrics (saved as CSVs)

---

## Architecture Details

### NLP Branch (DistilBERT)
- `distilbert-base-uncased` with a 2-class sequence classification head
- Input: free-form text (max 256 tokens), HuggingFace tokenizer
- Output: softmax probability of depression vs. normal

### Tabular Branch (XGBoost)
- `XGBClassifier`: 1,000 estimators, `max_depth=5`, `learning_rate=0.05`
- Input features: age, CGPA, academic pressure, sleep, financial stress, dietary habits, work/study hours, and more
- 15% stratified held-out test split

### Fusion Layer (PyTorch MLP)
- Shared trunk: 770→256→128, ReLU + Dropout 0.3
- Binary head: 128→64→1 — BCEWithLogitsLoss
- Severity head: 128→64→5 — CrossEntropyLoss (PHQ-9 thresholds: 0–4 Minimal, 5–9 Mild, 10–14 Moderate, 15–19 Severe, 20+ Extremely Severe)

---

## Datasets

| Component | Architecture | Training Data | Size |
|---|---|---|---|
| NLP Branch | DistilBERT | Kaggle: Sentiment Analysis for Mental Health | 53,043 total → 29,040 (English only) |
| Tabular Branch | XGBoost | Kaggle: Student Depression Dataset | `hopesb/student-depression-dataset` |
| Fusion Layer | PyTorch MLP | Synthetic paired distributions + custom survey | 200 synthetic pairs (prototype) |

A custom **22-question PHQ-9 calibrated survey** was designed for real-world data collection, covering demographics, lifestyle markers, clinical indicators (family history, prior depressive episodes, suicidal ideation), and all 9 PHQ-9 items.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **ML / DL** | PyTorch, HuggingFace Transformers (DistilBERT), XGBoost, scikit-learn, category_encoders |
| **Backend** | Python, Flask (`server.py`), modular wrappers (`nlp_wrapper.py`, `xgboost_wrapper.py`) |
| **Frontend** | React (Vite), JavaScript, HTML/CSS |
| **Data** | pandas, NumPy, kagglehub, langdetect |
| **Training Infra** | Kaggle Notebooks (Tesla T4 GPU), Apple MPS (fusion prototype) |
| **Serialization** | HuggingFace safetensors, XGBoost JSON, joblib |

---

## Project Structure

```
mindscope/
│
├── models/
│   ├── nlp_weights/          ← DistilBERT weights (config, safetensors, tokenizer)
│   └── xgboost/
│       ├── artifacts/        ← XGBoost artifacts (encoder, weights, importance)
│       └── src/              ← Tabular model source code
│
├── nlp_wrapper.py            ← DistilBERT inference class & probability extraction
├── xgboost_wrapper.py        ← Tabular data formatting & XGBoost inference
├── fusion_model.py           ← PyTorch MLP fusion architecture
├── train_fusion.py           ← Training loop, loss calculation, optimization
├── predict.py                ← End-to-end multi-modal prediction pipeline
├── generate_synthetic.py     ← Data augmentation for fusion layer training
│
├── requirements.txt
└── README.md
```

---

## Setup & Local Inference

**1. Set up the environment**

```bash
git clone https://github.com/ParthivPrasanth/MindScope.git
cd MindScope
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Download the Model Weights**

> Due to GitHub's file size limits, fine-tuned `.safetensors` and PyTorch `.pt` weights are hosted externally. Download them and place them in `models/nlp_weights/` and `models/xgboost/artifacts/`.

**3. Run a Prediction**

```bash
python predict.py
```

---

## Ethical Safeguards

Building AI for mental health requires strict ethical design. The following safeguards are hard-coded into the system:

- **Hardcoded Disclaimers:** All outputs include a disclaimer that this is a computational tool, not a clinical diagnosis.
- **Intervention Triggers:** Severe and Extremely Severe predictions automatically surface the iCall helpline (9152987821).
- **Confidence Thresholds:** The fusion network withholds predictions if confidence falls below a calibrated threshold, prompting the user to speak with a counsellor.
- **Zero-Logging Policy:** No user text, survey responses, or predictions are stored or logged at any point.
- **PHQ-9 Alignment:** Severity labels are calibrated against the clinically validated PHQ-9 framework.

---

## Limitations & Future Work

**Current limitations:**
- Fusion layer trained on synthetic data — real-world severity classification performance pending survey data collection
- XGBoost trained on a student-specific Kaggle dataset; generalization to broader populations unvalidated
- NLP model is binary (depression vs. normal); multi-class extension not yet built
- Model weights hosted externally due to GitHub file size constraints

**Planned improvements:**
- Collect real survey responses to retrain and properly evaluate the fusion layer
- Extend NLP branch to multi-class classification across all 7 mental health categories in the dataset
- Add SHAP explainability for XGBoost predictions
- Implement Bayesian calibration for better probability outputs
- Containerize with Docker for reproducible deployment
