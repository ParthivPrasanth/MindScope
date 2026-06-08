

# MindScope: Multi-Modal Depression Assessment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-1798D3)

## What is this?
MindScope is a machine learning pipeline that predicts depression risk by combining traditional survey data with natural language processing. 

This initially started as a group project for my Intro to AI class. However, I ended up taking ownership of the NLP models, the neural fusion layer, and the underlying inference architecture. I’ve spun my contributions out into this standalone repository to highlight the specific systems I built.

## How it Works
Clinical surveys can be rigid, so I wanted to see if we could get better predictions by combining them with free-form text. The system splits the work into two independent branches:

1. **The Tabular Branch:** An XGBoost classifier that handles structured clinical survey data.
2. **The NLP Branch:** A DistilBERT model (fine-tuned using Hugging Face) that reads unstructured text input to look for emotional distress markers.

Instead of just averaging their scores, I built a **late-fusion neural network** using PyTorch. It takes the probability outputs from both branches, weighs them against each other, and outputs a two-stage prediction:
* **Stage 1:** Binary (Depressed vs. Not Depressed)
* **Stage 2:** Severity Level (Minimal, Mild, Moderate, Severe, or Extremely Severe, calibrated using the PHQ-9 clinical framework).

```text
User Survey Input ──► XGBoost Classifier ──► P(tabular) ──┐
                                                          ├──► PyTorch Fusion Net ──► Binary + Severity
User Free Text    ──► DistilBERT Model   ──► P(text)    ──┘

```


## Project Structure

```text
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
├── fusion_model.py           ← PyTorch Multi-Layer Perceptron (MLP) fusion architecture
├── train_fusion.py           ← PyTorch training loop, loss calculation, and optimization
├── predict.py                ← End-to-end multi-modal prediction pipeline
├── generate_synthetic.py     ← Data augmentation script for fusion layer training
│
├── requirements.txt
└── README.md

```

## Setup & Local Inference

**1. Set up the environment**

```bash
git clone [https://github.com/YOUR_USERNAME/Mindscope.git](https://github.com/YOUR_USERNAME/Mindscope.git)
cd Mindscope
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

**2. Download the Model Weights**
*Note: Due to GitHub's file size limits, my fine-tuned `.safetensors` and PyTorch `.pt` weights are hosted externally.* Download them and place them in `models/nlp_weights/` and `models/xgboost/artifacts/`.

**3. Run a Prediction**
Execute the end-to-end prediction passing both text and survey data through the respective models and into the fusion layer:

```bash
python predict.py

```

## Datasets Used

To make sure each branch was an "expert" in its domain before fusing them, the models were trained independently:

| Component | Architecture | Training Data Source |
| --- | --- | --- |
| **NLP Branch** | DistilBERT | Kaggle: Sentiment Analysis for Mental Health |
| **Tabular Branch** | XGBoost | Kaggle: Student Depression Dataset |
| **Fusion Layer** | PyTorch MLP | Augmented synthetic paired distributions |

## Ethical Safeguards

Building AI for mental health requires strict ethical boundaries. I implemented the following safeguards in this architecture:

* **Hardcoded Disclaimers:** All outputs include a disclaimer that this is a computational tool, not a clinical diagnosis.
* **Intervention Triggers:** Severe and Extremely Severe results automatically display the iCall helpline (9152987821).
* **Confidence Thresholds:** The model withholds predictions if the fusion network is uncertain.
* **Zero-Logging:** No user data, text inputs, or predictions are stored or logged.

---

*Disclaimer: This project is for educational purposes only and is not intended for clinical use.*

```

```
