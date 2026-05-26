import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Path to your friend's saved NLP model weights
NLP_MODEL_PATH = "models/nlp_weights"

# Use Apple GPU if available, otherwise CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class NLPModel:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(NLP_MODEL_PATH)
        self.model = DistilBertForSequenceClassification.from_pretrained(NLP_MODEL_PATH)
        self.model.to(DEVICE)
        self.model.eval()
        print(f"NLP model loaded on {DEVICE}")

    def get_depression_probability(self, text: str) -> float:
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(DEVICE)

        # Run through model, no gradient needed for inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # Index 1 = Depression probability
        depression_prob = float(probs[0][1].item())
        return depression_prob
