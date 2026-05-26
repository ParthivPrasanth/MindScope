import torch
import torch.nn as nn

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Path where the trained fusion weights are saved
FUSION_WEIGHTS_PATH = "best_fusion_model.pt"


class FusionNetwork(nn.Module):
    """
    Takes two inputs:
        - XGBoost depression probability  (1 float)
        - NLP depression probability      (1 float)
    Total input size: 2

    Stage 1 output: binary prediction (depressed or not)
    Stage 2 output: severity level (0-4) — only meaningful if Stage 1 = 1
    """

    def __init__(self, input_size=2, dropout_rate=0.3):
        super(FusionNetwork, self).__init__()

        # Shared trunk — both heads learn from this
        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Stage 1 — is this person depressed?
        self.binary_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Stage 2 — how severe is the depression?
        self.severity_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

    def forward(self, x):
        shared_out    = self.shared(x)
        binary_logit  = self.binary_head(shared_out)
        severity_logit= self.severity_head(shared_out)
        return binary_logit, severity_logit


def load_fusion_model() -> FusionNetwork:
    model = FusionNetwork(input_size=2, dropout_rate=0.3).to(DEVICE)
    model.load_state_dict(torch.load(FUSION_WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print(f"Fusion model loaded from {FUSION_WEIGHTS_PATH}")
    return model
