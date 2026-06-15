# models/gru_classifier.py
import torch
import torch.nn as nn
from data.constants import N_DEMOGRAPHICS

# models/gru_classifier.py
class GRUClassifier(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=1, dropout=0.3,
                 n_measurements=None, use_context_mask=False, use_delta_t=False):
        super().__init__()
        self.hidden_dim       = hidden_dim
        self.num_layers       = num_layers
        self.use_context_mask = use_context_mask
        self.use_delta_t      = use_delta_t

        # Input: measurements + datetime (no treatments)
        gru_input_dim = n_measurements + 1  # +1 for datetime
        if use_context_mask:
            gru_input_dim += n_measurements
        if use_delta_t:
            gru_input_dim += n_measurements

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Demographics injected into initial hidden state
        self.demo_proj = nn.Sequential(
            nn.Linear(N_DEMOGRAPHICS, hidden_dim),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, measurements, datetime, demographics,
                context_mask=None, delta_t=None):
        parts = [measurements, datetime]
        if self.use_context_mask and context_mask is not None:
            parts.append(context_mask)
        if self.use_delta_t and delta_t is not None:
            parts.append(delta_t)

        x  = torch.cat(parts, dim=-1)
        h0 = self.demo_proj(demographics).unsqueeze(0).repeat(self.num_layers, 1, 1)
        _, h_n = self.gru(x, h0)
        return self.classifier(h_n[-1]).squeeze(-1)