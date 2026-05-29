# models/gru_predictor.py
import torch
import torch.nn as nn
from data.constants import N_DEMOGRAPHICS

class GRUPredictor(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, dropout=0.1,
                 target_steps=12, encoder_dim=64,
                 n_measurements=None, n_treatments=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.target_steps = target_steps
        self.n_measurements = n_measurements

        # Raw input: measurements + measurement_mask + time_since_last_obs + treatments + datetime
        gru_input_dim = n_measurements * 3 + n_treatments + 1


        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.demo_proj = nn.Sequential(
            nn.Linear(N_DEMOGRAPHICS, hidden_dim),
            nn.Tanh()
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_measurements * target_steps)
        )

    def forward(self, measurements, treatments, datetime, demographics, 
                context_mask, delta_t):
        B  = measurements.shape[0]
        x  = torch.cat([measurements, context_mask, delta_t, treatments, datetime], dim=-1)
        h0 = self.demo_proj(demographics)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        _, h_n = self.gru(x, h0)
        last_hidden = h_n[-1]
        out = self.output_proj(last_hidden)
        return out.view(B, self.target_steps, self.n_measurements)