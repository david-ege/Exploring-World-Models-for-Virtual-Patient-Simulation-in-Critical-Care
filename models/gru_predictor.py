# models/gru_predictor.py
import torch
import torch.nn as nn
from data.constants import N_DEMOGRAPHICS

class GRUPredictor(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, dropout=0.1,
                 target_steps=12, n_measurements=None, n_treatments=None,
                 use_context_mask=False, use_delta_t=False):
        super().__init__()
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.target_steps = target_steps
        self.n_measurements = n_measurements
        self.use_context_mask, self.use_delta_t = use_context_mask, use_delta_t

        enc_dim = n_measurements + n_treatments + 1
        if use_context_mask: enc_dim += n_measurements
        if use_delta_t:      enc_dim += n_measurements

        self.encoder = nn.GRU(enc_dim, hidden_dim, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.demo_proj = nn.Sequential(nn.Linear(N_DEMOGRAPHICS, hidden_dim), nn.Tanh())

        dec_in_dim = n_measurements + n_treatments + 1
        self.decoder_cell = nn.GRUCell(dec_in_dim, hidden_dim)
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_measurements)
        )

    def forward(self, measurements, treatments, datetime, demographics,
                future_treatments, future_datetime,
                context_mask=None, delta_t=None,
                future_targets=None, teacher_forcing_prob=0.0):
        B = measurements.shape[0]

        parts = [measurements, treatments, datetime]
        if self.use_context_mask: parts.append(context_mask)
        if self.use_delta_t:      parts.append(delta_t)
        x = torch.cat(parts, dim=-1)
        h0 = self.demo_proj(demographics).unsqueeze(0).repeat(self.num_layers, 1, 1)
        _, h_n = self.encoder(x, h0)
        h = h_n[-1]
        prev_meas = measurements[:, -1]

        preds = []
        for i in range(self.target_steps):
            step_in = torch.cat([prev_meas, future_treatments[:, i], future_datetime[:, i]], dim=-1)
            h = self.decoder_cell(step_in, h)
            pred_i = self.decoder_head(h)
            preds.append(pred_i)

            if future_targets is not None and teacher_forcing_prob > 0:
                use_real = (torch.rand(B, device=x.device) < teacher_forcing_prob).unsqueeze(-1)
                prev_meas = torch.where(use_real, future_targets[:, i], pred_i.detach())
            else:
                prev_meas = pred_i

        return torch.stack(preds, dim=1)