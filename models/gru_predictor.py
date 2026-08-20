# models/gru_predictor.py
import torch
import torch.nn as nn
from data.constants import N_DEMOGRAPHICS

class GRUPredictor(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, dropout=0.1,
                 target_steps=12, n_measurements=None, n_treatments=None,
                 use_context_mask=False, use_treatment_mask=False, use_delta_t=False, use_pat_summary = False, pat_summary_hidden=64):
        super().__init__()
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.target_steps = target_steps
        self.n_measurements = n_measurements
        self.use_context_mask, self.use_treatment_mask, self.use_delta_t, self.use_pat_summary = use_context_mask, use_treatment_mask, use_delta_t, use_pat_summary

        enc_dim = n_measurements + n_treatments + 1
        if use_context_mask: enc_dim += n_measurements
        if use_treatment_mask: enc_dim += n_treatments
        if use_delta_t:      enc_dim += n_measurements

        self.encoder = nn.GRU(enc_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        demo_in_dim = N_DEMOGRAPHICS
        if use_pat_summary:
            pat_summary_dim = 3 * (n_measurements + n_treatments)
            self.summary_proj = nn.Sequential(nn.Linear(pat_summary_dim, pat_summary_hidden), nn.ReLU())
            demo_in_dim += pat_summary_hidden
        self.demo_proj = nn.Sequential(nn.Linear(demo_in_dim, hidden_dim), nn.Tanh())

        dec_in_dim = n_measurements + n_treatments + 1
        if use_treatment_mask:
            dec_in_dim += n_treatments
        self.decoder_cell = nn.GRUCell(dec_in_dim, hidden_dim)
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_measurements)
        )

    def forward(self, measurements, treatments, datetime, demographics,
                future_treatments, future_datetime,
                context_mask=None, treatment_mask=None, future_treatments_mask = None,delta_t=None,
                future_targets=None, teacher_forcing_prob=0.0, pat_summary = None):
        B = measurements.shape[0]

        parts = [measurements, treatments, datetime]
        if self.use_context_mask: parts.append(context_mask)
        if self.use_treatment_mask: parts.append(treatment_mask)
        if self.use_delta_t:      parts.append(delta_t)
        x = torch.cat(parts, dim=-1)

        if self.use_pat_summary and pat_summary is not None:
            summary_embed = self.summary_proj(pat_summary)
            hidden_input = torch.cat([demographics, summary_embed], dim=-1)
        else:
            hidden_input = demographics
        h0 = self.demo_proj(hidden_input).unsqueeze(0).repeat(self.num_layers, 1, 1)
        _, h_n = self.encoder(x, h0)
        h = h_n[-1]
        prev_meas = measurements[:, -1]

        preds = []
        for i in range(self.target_steps):
            step_parts = [prev_meas, future_treatments[:, i]]
            if self.use_treatment_mask:
                step_parts.append(future_treatments_mask[:, i])
            step_parts.append(future_datetime[:, i])
            step_in = torch.cat(step_parts, dim=-1)
            h = self.decoder_cell(step_in, h)
            pred_i = self.decoder_head(h)
            preds.append(pred_i)

            if future_targets is not None and teacher_forcing_prob > 0:
                use_real = (torch.rand(B, device=x.device) < teacher_forcing_prob).unsqueeze(-1)
                prev_meas = torch.where(use_real, future_targets[:, i], pred_i.detach())
            else:
                prev_meas = pred_i

        return torch.stack(preds, dim=1)