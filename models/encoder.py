#LEGACY, CURRENTLY UNUSED

import torch
import torch.nn as nn

class LightEncoder(nn.Module):
    """
    Projects each variable independently into a d_model dimensional space,
    then pools across variables to get a single vector per timestep.
    This gives the GRU a richer representation than raw concatenated values.
    """
    def __init__(self, n_vars, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars
        # One linear layer per variable would be too many parameters
        # Instead use a shared projection + variable embedding
        self.value_proj = nn.Linear(1, d_model)
        self.var_embedding = nn.Embedding(n_vars, d_model)  # learned per-variable identity
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, V)
        returns: (B, T, d_model)
        """
        B, T, V = x.shape
        # Project each value
        val_emb = self.value_proj(x.unsqueeze(-1))  # (B, T, V, d_model)
        # Add variable identity embedding
        var_ids = torch.arange(V, device=x.device)
        var_emb = self.var_embedding(var_ids)        # (V, d_model)
        combined = val_emb + var_emb                 # (B, T, V, d_model) broadcast
        combined = self.norm(combined)
        # Mean pool across variables
        return combined.mean(dim=2)                  # (B, T, d_model)