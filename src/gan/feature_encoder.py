# src/gan/feature_encoder.py
import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    """
    Simple MLP encoder to turn numeric features (tempo, key, mode, etc.)
    into a compact embedding vector for GAN conditioning.
    """
    def __init__(self, in_dim: int, hidden_dims=(256,128), out_dim: int = 128, dropout: float = 0.2, use_sn: bool = False):
        """
        in_dim: Number of input numeric features (e.g., 6)
        hidden_dims: Tuple of hidden layer sizes
        out_dim: The dimension of the output embedding
        """
        super().__init__()
        layers = []
        # --- FIX #2 ---
        layers.append(nn.LayerNorm(in_dim)) # Use LayerNorm, it's more robust here
        # --- END OF FIX ---
        prev = in_dim
        for h in hidden_dims:
            lin = nn.Linear(prev, h)
            if use_sn:
                try:
                    # Apply spectral norm if requested
                    from torch.nn.utils import spectral_norm
                    lin = spectral_norm(lin)
                except Exception:
                    print("[WARN] Could not apply spectral_norm. Is PyTorch version compatible?")
                    pass
            layers.append(lin)
            layers.append(nn.GELU()) # Using GELU as a modern activation
            layers.append(nn.Dropout(dropout))
            prev = h
        
        # Output layer
        layers.append(nn.Linear(prev, out_dim))
        # layers.append(nn.Tanh())   # Bound the embedding space between -1 and 1
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, in_dim) float tensor of numeric features
        return self.net(x)