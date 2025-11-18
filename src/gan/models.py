"""
Generator and Discriminator (Critic) for MELO-GAN, updated for WGAN-GP
and numeric feature conditioning.

Key Changes:
- Generator: Accepts additional numeric_embedding and concatenates it with
  noise and (optional) latent embedding.
- Discriminator (Critic):
  - No BatchNorm1d (as recommended for WGAN-GP).
  - Accepts numeric_embedding and concatenates it with the feature
    vector before the final output heads.
  - real_fake head outputs a raw score (logit), not a probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple utility: MLP to expand combined input -> decoder-latent
class NoiseToLatent(nn.Module):
    def __init__(self, noise_dim, out_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, z):
        return self.net(z)

# Generator decoder (mostly unchanged)
class GeneratorDecoder(nn.Module):
    def __init__(self, latent_dim=128, max_notes=512, out_channels=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_notes = max_notes
        
        # Calculate intermediate dimensions
        # This structure (8x upscale) assumes max_notes is e.g. 100, 128, 512
        # For max_notes=100, 100//8 = 12. 
        # Let's make this more robust.
        self.reduced_len = max(1, max_notes // 8) # 8 = 2*2*2 (3 upsample layers)
        
        flattened = 256 * self.reduced_len
        
        self.pre = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, flattened),
            nn.ReLU(True),
        )
        
        # 3-layer ConvTranspose stack for 8x upsampling
        self.deconv = nn.Sequential(
            # Input: (B, 256, reduced_len)
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1), # (B, 128, reduced_len*2)
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1), # (B, 64, reduced_len*4)
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1), # (B, 4, reduced_len*8)
            # No final activation, output raw values
        )

    def forward(self, latent):
        b = latent.size(0)
        y = self.pre(latent)
        
        y = y.view(b, 256, self.reduced_len)
        
        out = self.deconv(y)            # (B, out_channels, notes)
        out = out.permute(0, 2, 1)      # (B, notes, out_channels)
        
        # Trim/pad to exact max_notes
        current_len = out.size(1)
        if current_len > self.max_notes:
            out = out[:, :self.max_notes, :]
        elif current_len < self.max_notes:
            pad = out.new_zeros((b, self.max_notes - current_len, out.size(2)))
            out = torch.cat([out, pad], dim=1)
            
        return out

class Generator(nn.Module):
    def __init__(self, noise_dim=128, latent_dim=128, mode="conditioning", 
                 hidden=512, max_notes=512, note_dim=4, numeric_embed_dim=0):
        super().__init__()
        assert mode in ("conditioning", "warm_start")
        self.mode = mode
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim # AE latent dim
        self.max_notes = max_notes
        self.note_dim = note_dim
        self.numeric_embed_dim = numeric_embed_dim

        # The input to the first MLP depends on the mode and conditioning
        self.input_dim = noise_dim + self.numeric_embed_dim
        if self.mode == "conditioning":
            self.input_dim += latent_dim
            
        print(f"[G] Init Generator. Mode: {self.mode}. Input MLP dim: {self.input_dim}")

        # This MLP maps the combined input vector to the decoder's expected latent_dim
        self.noise_to_latent = NoiseToLatent(self.input_dim, latent_dim, hidden=hidden)
        self.decoder = GeneratorDecoder(latent_dim=latent_dim, max_notes=max_notes, out_channels=note_dim)

    def forward(self, noise, encoder_latent=None, numeric_embedding=None):
        """
        noise: (B, noise_dim)
        encoder_latent: (B, latent_dim) only used in conditioning mode
        numeric_embedding: (B, numeric_embed_dim) used in all modes
        """
        
        if self.numeric_embed_dim > 0:
            assert numeric_embedding is not None, "numeric_embedding is required"
            inputs = [noise, numeric_embedding]
        else:
            inputs = [noise]

        if self.mode == "conditioning":
            assert encoder_latent is not None, "conditioning mode requires encoder latent input"
            inputs.append(encoder_latent)
        
        # Concatenate all available inputs
        x = torch.cat(inputs, dim=1)
            
        latent = self.noise_to_latent(x)  # (B, latent_dim)
        out = self.decoder(latent)
        return out, latent
# --- NEW DISCRIMINATOR (WGAN-GP) ---
class Discriminator(nn.Module):
    """
    This is the WGAN-GP Discriminator (Critic).
    It ONLY judges Real vs Fake and returns a single score tensor.
    """
    def __init__(self, max_notes=512, note_dim=4, emb_dim=256, numeric_embed_dim=0):
        super().__init__()
        # Conv stack
        self.conv = nn.Sequential(
            nn.Conv1d(note_dim, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True), # No BN for WGAN-GP
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True), # No BN for WGAN-GP
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Final head takes combined features
        self.combined_dim = emb_dim + numeric_embed_dim
        self.real_fake = nn.Linear(self.combined_dim, 1)

    def forward(self, notes, numeric_embedding=None):
        x = notes.permute(0, 2, 1)
        h = self.conv(x)
        h = self.pool(h)
        feat = self.fc(h.view(h.size(0), -1))
        
        if numeric_embedding is not None:
            feat = torch.cat([feat, numeric_embedding], dim=1)
        
        # Return ONLY the score tensor
        score = self.real_fake(feat)
        return score.squeeze(1)