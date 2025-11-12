"""
Generator and Discriminator for MELO-GAN.
Generator supports two modes:
 - conditioning: accepts noise z and optional encoder latent (concatenated)
 - warm_start: generator uses same decoder-like architecture as AE decoder (so AE decoder weights can be loaded)
Discriminator outputs:
 - real/fake probability
 - emotion logits (4 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple utility: MLP to expand noise -> decoder-latent
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

# Generator decoder mirror (similar idea to AE ConvDecoder)
class GeneratorDecoder(nn.Module):
    def __init__(self, latent_dim=128, max_notes=512, out_channels=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_notes = max_notes
        reduced_len = max(1, max_notes // 8)
        flattened = 256 * reduced_len
        self.pre = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, flattened),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, latent):
        b = latent.size(0)
        y = self.pre(latent)
        # reshape to (B, 256, reduced_len)
        total = y.shape[1]
        # attempt to reshape into (256, L)
        if total % 256 != 0:
            reduced_len = max(1, total // 256)
            y = y[:, :256 * reduced_len]
        else:
            reduced_len = total // 256
        y = y.view(b, 256, reduced_len)
        out = self.deconv(y)            # (B, out_channels, notes)
        out = out.permute(0, 2, 1)      # (B, notes, out_channels)
        # trim/pad to exact max_notes
        if out.size(1) > self.max_notes:
            out = out[:, :self.max_notes, :]
        elif out.size(1) < self.max_notes:
            pad = out.new_zeros((out.size(0), self.max_notes - out.size(1), out.size(2)))
            out = torch.cat([out, pad], dim=1)
        return out

class Generator(nn.Module):
    def __init__(self, noise_dim=128, latent_dim=128, mode="conditioning", hidden=512, max_notes=512, note_dim=4):
        super().__init__()
        assert mode in ("conditioning", "warm_start")
        self.mode = mode
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.max_notes = max_notes
        self.note_dim = note_dim

        if self.mode == "conditioning":
            # We will concatenate noise + encoder latent => pass through MLP to produced decoder-latent
            self.noise_to_latent = NoiseToLatent(noise_dim + latent_dim, latent_dim, hidden=hidden)
            self.decoder = GeneratorDecoder(latent_dim=latent_dim, max_notes=max_notes, out_channels=note_dim)
        else:
            # warm_start: noise maps to decoder latent, then decode
            self.noise_to_latent = NoiseToLatent(noise_dim, latent_dim, hidden=hidden)
            # decoder architecture should match AE decoder to accept loading
            self.decoder = GeneratorDecoder(latent_dim=latent_dim, max_notes=max_notes, out_channels=note_dim)

    def forward(self, noise, encoder_latent=None):
        """
        noise: (B, noise_dim)
        encoder_latent: (B, latent_dim) only used in conditioning mode
        """
        if self.mode == "conditioning":
            assert encoder_latent is not None, "conditioning mode requires encoder latent input"
            x = torch.cat([noise, encoder_latent], dim=1)
            latent = self.noise_to_latent(x)  # (B, latent_dim)
            out = self.decoder(latent)
            return out, latent
        else:
            latent = self.noise_to_latent(noise)
            out = self.decoder(latent)
            return out, latent

# Discriminator: classifies real/fake and predicts emotion (auxiliary)
class Discriminator(nn.Module):
    def __init__(self, max_notes=512, note_dim=4, emb_dim=256, num_emotions=4):
        super().__init__()
        # Conv stack across time
        self.conv = nn.Sequential(
            nn.Conv1d(note_dim, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # final pooling + dense
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1, emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # outputs
        self.real_fake = nn.Linear(emb_dim, 1)        # logits
        self.emotion = nn.Linear(emb_dim, num_emotions)

    def forward(self, notes):
        # notes: (B, MAX_NOTES, note_dim)
        x = notes.permute(0, 2, 1)  # (B, note_dim, MAX_NOTES)
        h = self.conv(x)
        h = self.pool(h)            # (B, 256, 1)
        feat = self.fc(h.view(h.size(0), -1))
        rf_logit = self.real_fake(feat)
        emo_logits = self.emotion(feat)
        return rf_logit.squeeze(1), emo_logits, feat
