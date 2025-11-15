"""
Emotion Discriminator model for MELO-GAN Phase 1A.

Supports two input modes:
 - 'latent' : accepts encoder latent vectors (e.g. 128-d)
 - 'notes'  : accepts (max_notes x note_dim) arrays, e.g. (512,4)

Design:
 - If input_mode == 'latent' -> MLP classifier
 - If input_mode == 'notes'  -> Conv1D encoder -> pooled -> MLP classifier
 - Optional spectral normalization and dropout for robustness.

Usage:
    model = EmotionDiscriminator(cfg)
    logits = model(x)             # raw logits (batch, n_classes)
    probs  = torch.softmax(logits, dim=-1)
"""

from typing import Optional, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_sn=False):
        super().__init__()
        conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        if use_sn:
            try:
                from torch.nn.utils import spectral_norm
                conv = spectral_norm(conv)
            except Exception:
                pass
        self.net = nn.Sequential(
            conv,
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class NotesEncoder(nn.Module):
    """
    Encodes (batch, max_notes, note_dim) -> (batch, hidden_dim)
    Uses a stack of Conv1D blocks over the time/notes axis.
    """
    def __init__(self, note_dim: int = 4, hidden_dim: int = 256, num_blocks: int = 4, use_sn: bool = False):
        super().__init__()
        layers = []
        in_ch = note_dim
        ch = 64
        for i in range(num_blocks):
            layers.append(ConvBlock1D(in_ch, ch, kernel_size=5 if i == 0 else 3, padding=2 if i == 0 else 1, use_sn=use_sn))
            in_ch = ch
            ch = min(ch * 2, hidden_dim)
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse temporal axis
        self.project = nn.Linear(in_ch, hidden_dim)

    def forward(self, notes):  # notes: (B, T, D)
        # convert to (B, D, T)
        x = notes.permute(0, 2, 1)
        x = self.conv(x)           # (B, C, T)
        x = self.pool(x).squeeze(-1)  # (B, C)
        x = self.project(x)        # (B, hidden_dim)
        return x


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(256, 128), n_classes: int = 4, dropout: float = 0.2, use_sn: bool = False):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            lin = nn.Linear(prev, h)
            if use_sn:
                try:
                    from torch.nn.utils import spectral_norm
                    lin = spectral_norm(lin)
                except Exception:
                    pass
            layers.append(lin)
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x):
        x = self.net(x)
        logits = self.head(x)
        return logits


class EmotionDiscriminator(nn.Module):
    """
    High-level wrapper for emotion classifier.

    Configurable via dictionary `cfg`:
      cfg = {
        'input_mode': 'latent' or 'notes',
        'latent_dim': 128,
        'note_dim': 4,
        'notes_hidden': 256,
        'notes_blocks': 4,
        'mlp_hidden': [256, 128],
        'n_classes': 4,
        'dropout': 0.2,
        'use_spectral_norm': False
      }
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg.copy()
        self.input_mode = cfg.get('input_mode', 'latent')
        self.n_classes = cfg.get('n_classes', 4)
        self.use_sn = cfg.get('use_spectral_norm', False)
        self.dropout = cfg.get('dropout', 0.2)

        if self.input_mode == 'latent':
            latent_dim = cfg.get('latent_dim', 128)
            self.encoder = None
            self.classifier = MLPClassifier(in_dim=latent_dim,
                                            hidden_dims=tuple(cfg.get('mlp_hidden', (256, 128))),
                                            n_classes=self.n_classes,
                                            dropout=self.dropout,
                                            use_sn=self.use_sn)
        elif self.input_mode == 'notes':
            note_dim = cfg.get('note_dim', 4)
            notes_hidden = cfg.get('notes_hidden', 256)
            notes_blocks = cfg.get('notes_blocks', 4)
            self.encoder = NotesEncoder(note_dim=note_dim,
                                        hidden_dim=notes_hidden,
                                        num_blocks=notes_blocks,
                                        use_sn=self.use_sn)
            self.classifier = MLPClassifier(in_dim=notes_hidden,
                                            hidden_dims=tuple(cfg.get('mlp_hidden', (256, 128))),
                                            n_classes=self.n_classes,
                                            dropout=self.dropout,
                                            use_sn=self.use_sn)
        else:
            raise ValueError("input_mode must be 'latent' or 'notes'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: either (B, latent_dim) when input_mode == 'latent'
           or (B, T, note_dim) when input_mode == 'notes'
        returns logits (B, n_classes)
        """
        if self.input_mode == 'latent':
            if x.dim() != 2:
                raise ValueError(f"Expected latent input shape (B, latent_dim), got {x.shape}")
            feats = x
        else:
            # notes path
            if x.dim() != 3:
                raise ValueError(f"Expected notes input shape (B, T, note_dim), got {x.shape}")
            feats = self.encoder(x)

        logits = self.classifier(feats)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def freeze_encoder(self):
        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self):
        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad = True


if __name__ == "__main__":
    # very small smoke test
    cfg_latent = {
        'input_mode': 'latent',
        'latent_dim': 128,
        'mlp_hidden': [256, 128],
        'n_classes': 4
    }
    m = EmotionDiscriminator(cfg_latent)
    x = torch.randn(8, 128)
    print("latent logits:", m(x).shape)

    cfg_notes = {
        'input_mode': 'notes',
        'note_dim': 4,
        'notes_hidden': 256,
        'notes_blocks': 3,
        'mlp_hidden': [256, 128],
        'n_classes': 4
    }
    m2 = EmotionDiscriminator(cfg_notes)
    x2 = torch.randn(8, 512, 4)
    print("notes logits:", m2(x2).shape)
