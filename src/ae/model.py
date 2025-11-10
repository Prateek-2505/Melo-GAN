# src/ae/model.py
import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=4, latent_dim=128):
        super().__init__()
        # input shape: (B, in_channels, MAX_NOTES)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool_out = None
        self._output_flat = None
        # We'll infer the flattened size at runtime using a dummy forward in init
        self._linear = None
        self.latent_dim = latent_dim

    def build_linear(self, seq_len):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test = torch.zeros(1, 4, seq_len).to(device)
        y = self.conv(test)
        flattened = y.view(1, -1).shape[1]
        self._linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.latent_dim)
        )

    def forward(self, x):
        # x: (B, MAX_NOTES, 4)
        x = x.permute(0,2,1)  # -> (B, 4, MAX_NOTES)
        y = self.conv(x)
        if self._linear is None:
            self.build_linear(x.shape[-1])
            self._linear = self._linear.to(x.device)
            
        z = self._linear(y.view(y.size(0), -1))
        return z

class ConvDecoder(nn.Module):
    def __init__(self, out_channels=4, max_notes=512, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_notes = max_notes
        # We'll invert the encoder mapping by mapping latent -> flattened conv shape then transposed conv
        # To keep it simple, we'll reconstruct with linear layers -> reshape -> conv transpose
        # We'll assume the encoder reduced length by 2^3 = 8 (approx) if using same strides
        reduced_len = max(1, max_notes // 8)
        flattened = 256 * reduced_len  # must match encoder final conv channels * reduced_len
        self.pre = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, flattened),
            nn.ReLU(inplace=True)
        )
        # transpose conv stack to produce (B, out_channels, MAX_NOTES)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(64, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            # final output will be (B, out_channels, ~MAX_NOTES)
        )

    def forward(self, z):
        b = z.size(0)
        y = self.pre(z)
        # determine reduced_len from y
        # infer channels = 256
        # reshape to (B, 256, reduced_len)
        total = y.shape[1]
        # choose 256 channels, compute len
        if total % 256 != 0:
            # fallback: reshape to (B, 256, total//256)
            reduced_len = max(1, total // 256)
            y = y[:, :256 * reduced_len]
        else:
            reduced_len = total // 256
        y = y.view(b, 256, reduced_len)
        out = self.deconv(y)
        # out shape: (B, out_channels, approx MAX_NOTES) -> trim/pad to exact MAX_NOTES
        out = out.permute(0,2,1)  # (B, MAX_NOTES_approx, out_channels)
        # Ensure we have exactly max_notes by trimming or padding zeros
        if out.size(1) > self.max_notes:
            out = out[:, :self.max_notes, :]
        elif out.size(1) < self.max_notes:
            pad = torch.zeros(out.size(0), self.max_notes - out.size(1), out.size(2), device=out.device)
            out = torch.cat([out, pad], dim=1)
        return out  # (B, MAX_NOTES, out_channels)

class Autoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=4, latent_dim=cfg['LATENT_DIM'])
        self.decoder = ConvDecoder(out_channels=4, max_notes=cfg['MAX_NOTES'], latent_dim=cfg['LATENT_DIM'])

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z