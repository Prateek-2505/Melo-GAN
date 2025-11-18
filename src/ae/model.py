import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=4, latent_dim=128, hidden_dim=512):
        '''
        Encoder. Outputs a hidden vector, not the final latent params.
        '''
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self._linear = None

    def build_linear(self, seq_len):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test = torch.zeros(1, 4, seq_len).to(device)
        y = self.conv(test)
        flattened = y.view(1, -1).shape[1]
        self._linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, self.hidden_dim),
            nn.ReLU(inplace=True)
        ).to(device)

    def forward(self, x):
        # x: (B, MAX_NOTES, 4)
        x = x.permute(0,2,1)  # -> (B, 4, MAX_NOTES)
        y = self.conv(x)
        if self._linear is None:
            self.build_linear(x.shape[-1])
            self._linear = self._linear.to(x.device)
            
        # Get the final hidden state
        h = self._linear(y.view(y.size(0), -1))
        return h

class ConvDecoder(nn.Module):
    def __init__(self, out_channels=4, max_notes=512, latent_dim=128, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_notes = max_notes
        self.hidden_dim=hidden_dim
        reduced_len = max(1, max_notes // 8)
        flattened = 128 * reduced_len
        
        self.pre = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, flattened),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(32, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh() # Add Tanh to match normalized data range [-1, 1]
        )

    def forward(self, z):
        b = z.size(0)
        y = self.pre(z)
        total = y.shape[1]
        
        if total % 128 != 0:
            reduced_len = max(1, total // 128)
            y = y[:, :128 * reduced_len]
        else:
            reduced_len = total // 128
            
        y = y.view(b, 128, reduced_len)
        out = self.deconv(y)
        out = out.permute(0,2,1)
        
        if out.size(1) > self.max_notes:
            out = out[:, :self.max_notes, :]
        elif out.size(1) < self.max_notes:
            pad = torch.zeros(out.size(0), self.max_notes - out.size(1), out.size(2), device=out.device)
            out = torch.cat([out, pad], dim=1)
        return out

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE)
    Replaces the standard Autoencoder to prevent posterior collapse.
    """
    def __init__(self, cfg):
        super().__init__()
        latent_dim = cfg['LATENT_DIM']
        hidden_dim = 512 # Or read from cfg if available
        
        self.encoder = ConvEncoder(
            in_channels=4, 
            latent_dim=latent_dim, 
            hidden_dim=hidden_dim
        )
        
        # VAE-specific layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = ConvDecoder(
            out_channels=4, 
            max_notes=cfg['MAX_NOTES'], 
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )

    def reparameterize(self, mu, log_var):
        """
        The reparameterization trick to allow backpropagation
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # eps ~ N(0, 1)
        return mu + eps * std

    def forward(self, x):
        # 1. Encode
        h = self.encoder(x)
        
        # 2. Get latent distribution parameters
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        # 3. Sample from distribution
        z = self.reparameterize(mu, log_var)
        
        # 4. Decode
        recon = self.decoder(z)
        
        return recon, z, mu, log_var