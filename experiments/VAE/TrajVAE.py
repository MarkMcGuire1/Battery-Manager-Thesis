import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajVAE(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim = [64, 64], sequence_len=20):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sequence_len = sequence_len
        self.input_dim = obs_dim + action_dim

        # Encoder: RNN-based encoder to process the input sequence
        self.encoder_rnn = torch.nn.GRU(self.input_dim, hidden_dim[0], batch_first=True)
        self.encoder_fc_mu = torch.nn.Linear(hidden_dim[0], latent_dim)
        self.decoder_logvar = torch.nn.Linear(hidden_dim[0], latent_dim)

        # Decoder: RNN-based decoder to reconstruct the input sequence
        self.decoder_input = torch.nn.Linear(latent_dim, hidden_dim[0])
        self.decoder_rnn = torch.nn.GRU(hidden_dim[0], hidden_dim[0], batch_first=True)
        self.decoder_fc = torch.nn.Linear(hidden_dim[0], self.input_dim)

    def encode(self, x):
        _, h_n = self.encoder_rnn(x)
        h = h_n.squeeze(0)
        mu = self.encoder_fc_mu(h)
        logvar = self.decoder_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z).unsqueeze(1)
        h = h.repeat(1, self.sequence_len, 1)
        out, _ = self.decoder_fc(h)
        reconstructions = self.decoder_fc(out)
        return reconstructions
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructions = self.decode(z)
        return reconstructions, mu, logvar
    
    def compute_loss(self, reconstructions, target, mu, logvar):
        reconstruction_loss = F.mse_loss(reconstructions, target, reduction='mean')
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + KLD, reconstruction_loss, KLD