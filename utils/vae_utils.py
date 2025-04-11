import numpy as np
import torch

def build_vae_sequences(trans, seq_len):
    seq = []
    curr = []

    for obs, action, _, _, done in trans:
        obs_action = np.concatenate([obs, action])
        curr.append(obs_action)
        if len(curr) == seq_len:
            seq.append(np.array(curr))
            curr = []

        if done:
            curr = []

    return seq

def train_VAE(vae, expert_rollouts, batch_size, n_epochs):
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae = vae.to("cpu")
    vae.train()

    vae_seq = build_vae_sequences(expert_rollouts, seq_len=24)
    tensor_data = torch.tensor(vae_seq, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = torch.utils.data.DataLoader(tensor_data, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        total_loss, total_recon, total_kl  = 0, 0, 0
        for batch in dataloader:
            opt.zero_grad()
            recon, mu, logvar = vae(batch)
            loss, recon_loss, kl_loss = vae.compute_loss(recon, batch, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(dataloader)}, Recon Loss: {total_recon/len(dataloader)}, KL Loss: {total_kl/len(dataloader)}")
