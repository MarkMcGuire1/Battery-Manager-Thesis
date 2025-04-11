# This framework uses generative trajectory modeling and latent optimization to shape early policy behavior,
# improving learning efficiency in sparse or risky environments.

from experiments.VAE.TrajVAE import TrajVAE
from experiments.Imitation_Learning.PSO_Opt import PSO_Opt, ActorNet, CriticNet
from experiments.Imitation_Learning.CoL import CoLTrainer, RelayBuffer
from env.modified_env import FixedSACTradingEnv
from utils.load_data import load_data
from utils.vae_utils import train_VAE
import pickle
import torch
import numpy as np

with open("data/expert_pso_rollouts.pkl", "rb") as f:
    expert_rollouts = pickle.load(f)

prices_array, preds_mean, preds_std = load_data(year=2019)
env = FixedSACTradingEnv(prices_array, preds_mean, preds_std)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

vae = TrajVAE(obs_dim=obs_dim, action_dim=action_dim, latent_dim=16).to("cpu")
train_VAE(vae, expert_rollouts, batch_size=64, n_epochs=50)

pso = PSO_Opt(env, swarm_size=20, max_iter=100, num_params=vae.latent_dim)
latent_seqs = pso.optimize(vae, num_particles=100, num_iterations=1000)

decoded_trajs = [vae.decode(z_seq) for z_seq in latent_seqs]

latent_buffer = RelayBuffer(capacity=100_000)

for traj in decoded_trajs:
    latent_buffer.add(traj)  

col_trainer = CoLTrainer()
# Option to pretrain from the buffer
col_trainer.train()

# Save model
torch.save(col_trainer.actor.state_dict(), 'models/vae_col_actor.pth')