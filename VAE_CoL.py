# This framework uses generative trajectory modeling and latent optimization to shape early policy behavior,
# improving learning efficiency in sparse or risky environments.

from experiments.VAE.TrajVAE import TrajVAE
from experiments.VAE.PsoLatent import PSO_Opt_Latent
from experiments.Imitation_Learning.PSO_Opt import PSO_Opt, ActorNet, CriticNet
from experiments.Imitation_Learning.CoL import CoLTrainer, RelayBuffer
from env.modified_env import FixedSACTradingEnv
from utils.load_data import load_data
from utils.vae_utils import train_VAE
import pickle
import torch
import numpy as np

with open("data/expert_pso_rollouts_500_eps.pkl", "rb") as f:
    expert_rollouts = pickle.load(f)

prices_array, preds_mean, preds_std = load_data(year=2019)
env = FixedSACTradingEnv(prices_array, preds_mean, preds_std)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

vae = TrajVAE(obs_dim=obs_dim, action_dim=action_dim, latent_dim=16).to("cpu")
train_VAE(vae, expert_rollouts, batch_size=64, n_epochs=50)

pso = PSO_Opt_Latent(env, vae, swarm_size=20, max_iter=100)
latent_seqs = [pso.optimize()[0] for _ in range(20)]  

decoded_trajs = [vae.decode(torch.tensor(z, dtype=torch.float32).unsqueeze(0)).detach().squeeze(0) for z in latent_seqs] 

latent_buffer = RelayBuffer(capacity=10_000)

for traj in decoded_trajs:
    traj = traj.cpu().numpy()
    for t in range(len(traj) - 1):
        obs = traj[t, :obs_dim]
        act = traj[t, obs_dim:]
        next_obs = traj[t + 1, :obs_dim]
        reward = 0.0
        done = (t == len(traj) - 2)
        latent_buffer.add((obs, act, reward, next_obs, done))

agent_buffer = RelayBuffer(capacity=100_000)  

actor = ActorNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
critic = CriticNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
actor_target = ActorNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
critic_target = CriticNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')

col_trainer = CoLTrainer(env, actor, critic, actor_target, critic_target, latent_buffer, agent_buffer, lambda_bc=1.0, lambda_q=1.0, lambda_actor=1.0, lambda_reg=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu')
# Option to pretrain from the buffer
col_trainer.train()

# Save model
torch.save(col_trainer.actor.state_dict(), 'models/vae_col_actor.pth')