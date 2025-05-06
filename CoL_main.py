from env.modified_env import TradingEnv
from utils.rl_utils import *
from utils.load_data import load_data
from stable_baselines3 import SAC
import pickle
from experiments.Imitation_Learning.PSO_Opt import PSO_Opt, ActorNet, CriticNet
from experiments.Imitation_Learning.CoL import CoLTrainer, RelayBuffer
import torch
import numpy as np
import pandas as pd
from time import time


prices_array, preds_mean, preds_std = load_data(year=2019)

env = TradingEnv(prices_array, preds_mean, preds_std)
model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))
obs_dim = model.observation_space.shape[0]
action_dim = model.action_space.shape[0]
test_actor = ActorNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
num_params = sum(p.numel() for p in test_actor.parameters())

swarm_size = 20
max_iter = 100

#start_time = time()
pso_optimizer = PSO_Opt(env, swarm_size, max_iter, num_params)
best_weights, _ = pso_optimizer.optimize()

mean_reward, expert_trajs = pso_optimizer._evaluate_weights(best_weights, n_episodes = 500, return_trajs=True, seed_reset=True)
with open('data/expert_pso_rollouts_500_eps.pkl', 'wb') as f:
    pickle.dump(expert_trajs, f)

run_CoL = True

if run_CoL:
    expert_buffer = RelayBuffer(capacity=10_000)
    for r in expert_trajs:
        expert_buffer.add(r)
    agent_buffer = RelayBuffer(capacity=100_000)

    env = TradingEnv(prices_array, preds_mean, preds_std)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = ActorNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
    critic = CriticNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
    actor_target = ActorNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
    critic_target = CriticNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    trainer = CoLTrainer(env, actor, critic, actor_target, critic_target, expert_buffer, agent_buffer, lambda_bc=1.0, lambda_q=1.0, lambda_actor=1.0, lambda_reg=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.train(total_steps=100000, log_interval=1000)
    #training_time = time() - start_time

    #mean_reward, std_reward = evaluate_policy(trainer, env, n_eval_episodes=20)

    torch.save(actor.state_dict(), 'models/col_actor.pth')

    # experiment_id = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # results_folder = f"./results/CoL_{experiment_id}"

    # log_training_results(trainer, training_time, None, mean_reward, std_reward, results_folder, experiment_id, "CoL",)

else:
    print('expert rollouts saved to data/expert_pso_rollouts.pkl')