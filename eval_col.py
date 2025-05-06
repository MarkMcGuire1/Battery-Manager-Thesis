import torch
from experiments.Imitation_Learning.CoL import ActorNet
from env.modified_env import TradingEnv
from utils.load_data import load_data
import matplotlib.pyplot as plt
import os
import numpy as np
from stable_baselines3 import SAC
import pickle
import json

with open('data/expert_pso_rollouts_500_eps.pkl', 'rb') as f:
    expert_trajs = pickle.load(f)


def action_mse(actor, expert_trajs):
    total_mse = 0
    count = 0
    for obs, expert_action, _, _, _ in expert_trajs:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to('cpu')
            action = actor(obs_tensor).cpu().numpy()[0]
        mse = np.mean((action - expert_action) ** 2)
        total_mse += mse
        count += 1
    return total_mse / count


model_paths = {
    'Full_CoL': 'models/col_actor_Full_CoL.pth',
    'BC_only': 'models/col_actor_BC_only.pth',
    'BC_Q': 'models/col_actor_BC_Q.pth',
    'Q_actor': 'models/col_actor_Q_actor.pth',
    'SAC_base': 'models/sac_model_baseline.zip',
}

accumulated_profits = {}
mse_log = {}

prices, predictions, uncertainties = load_data(year=2020)

env = TradingEnv(prices, predictions, uncertainties)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = ActorNet(obs_dim, action_dim).to('cpu')

for name, model_path in model_paths.items():
    if name == 'SAC_base':
        model = SAC.load(model_path, env=env)
    else:
        actor.load_state_dict(torch.load(model_path))
        actor.eval()
        mse = action_mse(actor, expert_trajs)
        mse_log[name] = mse
    # Evaluation
    accumulated_profit = [0]
    reward_log = []
    #action_counts = {0: 0, 1: 0, 2: 0}
    print("number of days in test set",len(prices))
    for i in range(len(prices) - 1):
        env.eval_index = i
        obs, _ = env.reset()
        done = False
        episode_profit = 0

        while not done:
            with torch.no_grad():
                if name == 'SAC_base':
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to('cpu')
                    action = actor(obs_tensor).cpu().numpy()[0]
            #action_counts[action.item()] += 1
            obs, reward, done, _, info = env.step(action)
            reward_log.append(reward)
            #print(info)
            episode_profit += info['profit']

        accumulated_profit.append(accumulated_profit[-1] + episode_profit)
    accumulated_profits[name] = accumulated_profit

results_dir = 'results_col'
os.makedirs(results_dir, exist_ok=True)
print('plotting profits')
fig = plt.figure()
for name, profits in accumulated_profits.items():
    plt.plot(profits, label=name)
    np.save(os.path.join(results_dir, f'accumulated_profit_{name}.npy'), profits)
    fig.savefig(os.path.join(results_dir, f'accumulated_profit_{name}.png'))
    np.save(os.path.join(results_dir, f'reward_log_col_{name}.npy'), reward_log)
plt.xlabel("Test Episode")
plt.ylabel("Total Profit (€)")
plt.title("CoL Evaluation: Accumulated Profit Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

with open(os.path.join(results_dir, 'mse_log.json'), 'w') as f:
    json.dump(mse_log, f, indent=2)