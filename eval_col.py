import torch
from experiments.Imitation_Learning.CoL import ActorNet
from env.modified_env import TradingEnv
from utils.load_data import load_data
import matplotlib.pyplot as plt
import os
import numpy as np

prices, predictions, uncertainties = load_data(year=2020)

env = TradingEnv(prices, predictions, uncertainties)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = ActorNet(obs_dim, action_dim).to('cpu')

actor.load_state_dict(torch.load('models/col_actor.pth'))
actor.eval()

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
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to('cpu')
            action = actor(obs_tensor).cpu().numpy()[0]
        #action_counts[action.item()] += 1
        obs, reward, done, _, info = env.step(action)
        reward_log.append(reward)
        #print(info)
        episode_profit += info['profit']

    accumulated_profit.append(accumulated_profit[-1] + episode_profit)

print('plotting profits')
fig = plt.figure()
plt.plot(accumulated_profit, label='Accumulated Profit', color='blue')
plt.xlabel("Test Episode")
plt.ylabel("Total Profit (€)")
plt.title("CoL Evaluation: Accumulated Profit Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

results_dir = 'results_col'
os.makedirs(results_dir, exist_ok=True)
np.save(os.path.join(results_dir, 'accumulated_profit_col.npy'), accumulated_profit)
fig.savefig(os.path.join(results_dir, 'accumulated_profit_col.png'))
np.save(os.path.join(results_dir, 'reward_log_col.npy'), reward_log)