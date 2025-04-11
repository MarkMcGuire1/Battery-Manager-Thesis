import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from stable_baselines3 import SAC
from env_template_test import EnergyTradingEnv
from modified_env import FixedSACTradingEnv
from new_env_test import DiscreteEnergyTradingEnv, SACTradingEnv
from rl_utils import *

from pathlib import Path

# Load the data
data = 'FR_2020.csv'
data_folder = 'data'
current_dir = Path(__file__).resolve().parent
file_path = current_dir.parent / data_folder / data

df = pd.read_csv('../data/FR.csv', index_col=0)
df = reconfigure_index_2(df)

data_2020 = df[df.index.year == 2020]
price_data_2020 = data_2020['Price'].values
n_hours = len(price_data_2020) 
assert n_hours % 24 == 0, 'Some days missing values'
price_data_2020 = price_data_2020.reshape(-1, 24)

forecasts_2020 = np.load('../data/prob_forecasts_2020.npz')
preds_mean = forecasts_2020['mean']
preds_std = forecasts_2020['std']

# Load the trained SAC model
targeted_experiment_id = f'sac_model_PSO_SAC_sequential_20250408_183818'
model_path = current_dir.parent / 'models' / targeted_experiment_id
results_dir = current_dir / 'Experimental files'

model = SAC.load(model_path)
print("SAC model loaded successfully")

# Check for NaNs in parameters
for name, param in model.policy.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN detected in parameter: {name}")

# Initialize environment
env = FixedSACTradingEnv(price_data_2020, preds_mean, preds_std, mode='Eval')

# Evaluation
n_episodes = 365
accumulated_profit = [0]
reward_log = []
#action_counts = {0: 0, 1: 0, 2: 0}
print("number of days in test set",len(price_data_2020))
for i in range(len(price_data_2020) - 1):
    
    print(i)
    env.eval_index = i
    obs, _ = env.reset()
    done = False
    episode_profit = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        #action_counts[action.item()] += 1
        obs, reward, done, _, info = env.step(action)
        reward_log.append(reward)
        #print(info)
        episode_profit += info['profit']

    accumulated_profit.append(accumulated_profit[-1] + episode_profit)

# print("Action counts:", action_counts)

# Log results
log_evaluation_results(model, env, accumulated_profit, reward_log, results_dir, [], [], targeted_experiment_id)

# Optional: plot results here if needed
# Plotting
plots_dir = Path('..') / 'plots'
plots_dir.mkdir(exist_ok=True)

# Accumulated Profit Plot
plt.figure()
plt.plot(accumulated_profit, label='Accumulated Profit', color='blue')
plt.xlabel("Test Episode")
plt.ylabel("Total Profit (€)")
plt.title("SAC Evaluation: Accumulated Profit Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'sac_eval_{targeted_experiment_id}.png'))
plt.close()

