import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from stable_baselines3 import DQN
from env_template_test import EnergyTradingEnv
from rl_utils import reconfigure_index, create_dict, add_noise, log_evaluation_results

from pathlib import Path

data = 'FR.csv'

data_folder = 'data'

current_dir = Path(__file__).resolve().parent
file_path = current_dir.parent / data_folder / data

df = pd.read_csv(file_path)
df = df.dropna(axis=1)

df = reconfigure_index(df)

#real_price_data = create_dict(df, 'price_euros_mwh')
noisy_price = add_noise(df, 'price_euros_mwh')
noisy_price_data = create_dict(noisy_price, 'price_euros_mwh')
price_data_2020 = create_dict(df, 'price_euros_mwh')

targeted_experiment_id = f'SAC_PSO_concurrent_{data.split(".")[0]}_2020'
model_path = current_dir.parent/ 'models' / 'RL' / f'dqn_model_{targeted_experiment_id}'
results_dir = current_dir / 'Experimental files'

model = DQN.load(model_path)
print("model loaded sucessfully")
for name, param in model.q_net.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN detected in parameter: {name}")

env = EnergyTradingEnv(price_data_2020, noisy_price_data, algo = 'DQN')

n_episodes = 365
accumulated_profit = [0]
reward_log = []
action_counts = {0: 0, 1: 0, 2: 0}
for _ in range(n_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    episode_profit = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_counts[action.item()] += 1
        obs, reward, done, _, info = env.step(action)
        reward_log.append(reward)
        print(info)
        episode_profit += info['profit']
    accumulated_profit.append(accumulated_profit[-1] + episode_profit)
print("action counts:", action_counts)
        # obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        # q_values = model.q_net(obs_tensor).detach().numpy()
        # print(f"Hour {env.hour}, Q-values: {q_values}")

        #print(reward)
        #total_reward += reward
        #obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        #q_values = model.q_net(obs_tensor).detach().numpy()

        # print("Q-values for each action:", q_values)
        # print("Action chosen:", np.argmax(q_values))
#     rewards_list.append(total_reward)
#     print(f"Episode {episode + 1} Total Reward: {total_reward}")

# print(f"Mean Reward: {np.mean(rewards_list)}, Std Dev: {np.std(rewards_list)}")

log_evaluation_results(model, env, accumulated_profit, reward_log, results_dir, [], [], targeted_experiment_id)



# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(accumulated_profit, label='accumulated profit', color='blue')
# plt.xlabel("Test Episode")
# plt.ylabel("Total Profit")
# plt.title("DQN Evaluation Performance")
# plt.legend()
# plt.savefig(os.path.join(plots_dir, f'dqn_eval_{targeted_experiment_id}.png'))
# plt.close()

# plt.figure()
# plt.plot(reward_log, label='reward', color='red')
# plt.xlabel("Step")
# plt.ylabel("Reward")
# plt.title("Reward over Time")
# plt.savefig(os.path.join(plots_dir, f'dqn_reward_{targeted_experiment_id}.png'))
# plt.close()