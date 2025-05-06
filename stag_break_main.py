from utils.rl_utils import *
from utils.load_data import load_data
from env.modified_env import TradingEnv
from experiments.Stagnation_break.SAC_PSO_framework import *
from stable_baselines3 import SAC
from time import time
import matplotlib.pyplot as plt


prices_array, preds_mean, preds_std = load_data(2019)

env = TradingEnv(prices_array, preds_mean, preds_std)

print("Creating SAC model...")
model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))

# Create callback with PSO integration
callback = SequentialPSOCallback(env, check_freq=1000, eval_freq=500, pso_freq=2500)  # Increased pso_freq

print("Starting training...")
start_time = time()
model.learn(total_timesteps=100000, callback=callback, log_interval=1000)
training_time = time() - start_time
print("Training complete.")

# Save model
model.save(f'../models/sac_model_{id}')
print('Model saved.')

# Final evaluation
print("Performing final evaluation...")
sac_mean_reward, sac_std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Final SAC Policy Mean Reward: {sac_mean_reward} ± {sac_std_reward}")

# Generate final plots
plt.figure(figsize=(12, 8))

# Plot SAC rewards
if sac_rewards:
    plt.plot(timestamps, sac_rewards, 'b-', label='SAC')

# Plot PSO rewards
if pso_rewards:
    pso_steps = [x[0] for x in pso_rewards]
    pso_reward_values = [x[1] for x in pso_rewards]
    plt.plot(pso_steps, pso_reward_values, 'r-', label='PSO')

# Mark distillation events
for event in distillation_events:
    plt.axvline(x=event[0], color='g', linestyle='--', alpha=0.5, 
                label='Distillation' if event == distillation_events[0] else '')

plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title('Comparison of SAC and PSO Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'logs/final_comparison_{id}.png')
plt.show()

# Save action differences plot
if action_differences:
    plt.figure(figsize=(10, 5))
    plt.plot(action_differences)
    plt.xlabel('Check Frequency')
    plt.ylabel('Action MSE')
    plt.title('Policy Difference Over Time (Action MSE)')
    plt.grid(True, alpha=0.3)
    plt.savefig('logs/action_differences.png')
    plt.close()

print("Training complete. Check the logs directory for performance plots.")

optimize_time = None
experiment_id = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
results_dir = f"./results_stagnation_break/{experiment_id}"

# Log Training Results
log_training_results(
    model, training_time, optimize_time, sac_mean_reward, sac_std_reward, results_dir, experiment_id, "SAC_PSO",
)