from utils.rl_utils import *
from utils.load_data import load_data
from env.modified_env import TradingEnv
from experiments.Stagnation_break.SAC_PSO_framework import *
from stable_baselines3 import SAC
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import pickle

seeds = [0, 42, 123, 777, 999]

for seed in seeds:
    set_global_seed(seed)

    prices_array, preds_mean, preds_std = load_data(2019)

    env = TradingEnv(prices_array, preds_mean, preds_std)
    env.reset(seed=seed)

    print("Creating baseline SAC model...")
    baseline_model = SAC('MlpPolicy', env, verbose=0, policy_kwargs=dict(net_arch=[64, 64]), seed=seed)
    start_time = datetime.now()
    baseline_model.learn(total_timesteps=100000, log_interval=1000)
    base_training_time = datetime.now() - start_time
    baseline_mean_reward, baseline_std_reward = evaluate_policy(baseline_model, env, n_eval_episodes=20)
    print(f"[Baseline] Final Reward: {baseline_mean_reward} ± {baseline_std_reward}")

    print("Creating SAC model...")
    model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))

    # Create callback with PSO integration
    callback = SequentialPSOCallback(env, check_freq=1000, eval_freq=500, pso_freq=2500)  # Increased pso_freq

    print("Starting training...")
    start_time = time.time()
    model.learn(total_timesteps=100000, callback=callback, log_interval=1000)
    training_time = time.time() - start_time
    print("Training complete.")

    distillation_events = callback.distillation_events
    sac_rewards = callback.sac_rewards
    pso_rewards = callback.pso_rewards
    action_differences = callback.action_differences
    timestamps = callback.timestamps

    with open('results_stagnation_break/sac_pso_results.pkl', 'wb') as f:
        pickle.dump({
            'sac_rewards': sac_rewards,
            'pso_rewards': pso_rewards,
            'distillation_events': distillation_events,
            'action_differences': action_differences
        }, f)

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
    experiment_id = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_seed_{seed}'
    results_dir = f"./results_stagnation_break/{experiment_id}"

    # Log Training Results
    log_training_results(
        model, training_time, optimize_time, sac_mean_reward, sac_std_reward, results_dir, experiment_id, "SAC_PSO",
    )

    all_results = {"pso": [], "baseline": []}

    all_results["pso"].append((experiment_id, training_time, optimize_time, sac_mean_reward, sac_std_reward))
    all_results["baseline"].append((experiment_id, base_training_time, None, baseline_mean_reward, baseline_std_reward))

pso_stat_anal = np.array(all_results["pso"])
baseline_stat_anal = np.array(all_results["baseline"])

with open('results_stagnation_break/statistical_analysis.txt', 'w') as f:
    f.write("Statistical Analysis of PSO and Baseline Results\n")
    f.write("=============================================\n")
    f.write(f"PSO Mean Reward: {np.mean(pso_stat_anal[:, 3])} ± {np.std(pso_stat_anal[:, 3])}\n")
    f.write(f"Baseline Mean Reward: {np.mean(baseline_stat_anal[:, 3])} ± {np.std(baseline_stat_anal[:, 3])}\n")
    f.write("\n")
    f.write("T-test Results:\n")
    t_stat, p_value = ttest_ind(pso_stat_anal[:, 3], baseline_stat_anal[:, 3])
    f.write(f"T-statistic: {t_stat}, P-value: {p_value}\n")
    f.write("\n")