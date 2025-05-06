import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, DDPG, DQN
from env.modified_env import TradingEnv
from utils.load_data import load_data

base_dir = "results"  # Directory where DDPG_init, SAC_no_init etc. live
output_dir = "evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# Load evaluation data
prices, predictions, uncertainties = load_data(year=2020)
env_cont = TradingEnv(prices, predictions, uncertainties)
env_discrete = TradingEnv(prices, predictions, uncertainties, action_type='discrete')

def warm_start_eval(algo_map, action_type):
    for algo_dir in os.listdir(base_dir):
        algo_path = os.path.join(base_dir, algo_dir)
        if not os.path.isdir(algo_path):
            continue

        algo_class = None
        for name, cls in algo_map.items():
            if name in algo_dir.upper():
                algo_class = cls
                break

        if algo_class is None:
            print(f"Skipping {algo_dir} (unrecognized algorithm)")
            continue

        print(f"Evaluating {algo_dir} with {algo_class.__name__}")

        zip_files = [f for f in os.listdir(algo_path) if f.endswith(".zip")]
        if zip_files:
            for zip_file in zip_files:
                model_path = os.path.join(algo_path, zip_file)
                model_name = f"{algo_dir}_{zip_file.replace('.zip', '')}"
        else:
            model_path = None
            model_name = algo_dir

            for subdir, _, files in os.walk(algo_path):
                if "policy.pth" in files:
                    model_path = subdir
                    break
            if not model_path:
                print(f"No valid model found in {algo_path}")
                continue
        try:
            if action_type == 'discrete':
                env = env_discrete
            else:
                env = env_cont
            model = algo_class.load(model_path, env=env)
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            continue

        accumulated_profit = [0]
        reward_log = []

        for i in range(len(prices) - 1):
            env.eval_index = i
            obs, _ = env.reset()
            done = False
            episode_profit = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                reward_log.append(reward)
                episode_profit += info['profit']

            accumulated_profit.append(accumulated_profit[-1] + episode_profit)

        # Save plot
        plot_path = os.path.join(output_dir, f"{model_name}_profit.png")
        plt.figure()
        plt.plot(accumulated_profit, label='Accumulated Profit (€)')
        plt.xlabel("Test Episode")
        plt.ylabel("Total Profit (€)")
        plt.title(model_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"    Saved plot to: {plot_path}")

        # Save data
        results_df = pd.DataFrame({
            "Episode": list(range(len(accumulated_profit))),
            "Accumulated_Profit": accumulated_profit,
        })

        reward_df = pd.DataFrame({
            "Step": list(range(len(reward_log))),
            "Reward": reward_log
        })

        results_df.to_csv(os.path.join(output_dir, f"{model_name}_accumulated_profit.csv"), index=False)
        reward_df.to_csv(os.path.join(output_dir, f"{model_name}_step_rewards.csv"), index=False)
        print(f"    Saved data for {model_name}")


def stagnation_break_eval(action_type='continuous'):
    summary_records = []  # To store summary stats for each model

    # Evaluate standalone zipped models
    zipped_model_dir = "results_stagnation_break"  
    for file in os.listdir(zipped_model_dir):
        if file.endswith(".zip") and "SAC_PSO_model" in file:
            model_path = os.path.join(zipped_model_dir, file)
            model_name = file.replace(".zip", "")
            print(f"Evaluating zipped model: {model_name}")

            try:
                if action_type == 'discrete':
                    env = env_discrete
                else:
                    env = env_cont
                model = SAC.load(model_path, env=env)
            except Exception as e:
                print(f"Failed to load zipped model {file}: {e}")
                continue

            accumulated_profit = [0]
            reward_log = []

            for i in range(len(prices) - 1):
                env.eval_index = i
                obs, _ = env.reset()
                done = False
                episode_profit = 0

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    reward_log.append(reward)
                    episode_profit += info['profit']

                accumulated_profit.append(accumulated_profit[-1] + episode_profit)

            # Save plot
            plot_path = os.path.join(output_dir, f"{model_name}_profit.png")
            plt.figure()
            plt.plot(accumulated_profit, label='Accumulated Profit (€)')
            plt.xlabel("Test Episode")
            plt.ylabel("Total Profit (€)")
            plt.title(model_name)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"    Saved plot to: {plot_path}")

            # Save data
            results_df = pd.DataFrame({
                "Episode": list(range(len(accumulated_profit))),
                "Accumulated_Profit": accumulated_profit,
            })
            reward_df = pd.DataFrame({
                "Step": list(range(len(reward_log))),
                "Reward": reward_log
            })

            results_df.to_csv(os.path.join(output_dir, f"{model_name}_accumulated_profit.csv"), index=False)
            reward_df.to_csv(os.path.join(output_dir, f"{model_name}_step_rewards.csv"), index=False)

            # Collect summary
            summary_records.append({
                "Model": model_name,
                "Final_Profit": accumulated_profit[-1],
                "Mean_Reward": sum(reward_log) / len(reward_log),
                "Max_Reward": max(reward_log),
                "Min_Reward": min(reward_log)
            })

    # Save summary CSV for all evaluated models
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(output_dir, "summary_stag_break.csv"), index=False)
    print("Summary saved to summary.csv")
    
algo_map = {
    "SAC": SAC,
    "DDPG": DDPG,
    }

algo_map_discrete = {
    "DQN": DQN,
    }

warm_start_eval(algo_map, action_type='continuous')
warm_start_eval(algo_map_discrete, action_type='discrete')
stagnation_break_eval()