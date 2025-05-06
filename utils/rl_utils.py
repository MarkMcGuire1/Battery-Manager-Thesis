import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path

def create_24h_preds(predictions, test_data):
    predictions_dict = {}
    real_prices_dict = {}
    for i in range(len(predictions)):
        start_time = test_data.index[i]
        forecast_range = pd.date_range(start=start_time, periods=24, freq='h')
        if start_time.hour == 23:
            formatted_key = start_time.strftime("%Y-%m-%d %H:%M")

            predictions_dict[formatted_key] = predictions[i,:].tolist()

            valid_forecast_times = [t for t in forecast_range if t in test_data.index]

            if valid_forecast_times:
                real_prices_dict[formatted_key] = test_data.loc[valid_forecast_times, test_data.columns[-1]].tolist()

    return predictions_dict, real_prices_dict         

def create_dict(data, value_column):
    d = {}
    for date, group in data.groupby(data.index.date):
        if len(group) == 24:
            key = pd.Timestamp(f"{date} 23:00")
            d[key] = group[value_column].values
    return d

def add_noise(data, column, noise_std = 0.05):
    noisy_data = data.copy()
    noise = np.random.normal(loc = 0, scale = noise_std * data[column].std(), size = len(data))
    noisy_data[column] += noise

    return noisy_data

def price_per_hour(data):
    data['price_euros_mwh'] = data['Price']
    data.drop(columns=['Price', 'LOAD_DA_FORECAST', 'Renewables_DA_Forecast', 'EUA', 'API2_COAL', 'TTF_GAS', 'Brent_oil'], inplace=True)
    return data


def reconfigure_index(data):
    
    data['DateTime'] = pd.to_datetime(data['date'] + ' ' + data['start_hour'], format='%d/%m/%Y %H:%M')
    data = data.set_index('DateTime')
    data.drop(columns=['date', 'start_hour', 'end_hour'], inplace=True)
    return data

def reconfigure_index_2(data):
    data['DateTime'] = pd.to_datetime(data.index)
    data.index = data['DateTime']
    return data

def log_evaluation_results(model, env, accumulated_profit, reward_log, results_dir, loss_log, epsilon_values, experiment_id):
    """
    Logs evaluation metrics (training time, performance, convergence) and saves plots for a DQN model.

    Parameters:
    - model: The trained RL model.
    - env: The evaluation environment.
    - accumulated_profit: List of accumulated profit per episode.
    - reward_log: List of reward values over time.
    - loss_log: List of loss values during training.
    - epsilon_values: List of epsilon values over training.
    - run_type: A string ("No Init" or "PSO Init") to differentiate runs.

    Saves:
    - Training time and performance metrics to a CSV file.
    - Training reward, loss, epsilon decay, and accumulated profit plots.
    """

    # Create a unique directory for storing results
    results_dir = results_dir / 'Model Evaluation' / experiment_id
    os.makedirs(results_dir, exist_ok=True)

    # Evaluate final policy performance
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=365)
    
    # Calculate sample efficiency (steps to reach 90% of max reward)
    def find_efficiency_threshold(reward_log, threshold=0.9):
        max_reward = max(reward_log)
        target_reward = threshold * max_reward
        for step, reward in enumerate(reward_log):
            if reward >= target_reward:
                return step
        return None

    steps_to_90pct_reward = find_efficiency_threshold(reward_log)

    # Save numerical evaluation results to CSV
    csv_file = os.path.join(results_dir, "evaluation_results.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Experiment ID", experiment_id])
        #writer.writerow(["Final Mean Reward", mean_reward])
        #writer.writerow(["Final Reward Std Dev", std_reward])
        writer.writerow(["Steps to 90% of Max Reward", steps_to_90pct_reward])

    # Generate and save plots
    def save_plot(data, xlabel, ylabel, title, filename, color="blue"):
        plt.figure()
        plt.plot(data, label=title, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(results_dir, filename))
        plt.close()

    # Plot and save results
    save_plot(reward_log, "Step", "Reward", "Reward Over Time", "reward_over_time.png", "red")
    # save_plot(loss_log, "Training Step", "Loss", "Loss Convergence", "loss_convergence.png", "green")
    # save_plot(epsilon_values, "Training Step", "Epsilon", "Exploration Decay", "epsilon_decay.png", "purple")
    save_plot(accumulated_profit, "Episode", "Total Profit", "Accumulated Profit", "accumulated_profit.png", "blue")

    print(f"Evaluation results saved in {results_dir}")


def log_training_results(model, training_time, optimize_time, mean_reward, std_reward, results_dir, experiment_id, model_name):
    results_dir = Path(results_dir) / f'Model Evaluation_{model_name}' / experiment_id
    os.makedirs(results_dir, exist_ok=True)
    model_path = results_dir / f"{model_name}_model_{experiment_id}.zip"
    model.save(model_path)
    csv_file = os.path.join(results_dir, f"{model_name}_training_results.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Training Time", training_time])
        writer.writerow(["Optimize Time", optimize_time])
        writer.writerow(["Mean Reward", mean_reward])
        writer.writerow(["Reward Std Dev", std_reward])

def set_global_seed(seed):
    import random
    import numpy as np
    import torch
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
