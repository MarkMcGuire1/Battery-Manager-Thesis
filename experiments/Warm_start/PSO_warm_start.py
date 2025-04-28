from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date
import os
import sys
import argparse

from rl_utils import reconfigure_index, create_dict, add_noise, log_training_results
from DQN_ import train_DQN
from SAC_ import train_SAC
from DDPG_ import train_DDPG


parent_dir = Path(__file__).resolve().parent.parent

# Add PSO directory to path
PSO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RL_Weight_Initialization'))
sys.path.append(PSO_DIR)
from PSO_param_opt import optimize_parameters # type: ignore
from PSO_Weight_Init_1 import pso_objective # type: ignore

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent for energy trading')
    parser.add_argument('--data_2019', type=str, default='FR_2019.csv', help='Training data file')
    parser.add_argument('--data_2020', type=str, default='FR_2020.csv', help='Validation data file')
    parser.add_argument('--predictions', type=str, default='prob_forecasts.npz', help='2020 probabilistic predictions')
    parser.add_argument('--predictions_2019', type=str, default='prob_forecasts_2019.npz', help = '2019 probabilistic predictions')
    parser.add_argument("--initialize_weights", action='store_true', help='Use PSO for weight initialization')
    parser.add_argument('--experiment_id', type=str, default=f"Experimental_runs_{date.today().strftime('%Y%m%d')}", help='Experiment ID')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # Configuration
    data_folder = 'data'

    # Setup paths
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir.parent / data_folder / args.data_2019
    file_path_2020 = current_dir.parent / data_folder / args.data_2020
    file_path_predictions = current_dir.parent / data_folder / args.predictions
    file_path_predictions_2019 = current_dir.parent / data_folder / args.predictions_2019
    results_dir = current_dir / 'Experimental files'

    # Load and preprocess training data
    print("\nLoading training data...")
    df = pd.read_csv(file_path)
    df = df.dropna(axis=1)
    print("Training data info:")
    print(df.info())

    # Load validation data
    print("\nLoading validation data...")
    df_2020 = pd.read_csv(file_path_2020)
    print("Validation data info:")
    print(df_2020.info())

    # Prepare data for training
    print("\nPreparing data...")
    df = reconfigure_index(df)
    noisy_df = add_noise(df, 'price_euros_mwh')

    # Load Prediction Data (mean and std) - 2020
    print('Loading prediction data')
    predictions = np.load(file_path_predictions)
    preds_mean = predictions['mean']
    preds_std = predictions['std']

    # Load Prediction Data (mean and std) - 2019
    predictions_2019 = np.load(file_path_predictions_2019)
    preds_mean_2019 = predictions_2019['mean']
    preds_std_2019 = predictions_2019['std']
    
    real_price_data = create_dict(df, 'price_euros_mwh')
    noisy_price_data = create_dict(noisy_df, 'price_euros_mwh')
    


    # Optimize PSO parameters
    # best_params = optimize_parameters(model, real_price_data, noisy_price_data, experiment_id)
    # best_params.to_csv(results_dir / experiment_id / 'PSO_best_params.csv', index=False)

    # Print data statistics
    print("\nData Statistics:")
    all_prices = np.array([prices for prices in real_price_data.values()])
    print(f"Price range: {all_prices.min():.2f} to {all_prices.max():.2f} €/MWh")
    print(f"Mean price: {all_prices.mean():.2f} €/MWh")
    print(f"Number of days: {len(real_price_data)}")

    # Train DQN model
    print("\nStarting DQN training...")
    DQN_model, training_time, optimize_time, mean_reward, std_reward = train_DQN(real_price_data, noisy_price_data, args.experiment_id, initialize_weights=args.initialize_weights, pso_params=None)
    DQN_model_GWO, training_time_DQN_GWO, optimize_time_DQN_GWO, mean_reward_DQN_GWO, std_reward_DQN_GWO = train_DQN(real_price_data, noisy_price_data, args.experiment_id, args.initialize_weights, search_algo='gwo', pso_params=None)
    #SAC_model_GWO, training_time_SAC_GWO, optimize_time_SAC_GWO, mean_reward_SAC_GWO, std_reward_SAC_GWO = train_SAC(real_price_data, noisy_price_data, args.experiment_id, args.initialize_weights, search_algo='gwo', pso_params=None)
    #SAC_model_PSO, training_time_SAC_PSO, optimize_time_SAC_PSO, mean_reward_SAC_PSO, std_reward_SAC_PSO = train_SAC(real_price_data, noisy_price_data, args.experiment_id, args.initialize_weights, search_algo='pso', pso_params=None)
    #DDPG_model_GWO, training_time_DDPG_GWO, optimize_time_DDPG_GWO, mean_reward_DDPG_GWO, std_reward_DDPG_GWO = train_DDPG(real_price_data, noisy_price_data, args.experiment_id, args.initialize_weights, search_algo='gwo', pso_params=None)
    #DDPG_model_PSO, training_time_DDPG_PSO, optimize_time_DDPG_PSO, mean_reward_DDPG_PSO, std_reward_DDPG_PSO = train_DDPG(real_price_data, noisy_price_data, args.experiment_id, args.initialize_weights, search_algo='pso', pso_params=None)

    log_training_results(training_time, optimize_time, mean_reward, std_reward, results_dir, args.experiment_id, "DQN_PSO")
    log_training_results(training_time_DQN_GWO, optimize_time_DQN_GWO, mean_reward_DQN_GWO, std_reward_DQN_GWO, results_dir, args.experiment_id, "DQN_GWO")
    #log_training_results(training_time_SAC_GWO, optimize_time_SAC_GWO, mean_reward_SAC_GWO, std_reward_SAC_GWO, results_dir, args.experiment_id, "SAC_GWO")
    #log_training_results(training_time_SAC_PSO, optimize_time_SAC_PSO, mean_reward_SAC_PSO, std_reward_SAC_PSO, results_dir, args.experiment_id, "SAC_PSO")
    #log_training_results(training_time_DDPG_GWO, optimize_time_DDPG_GWO, mean_reward_DDPG_GWO, std_reward_DDPG_GWO, results_dir, args.experiment_id, "DDPG_GWO")
    #log_training_results(training_time_DDPG_PSO, optimize_time_DDPG_PSO, mean_reward_DDPG_PSO, std_reward_DDPG_PSO, results_dir, args.experiment_id, "DDPG_PSO")
    # print(f"\nDQN training completed in {training_time}!")
    # print(f"Mean Reward: {mean_reward}, Std Dev: {std_reward}")

    # log_training_results(training_time, optimize_time, mean_reward, std_reward, results_dir, args.experiment_id)
else:
    # Configuration
    current_date = date.today().strftime("%Y%m%d")
    data = 'FR_2019.csv'
    data_2020 = 'FR_2020.csv'
    data_folder = 'data'
    experiment_id = f'DQN_PSO_weight_init_{current_date}'

    # Setup paths
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir.parent / data_folder / data
    file_path_2020 = current_dir.parent / data_folder / data_2020
    results_dir = current_dir / 'Experimental files'

    # Load and preprocess training data
    print("\nLoading training data...")
    df = pd.read_csv(file_path)
    df = df.dropna(axis=1)
    print("Training data info:")
    print(df.info())

    # Load validation data
    print("\nLoading validation data...")
    df_2020 = pd.read_csv(file_path_2020)
    print("Validation data info:")
    print(df_2020.info())

    # Prepare data for training
    print("\nPreparing data...")
    df = reconfigure_index(df)
    noisy_df = add_noise(df, 'price_euros_mwh')

    real_price_data = create_dict(df, 'price_euros_mwh')
    noisy_price_data = create_dict(noisy_df, 'price_euros_mwh')

    # Optimize PSO parameters
    # best_params = optimize_parameters(model, real_price_data, noisy_price_data, experiment_id)
    # best_params.to_csv(results_dir / experiment_id / 'PSO_best_params.csv', index=False)

    # Print data statistics
    print("\nData Statistics:")
    all_prices = np.array([prices for prices in real_price_data.values()])
    print(f"Price range: {all_prices.min():.2f} to {all_prices.max():.2f} €/MWh")
    print(f"Mean price: {all_prices.mean():.2f} €/MWh")
    print(f"Number of days: {len(real_price_data)}")

    # Train DQN model
    print("\nStarting DQN training...")
    DQN_model_GWO, training_time_DQN_GWO, optimize_time_DQN_GWO, mean_reward_DQN_GWO, std_reward_DQN_GWO = train_DQN(real_price_data, noisy_price_data, experiment_id, initialize_weights=True, search_algo='gwo', pso_params=None)
    SAC_model_GWO, training_time_SAC_GWO, optimize_time_SAC_GWO, mean_reward_SAC_GWO, std_reward_SAC_GWO = train_SAC(real_price_data, noisy_price_data, experiment_id, initialize_weights=True, search_algo='gwo', pso_params=None)
    SAC_model_PSO, training_time_SAC_PSO, optimize_time_SAC_PSO, mean_reward_SAC_PSO, std_reward_SAC_PSO = train_SAC(real_price_data, noisy_price_data, experiment_id, initialize_weights=True, search_algo='pso', pso_params=None)
    DDPG_model_GWO, training_time_DDPG_GWO, optimize_time_DDPG_GWO, mean_reward_DDPG_GWO, std_reward_DDPG_GWO = train_DDPG(real_price_data, noisy_price_data, experiment_id, initialize_weights=True, search_algo='gwo', pso_params=None)
    DDPG_model_PSO, training_time_DDPG_PSO, optimize_time_DDPG_PSO, mean_reward_DDPG_PSO, std_reward_DDPG_PSO = train_DDPG(real_price_data, noisy_price_data, experiment_id, initialize_weights=True, search_algo='pso', pso_params=None)

    log_training_results(training_time_DQN_GWO, optimize_time_DQN_GWO, mean_reward_DQN_GWO, std_reward_DQN_GWO, results_dir, experiment_id, "DQN_GWO")
    log_training_results(training_time_SAC_GWO, optimize_time_SAC_GWO, mean_reward_SAC_GWO, std_reward_SAC_GWO, results_dir, experiment_id, "SAC_GWO")
    log_training_results(training_time_SAC_PSO, optimize_time_SAC_PSO, mean_reward_SAC_PSO, std_reward_SAC_PSO, results_dir, experiment_id, "SAC_PSO")
    log_training_results(training_time_DDPG_GWO, optimize_time_DDPG_GWO, mean_reward_DDPG_GWO, std_reward_DDPG_GWO, results_dir, experiment_id, "DDPG_GWO")
    log_training_results(training_time_DDPG_PSO, optimize_time_DDPG_PSO, mean_reward_DDPG_PSO, std_reward_DDPG_PSO, results_dir, experiment_id, "DDPG_PSO")
