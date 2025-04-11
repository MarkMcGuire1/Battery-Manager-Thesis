import torch
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from env_template_test import EnergyTradingEnv
from datetime import datetime
from pathlib import Path


parent_dir = Path(__file__).resolve().parent.parent

time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
hour_now = datetime.now().strftime("%H%M%S")


models_dir = parent_dir / "models" / "RL"
models_dir.mkdir(parents=True, exist_ok=True)

def train_A2C(price_dict, forecast_dict, experiment_id, initialize_weights=False, pso_params=None):
    env = EnergyTradingEnv(price_dict, forecast_dict, algo = 'A2C')
    check_env(env, warn=True)

    log_dir = f"./logs/{experiment_id}_{hour_now}"
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    env = Monitor(env, log_dir)

    if not initialize_weights:
        model = A2C("MlpPolicy", env = DummyVecEnv([lambda: env]), verbose = 1)
    else:
        model = A2C("MlpPolicy", env = DummyVecEnv([lambda: env]), verbose = 1)
