import gym
import numpy as np
from datetime import datetime
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from env_template_test import EnergyTradingEnv
from new_envs_samples import DiscreteEnergyTradingEnv
import os
import sys
from pathlib import Path
from pyswarm import pso
from mealpy import GWO, FloatVar
import time

parent_dir = Path(__file__).resolve().parent.parent

# Add PSO directory to path
# PSO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RL_Weight_Initialization'))
# sys.path.append(PSO_DIR)
# from PSO_Weight_Init import PSOWeightInitializer # type: ignore
# from PSO_Weight_Init_1 import pso_objective # type: ignore

time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
hour_now = datetime.now().strftime("%H%M%S")


models_dir = parent_dir / "models" / "RL"
models_dir.mkdir(parents=True, exist_ok=True)



def train_DQN(prices, forecasts, experiment_id, uncertainties, initialize_weights=False, search_algo = None, pso_params=None):
    env = DiscreteEnergyTradingEnv(prices, forecasts, uncertainties)
    check_env(env, warn=True)

    log_dir = f"./logs/DQN_{search_algo}_{experiment_id}_{hour_now}"
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    env = Monitor(env, log_dir)
    if initialize_weights:
        model = DQN("MlpPolicy", env, verbose=1,
                    learning_rate=0.0001,  
                    buffer_size=1000000,   
                    batch_size=128,        
                    gamma=0.99,            
                    exploration_fraction=0.05,  
                    exploration_final_eps=0.02,  
                    target_update_interval=1000,  
                    train_freq=4,         
                    gradient_steps=1,
                    learning_starts=50000,  
                    policy_kwargs=dict(
                        net_arch=[256, 256],  
                        activation_fn=torch.nn.ReLU
                    ))
    else:
        model = DQN("MlpPolicy", env, verbose=1,
                    learning_rate=0.0001,  
                    buffer_size=1000000,   
                    batch_size=128,        
                    gamma=0.99,            
                    exploration_fraction=0.3,  
                    exploration_final_eps=0.02,  
                    target_update_interval=1000,  
                    train_freq=4,         
                    gradient_steps=1,
                    learning_starts=50000,  
                    policy_kwargs=dict(
                        net_arch=[256, 256],  
                        activation_fn=torch.nn.ReLU
                    ))
        
    model.set_logger(logger)

    def fitness(weights):
        params = model.policy.state_dict()
        flat_params = np.concatenate([p.flatten() for p in params.values()])
        if len(weights) != len(flat_params):
            raise ValueError("Weights and parameters have different shapes")
        new_params = {}
        i = 0
        for key in params:
            shape = params[key].shape
            size = np.prod(shape)
            new_params[key] = torch.tensor(weights[i:i+size].reshape(shape), dtype=torch.float32)
            i += size
        model.policy.load_state_dict(new_params)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)

        return (mean_reward,)
    
    def pso_objective(weights):
        return -fitness(weights)[0]
        
    if initialize_weights:
        print(f"Initializing weights with {search_algo}")
        num_params = sum(p.numel() for p in model.policy.parameters())
        params = model.policy.state_dict()
        if search_algo == 'pso':
            start_time_pso = datetime.now()
            best_weights, best_value = pso(
                pso_objective,
                lb=np.full(num_params,-1),
                ub=np.full(num_params,1),
                swarmsize=20,
                maxiter=100,
            )
            optimize_time = datetime.now() - start_time_pso 
            print("best reward: ", best_value)
            i = 0
            new_params = {}
            for key in params:
                shape = params[key].shape
                size = np.prod(shape)
                new_params[key] = torch.tensor(best_weights[i:i + size].reshape(shape), dtype=torch.float32)
                i += size
        elif search_algo == 'gwo':
            problem = {
                "bounds": FloatVar(lb=np.full(num_params,-1), ub=np.full(num_params,1), name = "weights"),
                "minmax": "min",
                "obj_func": pso_objective
            }
            start_time_gwo = datetime.now()
            optimizer = GWO.OriginalGWO(epoch=100, pop_size=20)
            best_solution = optimizer.solve(problem)
            best_position = best_solution.solution
            optimize_time = datetime.now() - start_time_gwo
            new_params = {}
            i = 0
            for key in params:
                shape = params[key].shape
                size = np.prod(shape)
                new_params[key] = torch.tensor(best_position[i: i + size].reshape(shape), dtype=torch.float32)
                i += size
                
        print("Loading weights into DQN model")
        model.policy.load_state_dict(new_params)
        print("Weights initialized")
    
    start_time_DQN = datetime.now()
    model.learn(total_timesteps=500000)
    training_time = datetime.now() - start_time_DQN

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=365)

    models_path = models_dir / f"dqn_model_{experiment_id}.zip"
    model.save((models_path))

    env.close()

    return model, training_time, optimize_time, mean_reward, std_reward

