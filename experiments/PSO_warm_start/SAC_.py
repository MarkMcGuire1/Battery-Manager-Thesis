from stable_baselines3 import SAC
from pyswarm import pso
from mealpy import GWO, FloatVar
import torch
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime
from pathlib import Path


parent_dir = Path(__file__).resolve().parent.parent

time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
hour_now = datetime.now().strftime("%H%M%S")


models_dir = parent_dir / "models" / "RL"
models_dir.mkdir(parents=True, exist_ok=True)

def train_SAC(experiment_id, env, initialize_weights=False, search_algo = None):
    check_env(env, warn=True)

    log_dir = f"./logs/SAC_{search_algo}_{experiment_id}_{hour_now}"
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    env = Monitor(env, log_dir)

    
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

    if not initialize_weights:
        model = SAC("MlpPolicy", env = DummyVecEnv([lambda: env]), verbose = 1)
    else:
        model = SAC("MlpPolicy", env = DummyVecEnv([lambda: env]), verbose = 1)
        num_params = sum(p.numel() for p in model.policy.parameters())

        if search_algo == 'pso':
            start_time = datetime.now()
            best_weights, _ = pso(
                pso_objective,
                lb=np.full(num_params,-1),
            ub=np.full(num_params,1),
            swarmsize=20,
            maxiter=100,
        )
            
        elif search_algo == 'gwo':
            problem = {
                "bounds": FloatVar(lb=np.full(num_params,-1), ub=np.full(num_params,1), name = "weights"),
                "minmax": "min",
                "obj_func": pso_objective
            }
            start_time = datetime.now()
            optimizer = GWO.OriginalGWO(epoch=100, pop_size=20)
            best = optimizer.solve(problem)
            best_weights = best.solution

                
        optimize_time = datetime.now() - start_time 
        params = model.policy.state_dict()
        i = 0
        new_params = {}
        for key in params:
            shape = params[key].shape
            size = np.prod(shape)
            new_params[key] = torch.tensor(best_weights[i:i + size].reshape(shape), dtype=torch.float32)
            i += size

    model.set_logger(logger)

    if initialize_weights:
        print("Loading weights into DQN model")
        model.policy.load_state_dict(new_params)
        print("Weights initialized")
    else:
        optimize_time = 0
    start_time_SAC = datetime.now()
    model.learn(total_timesteps=240000)
    training_time = datetime.now() - start_time_SAC

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=365)

    models_path = models_dir / f"sac_model_{experiment_id}.zip"
    model.save(models_path)

    env.close()

    return model, training_time, optimize_time, mean_reward, std_reward


