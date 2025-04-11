from stable_baselines3 import SAC
from modified_env import FixedSACTradingEnv
import pandas as pd
import numpy as np
from rl_utils import *
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn.functional as F
import time
from pyswarm import pso
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

id = f'PSO_SAC_sequential_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

# Tracking progress
pso_rewards = []
sac_rewards = []
distillation_events = []
action_differences = []
timestamps = []

class SequentialPSOCallback(BaseCallback):
    def __init__(self, check_freq=1000, eval_freq=5000, pso_freq=10000, mse_thres = 0.1, patience = 3, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.eval_freq = eval_freq
        self.pso_freq = pso_freq
        self.mse_thres = mse_thres
        self.patience = patience
        self.no_improvement_count = 0
        self.start_time = time.time()
        self.best_weights = None
        self.best_reward = -np.inf
        print(f"[CALLBACK] Initialized with pso_freq={pso_freq}, eval_freq={eval_freq}")
        
        # Store environment and model references for PSO
        self._env = None
        self._model = None
    
    def _on_step(self):
        # Store environment and model references if not already stored
        if self._env is None:
            self._env = self.model.get_env()
        if self._model is None:
            self._model = self.model
        
        # Update global step counter
        global_step_count = self.n_calls
        # Periodically evaluate SAC policy
        if self.n_calls % self.eval_freq == 0:
            sac_mean_reward, _ = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=5)
            print(f"[SAC Evaluation] Step {self.n_calls}, Mean reward: {sac_mean_reward}")
            
            sac_rewards.append(sac_mean_reward)
            timestamps.append(self.n_calls)
            self._save_learning_curve()

            if sac_mean_reward > self.best_reward + 0.5:
                self.best_reward = sac_mean_reward
                self.no_improvement_count = 0
                print(f"[SAC] SAC policy improved to {sac_mean_reward}, Patience reset")
            else:
                self.no_improvement_count += 1
                print(f"[SAC] No improvement found: {self.no_improvement_count}")

        
        # # Run PSO optimization periodically
        # if self.n_calls % self.pso_freq == 0 and self.n_calls > 0:
        #     print(f"[PSO] Starting optimization at step {self.n_calls}")

        # Trigger PSO optimization if conditions are met
        if self.no_improvement_count > self.patience:    
            print(f"[PSO] Starting optimization at step {self.n_calls}")
            self.no_improvement_count = 0 # Reset counter
            # Count parameters in the ActorNet
            obs_dim = self._model.observation_space.shape[0]
            action_dim = self._model.action_space.shape[0]
            test_actor = ActorNet(obs_dim, action_dim)
            num_params = sum(p.numel() for p in test_actor.parameters())

            # Get actor weights
            actor = self.model.policy.actor

            # Get flattened weights
            flattened_weights = np.concatenate([param.data.cpu().numpy().flatten() for param in actor.parameters()])

            
            # Create a closure for the PSO objective
            def pso_objective(weights):
                return -self._evaluate_weights(weights)[0]
            
            try:
                # Run PSO optimization with adjusted parameters
                best_weights, best_value = pso(
                    pso_objective,
                    lb=np.full(num_params, -1.0),
                    ub=np.full(num_params, 1.0),
                    swarmsize=10,  
                    maxiter=10  
                )
            
                # Negate the value back since pso_obj negates fitness
                candidate_reward = -best_value
                print(f"[PSO] Completed with reward: {candidate_reward}")
    
                # Update best weights if improved
                if candidate_reward > self.best_reward:
                    self.best_reward = candidate_reward
                    self.best_weights = best_weights.copy()
                    print(f"[PSO] New best reward: {candidate_reward}")
                    pso_rewards.append((self.n_calls, candidate_reward))
                
                # Distill PSO policy into SAC if significant improvement
                if self.best_weights is not None:
                    self._distill_pso_policy()
            except Exception as e:
                print(f"Error in PSO optimization: {e}")
        return True
    
    def _evaluate_weights(self, weights):
        """Evaluate a set of weights using the current environment"""
        try:
            # For evaluation, create a custom policy and load weights
            obs_dim = self._env.observation_space.shape[0]
            action_dim = self._env.action_space.shape[0]
            
            # Create a temporary actor
            temp_actor = ActorNet(obs_dim, action_dim)
            
            # Load weights into the temp_actor
            state_dict = temp_actor.state_dict()
            new_state_dict = {}
            i = 0
            for key in state_dict:
                shape = state_dict[key].shape
                size = np.prod(shape)
                new_state_dict[key] = torch.tensor(weights[i:i+size].reshape(shape), dtype=torch.float32)
                i += size
            temp_actor.load_state_dict(new_state_dict)
            
            # Evaluate the policy
            reward_sum = 0
            n_episodes = 10
            
            for _ in range(n_episodes):
                done = False
                obs, _ = env.reset()
                episode_reward = 0
                
                # Track hour to detect boundary condition
                hour = 0
                
                while not done and hour < 24:  # Add explicit hour check
                    # Get action using the actor
                    with torch.no_grad():
                        action = temp_actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                        action = action.squeeze().numpy()
                    
                    # Clip action to environment limits
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    
                    try:
                        # Apply action
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        episode_reward += reward
                        hour += 1
                    except IndexError:
                        # Handle the boundary case
                        print(f"[Fitness] Caught boundary error at hour {hour}")
                        done = True
                
                reward_sum += episode_reward
            
            mean_reward = reward_sum / n_episodes
            print(f"[Fitness] Evaluated weights, mean reward: {mean_reward}")
            
            return (mean_reward,)
        
        except Exception as e:
            print(f"Error in fitness function: {e}")
            return (-np.inf,)

    def _distill_pso_policy(self):
        """Distill the best PSO policy into the SAC policy"""
        print("[Distillation] Starting policy distillation")
        
        # Sample random observations from the replay buffer
        replay_data = self.model.replay_buffer.sample(256)
        obs = torch.tensor(replay_data[0], dtype=torch.float32)
        
        # Get actions from current policy
        with torch.no_grad():
            sac_actions, _ = self.model.predict(replay_data[0], deterministic=True)
            sac_actions = torch.tensor(sac_actions, dtype=torch.float32)
        
        # Create PSO policy
        pso_policy = self._build_model_from_pso_weights(
            self.best_weights,
            input_dim=obs.shape[1],
            output_dim=self.model.action_space.shape[0]
        )
        
        # Get actions from PSO policy
        with torch.no_grad():
            pso_actions = pso_policy(obs)
        
        # Compute MSE between actions
        action_diff = F.mse_loss(pso_actions, sac_actions)
        print(f"[Distillation] Action difference MSE: {action_diff.item()}")
        action_differences.append(action_diff.item())
        
        # If actions are significantly different, distill PSO policy
        if action_diff.item() > self.mse_thres:
            print("[Distillation] Distilling PSO policy into SAC policy")
            
            # Extract actor model from SAC
            actor = self.model.actor
            
            # Create optimizer for actor if it doesn't exist
            if not hasattr(actor, 'optimizer'):
                actor.optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
            
            # Update actor to match PSO policy
            for _ in range(10):  # Multiple update steps
                current_actions = actor(obs)
                loss = F.mse_loss(current_actions, pso_actions)
                
                actor.optimizer.zero_grad()
                loss.backward()
                actor.optimizer.step()
            
            print(f"[Distillation] Distillation complete, final loss: {loss.item()}")
            distillation_events.append((self.n_calls, self.best_reward))

    def _build_model_from_pso_weights(self, pso_weights, input_dim, output_dim, hidden_dims=[64, 64]):
        model = ActorNet(input_dim, output_dim, hidden_dims)
        model_weights = self._unflatten_weights(model, pso_weights)
        model.load_state_dict(model_weights)
        return model
    
    def _unflatten_weights(self, model, flat_weights):
        flat_weights = torch.tensor(flat_weights, dtype=torch.float32)
        state_dict = model.state_dict()
        new_state_dict = {}
        i = 0
        for key, param in state_dict.items():
            num_params = param.numel()
            new_params = flat_weights[i:i + num_params].view(param.shape)
            new_state_dict[key] = new_params
            i += num_params
        return new_state_dict
    
    
    def _save_learning_curve(self):
        """Generate and save learning curve plot"""
        plt.figure(figsize=(10, 6))
        
        if sac_rewards:
            plt.plot(timestamps, sac_rewards, 'b-', label='SAC')
        
        if pso_rewards:
            pso_steps = [x[0] for x in pso_rewards]
            pso_reward_values = [x[1] for x in pso_rewards]
            plt.plot(pso_steps, pso_reward_values, 'r-', label='PSO')
        
        for step, reward in distillation_events:
            plt.axvline(x=step, color='g', linestyle='--', alpha=0.5)
        
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward')
        plt.title('Learning Curve: SAC vs PSO')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/learning_curve_{timestamp}.png"
        plt.savefig(filename)
        plt.close()

# ActorNet class remains the same
class ActorNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers.append(torch.nn.Linear(last_dim, dim))
            layers.append(torch.nn.ReLU())
            last_dim = dim
        self.hidden = torch.nn.Sequential(*layers)
        self.output = torch.nn.Linear(last_dim, output_dim)
    
    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)

# Setup environment and model
print("Setting up environment...")
data = pd.read_csv('../data/FR.csv', index_col=0)
data_reindexed = reconfigure_index_2(data)
data_2019 = data_reindexed[data_reindexed.index.year == 2019]
hourly_prices = data_2019['Price'].values
n_hours = len(hourly_prices)
n_days = n_hours // 24
assert n_hours % 24 == 0, 'Some days missing values'
prices_array = hourly_prices.reshape(n_days, 24)
preds_2019 = np.load('../data/prob_forecasts_2019.npz')
preds_mean = preds_2019['mean']
preds_std = preds_2019['std']

env = FixedSACTradingEnv(prices_array, preds_mean, preds_std)

print("Creating SAC model...")
model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))

# Create callback with PSO integration
callback = SequentialPSOCallback(check_freq=1000, eval_freq=500, pso_freq=2500)  # Increased pso_freq

print("Starting training...")
model.learn(total_timesteps=100000, callback=callback, log_interval=1000)
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