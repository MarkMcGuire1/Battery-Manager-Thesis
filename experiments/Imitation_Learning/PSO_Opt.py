from pyswarm import pso
import numpy as np
import torch
import torch.nn.functional as F

class PSO_Opt:
    def __init__(self, env, swarm_size, max_iter, num_params):
        self._env = env
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.num_params = num_params
        self.device = 'cpu'

    def optimize(self):
        # Define the objective function for PSO
        def pso_obj(weights):
            return - self._evaluate_weights(weights)[0]  
        # Run PSO
        best_weights, best_value = pso(
            pso_obj,
            lb=np.full(self.num_params, -1.0),
            ub=np.full(self.num_params, 1.0),
            swarmsize=self.swarm_size,  
            maxiter=self.max_iter  
                )
        
        return best_weights, best_value
    
    def _evaluate_weights(self, weights, n_episodes = 10, return_trajs = False, seed_reset = False):
        """Evaluate a set of weights using the current environment"""
        # For evaluation, create a custom policy and load weights
        obs_dim = self._env.observation_space.shape[0]
        action_dim = self._env.action_space.shape[0]
        
        # Create a temporary actor
        temp_actor = ActorNet(obs_dim, action_dim).to(self.device)
        
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
        all_trajs = []
        
        for _ in range(n_episodes):
            done = False
            if seed_reset:
                seed = np.random.randint(0, 1e6)
                self._env.seed(seed)
            obs, _ = self._env.reset()
            episode_reward = 0
            ep_trajs = []
            
            # Track hour to detect boundary condition
            hour = 0
            
            while not done and hour < 24:  # Add explicit hour check
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                # Get action using the actor
                with torch.no_grad():
                    action = temp_actor(obs_tensor).cpu().numpy()[0]                
                # Clip action to environment limits
                action = np.clip(action, self._env.action_space.low, self._env.action_space.high)
                
                try:
                    # Apply action
                    next_obs, reward, terminated, truncated, _ = self._env.step(action)
                    done = terminated or truncated
                except IndexError:
                    # Handle the boundary case
                    print(f"[Fitness] Caught boundary error at hour {hour}")
                episode_reward += reward
                hour += 1

                if return_trajs:
                    transition = (obs, action, reward, next_obs, done)
                    ep_trajs.append(transition)
                
                obs = next_obs
        
            reward_sum += episode_reward
            if return_trajs:
                all_trajs.extend(ep_trajs)
        
        mean_reward = reward_sum / n_episodes
        print(f"[Fitness] Evaluated weights, mean reward: {mean_reward}")
        
        if return_trajs:
            return (mean_reward, all_trajs)
        return (mean_reward,)


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
        return torch.tanh(self.output(x))
    
class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        super().__init__()
        self.input_layer = torch.nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.output_layer = torch.nn.Linear(hidden_dims[-1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
    