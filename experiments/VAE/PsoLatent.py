import torch
import numpy as np
from pyswarm import pso

class PSO_Opt_Latent:
    def __init__(self, env, vae, swarm_size, max_iter):
        self.env = env
        self.vae = vae
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.latent_dim = vae.latent_dim
        self.device = 'cpu'

    def optimize(self):
        print('optimizing latent space')
        def pso_obj(z):
            z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)  # [1, latent_dim]
            traj = self.vae.decode(z_tensor).detach().cpu().numpy()[0]
            return -self.evaluate_traj(traj)  # negative reward

        best_z, best_value = pso(
            pso_obj,
            lb=np.full(self.latent_dim, -1.0),
            ub=np.full(self.latent_dim, 1.0),
            swarmsize=self.swarm_size,
            maxiter=self.max_iter
        )
        return best_z, best_value

    def evaluate_traj(self, decoded_traj):
        reward_sum = 0
        self.env.reset()
        for t in decoded_traj:
            obs = self.env._get_observation()
            action = t[-1]  # Assuming action is last component in [obs + act]
            _, reward, done, _, _ = self.env.step(np.array([action]))
            reward_sum += reward
            if done:
                break
        return reward_sum
