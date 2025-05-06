import torch
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import random
from env.modified_env import TradingEnv
import pickle
from experiments.Imitation_Learning.PSO_Opt import ActorNet, CriticNet

class RelayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class CoLTrainer:
    def __init__(self, env, actor, critic, actor_target, critic_target,
                 expert_buffer, agent_buffer, lambda_bc, lambda_q, lambda_actor,
                 lambda_reg, buffer_size=1000000, batch_size=256, gamma=0.99, tau=0.005, lr=3e-4, device='cpu'):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target

        self.expert_buffer = expert_buffer
        self.agent_buffer = agent_buffer

        self.lambda_bc = lambda_bc
        self.lambda_q = lambda_q
        self.lambda_actor = lambda_actor
        self.lambda_reg = lambda_reg

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.device = device

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        self.rewards_history = []

    def select_action(self, state):
        with torch.no_grad():
            action = self.actor_critic.act(state)
        return action
    
    def sample_mixed_batch(self, batch):
        num_expert = int(self.batch_size * 0.25)
        num_agent = self.batch_size - num_expert

        expert_batch = self.expert_buffer.sample(num_expert)
        agent_batch = self.agent_buffer.sample(num_agent)

        return expert_batch + agent_batch
    
    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # BC Loss
        expert_states = torch.FloatTensor([s for i, (s, _, _, _, _) in enumerate(batch) if i < int(self.batch_size * 0.25)]).to(self.device)
        expert_actions = torch.FloatTensor([a for i, (_, a, _, _, _) in enumerate(batch) if i < int(self.batch_size * 0.25)]).to(self.device)
        if len(expert_states) > 0:
            bc_loss = F.mse_loss(self.actor(expert_states), expert_actions)
        else:
            bc_loss = torch.tensor(0.0).to(self.device)

        # Critic loss 
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_critic = self.critic_target(next_states, target_actions)
            target_Q = rewards + self.gamma * (1 - dones) * target_critic
            target_Q = torch.clamp(target_Q, min = -10.0, max = 10.0)
        current_Q = self.critic(states, actions)
        q_loss = F.mse_loss(current_Q, target_Q)

        # Actor loss
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()

        # L2 Regularization
        reg_loss = torch.tensor(0.0).to(self.device)
        for param in self.actor.parameters():
            reg_loss += torch.norm(param, p=2)

        # Total CoL loss
        total_loss = (self.lambda_bc * bc_loss +
                      self.lambda_q * q_loss +
                      self.lambda_actor * actor_loss +
                      self.lambda_reg * reg_loss)

        return total_loss, bc_loss.item(), q_loss.item(), actor_loss.item()
    
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, total_steps = 100_000, log_interval = 1000):
        state, _ = self.env.reset()
        episode_reward = 0
        for step in range(1, total_steps + 1):
            action = self.actor(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()[0]
            next_state, reward, done, _, _ = self.env.step(action)
            transition = (state, action, reward, next_state, done)
            self.agent_buffer.add(transition)
            state = next_state
            episode_reward += reward

            if done:
                self.rewards_history.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0
            
            if len(self.agent_buffer) >= self.batch_size:
                ## ToDo: Sample seperately and process each type more clearly. Sample from expert buffer directly
                batch = self.sample_mixed_batch(self.agent_buffer)
                loss, bc_loss, q_loss, actor_loss = self.compute_loss(batch)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                self.soft_update(self.actor, self.actor_target)
                self.soft_update(self.critic, self.critic_target)

                # if q_loss > 10.0:
                #     print(f"Warning: Q loss is high: {q_loss}")
                #     self.lambda_bc *= 1.1
                #     self.lambda_q *= 0.9
                #     print(f"Adjusted lambda_bc: {self.lambda_bc}, lambda_q: {self.lambda_q}") 

            if step % log_interval == 0:
                print(f"Step: {step}, AvgReward: {sum(self.rewards_history[-10:]) / max(1, len(self.rewards_history[-10:])):.2f}, BC: {bc_loss:.4f}, Q: {q_loss:.4f}, Actor: {actor_loss:.4f}")

    def predict(self, obs, deterministic=True):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        return action, None

if __name__ == '__main__':
    # with open("expert_pso_rollouts.pkl", "rb") as f:
    #     expert_rollouts = pickle.load(f)

    # expert_buffer = RelayBuffer(capacity=10_000)
    # for r in expert_rollouts:
    #     expert_buffer.add(r)
    # agent_buufer = RelayBuffer(capacity=100_000)

    # env = FixedSACTradingEnv()
    # obs_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]

    # actor = ActorNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
    # critic = CriticNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
    # actor_target = ActorNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
    # critic_target = CriticNet(obs_dim, action_dim).to('cuda' if torch.cuda.is_available() else 'cpu')

    # actor_target.load_state_dict(actor.state_dict())
    # critic_target.load_state_dict(critic.state_dict())

    # trainer = CoLTrainer(env, actor, critic, actor_target, critic_target, expert_rollouts, buffer_size=1000000, batch_size=256, lambda_bc=1.0, lambda_q=1.0, lambda_actor=1.0, lambda_reg=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu')
    # trainer.train(total_steps=100000, log_interval=1000)
    print()