import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import datetime
import csv
import os


class GRPOAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super(GRPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, obs):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        return dist


class GRPO:
    def __init__(self, env, learning_rate=3e-4, device="cpu", batch_size=32):
        self.env = env
        self.device = torch.device(device)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.agent = GRPOAgent(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.batch_size = batch_size

    def train(self, total_episodes=100, steps_per_episode=200):
        all_episode_rewards = []

        for episode in range(total_episodes):
            obs_info = self.env.reset()
            if isinstance(obs_info, tuple):
                obs, info = obs_info
            else:
                obs = obs_info

            episode_reward = 0
            log_probs = []
            rewards = []

            for step in range(steps_per_episode):
                obs = np.array(obs, dtype=np.float32)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

                dist = self.agent(obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                step_result = self.env.step(action.item())
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result

                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward

                if done:
                    break

                obs = next_obs

            all_episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}")

        # Flush Monitor logs
        self.env.close()

    def predict(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        dist = self.agent(obs_tensor)
        action = dist.sample()
        return action.item()

    def save(self, model_name="grpo_model"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = f"{model_name}.{timestamp}.pth"
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        torch.save(self.agent.state_dict(), f"models/{model_path}")
        print(f"Model saved to models/{model_path}")

    def load(self, path):
        self.agent.load_state_dict(torch.load(path))
        self.agent.eval()