import argparse
import datetime
import yaml
import os

import numpy as np
import pandas as pd

# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import env_checker

from stable_baselines3 import DQN
from stable_baselines3 import PPO
from grpo import GRPO

from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.callbacks import BaseCallback
#from stable_baselines3.common.results_plotter import load_results, ts2xy
#from active_simulation.threshold_finder_test import adaptive_step


class LatencyOptimizationEnv(gym.Env):
    """Custom RL environment for optimizing network & computing latency."""

    def __init__(self): #, data):
        super(LatencyOptimizationEnv, self).__init__()

        # Load dataset
        #self.data = data
        self.index = 0  # Track current sample
        self.step_count = 0
        self.max_steps_per_episode = 10000

        # Define action space: 0 = Keep latencies, 1 = Decrease Network, 2 = Decrease Computing, 3 = Decrease Both
        self.action_space = spaces.Discrete(4)

        # Define observation space (normalized values)
        self.observation_space = spaces.Dict(
            {
            "infrastructure": spaces.Box(low=np.array([0, 0]), high=np.array([15, 45]), dtype=np.float32),
            "application": spaces.Discrete(7)
            }
        )

    def reset(self, *, seed=None, options=None):
        """Reset environment (start with a random sample)."""
        super().reset(seed=seed)  # ensures reproducibility

        self.step_count = 0
        self.index = np.random.randint(0, len(self.data))
        obs = self._get_observation()
        print(obs)
        return obs, {}

    def step(self, action):
        """Take an action and return new state, reward, and done flag."""
        # Retrieve current sample
        self.step_count += 1

        sample = self.data.iloc[self.index].copy()

        # Target bitrate for maintaining 1440p resolution
        target_throughput = 15000  # Adjust based on dataset

        # Current bitrate from the sample
        current_throughput = sample['Throughput (kbps)']

        # Maximum possible reduction step
        max_reduction = 5  # Change this value based on experimentation
        alpha = 4  # Controls how aggressively reductions are applied

        # Compute a scaling factor for the reduction
        scaling_factor = np.exp(-alpha * (current_throughput / target_throughput))

        # Compute adaptive reduction step
        adaptive_step = max_reduction * scaling_factor


        old_network_latency = sample['Total Network Latency (s)']
        old_computing_latency = sample['Total Computing Latency (s)']


        # Apply dynamic reduction based on action
        if action == 1:  # Decrease Network Latency
            sample['Total Network Latency (s)'] = max(sample['Total Network Latency (s)'] - adaptive_step, 0)
        elif action == 2:  # Decrease Computing Latency
            sample['Total Computing Latency (s)'] = max(sample['Total Computing Latency (s)'] - adaptive_step, 0)
        elif action == 3:  # Decrease Both
            sample['Total Network Latency (s)'] = max(sample['Total Network Latency (s)'] - adaptive_step, 0)
            sample['Total Computing Latency (s)'] = max(sample['Total Computing Latency (s)'] - adaptive_step, 0)


        # Step 2: Compute throughput for each test case
        chunk_size = 4 * 1024 * 1024 * 8

        # Compute throughput based on total latency
        sample["Throughput (kbps)"] = chunk_size / (sample["Total Network Latency (s)"] + sample[
            "Total Computing Latency (s)"]) / 1000  # Convert bps to kbps

        # Determine reward (high for achieving 1440p)
        bitrates = np.array([300, 700, 1000, 2000, 4000, 8000, 15000, 30000])

        # Function to select the best bitrate
        def select_bitrate(throughput, bitrates):
            valid_bitrates = bitrates[bitrates <= throughput]
            return max(valid_bitrates) if len(valid_bitrates) > 0 else min(bitrates)  # Fallback to lowest bitrate

        old_bitrate = sample["Selected Bitrate (kbps)"]

        sample["Selected Bitrate (kbps)"] = select_bitrate(sample["Throughput (kbps)"], bitrates)

        bitrates_to_res = {
            300: '240p',
            700: '240p',
            1000: '360p',
            2000: '480p',
            4000: '720p',
            8000: '1080p',
            15000: '1440p',
            30000: '2160p'
        }

        res_to_code = {}

        def select_resolution(bitrate, bitrates_to_res):
            resolution = bitrates_to_res[bitrate]
            return resolution

        old_resolution = sample["Resolution"]
        sample["Resolution"] = select_resolution(sample["Selected Bitrate (kbps)"], bitrates_to_res)

        bitrate = sample['Selected Bitrate (kbps)']
        resolution = sample['Resolution']

        if old_resolution != '1440p' and resolution == '1440p':#resolution == '1440p':
            reward = 10
        elif old_resolution != '1440p' and resolution != '1440p':  # resolution == '1440p':
            reward = -2
        elif old_resolution == '1440p' and resolution == '1440p':
            if current_throughput < sample["Throughput (kbps)"]:
                reward = -1
            else:
                reward = 3


        # reward_penalty = 0
        #
        #
        # reward_bonus = 0
        # if old_resolution != '1440p' and resolution == '1440p':
        #     reward_bonus = 5


        total_reward = reward #+ reward_penalty + reward_bonus

        # Move to next sample
        self.index = (self.index + 1) % len(self.data)
        obs = self._get_observation()

        done = (self.step_count >= self.max_steps_per_episode)

        # Gymnasium separates `terminated` and `truncated`
        terminated = False  # Set to True if episode ends due to game logic
        truncated = done  # Set to True if episode ends due to time/limit

        info = {
            'resolution': resolution,
            'throughput': sample["Throughput (kbps)"]
        }

        return obs, total_reward, terminated, truncated, info

    def _get_observation(self):
        """Return the current state as a NumPy array."""
        sample = self.data.iloc[self.index]
        return np.array([
            sample['Total Network Latency (s)'],
            sample['Total Computing Latency (s)'],
            sample['Resolution']
        ], dtype=np.float32)
        # sample['Throughput (kbps)'],
        # sample['Selected Bitrate (kbps)']

def train(model_name, data_path, config, train_timesteps):
    df = pd.read_csv(data_path)
    df['Resolution_Binary'] = [int(res in ['1440p', '2160p']) for res in df["Resolution"].values]
    base_env = LatencyOptimizationEnv(df)

    # Create log directory if it doesn't exist
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with Monitor for ALL models (including GRPO)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    monitor_file = os.path.join(log_dir, f"{model_name.lower()}_monitor_{timestamp}")

    env = Monitor(
        base_env,
        filename=monitor_file,
        info_keywords=('resolution', 'throughput')  # Track these info dict keys
    )

    try:
        env_checker.check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

    input("check done")

    if model_name == "DQN":
        model_cls = DQN
        model = model_cls("MlpPolicy", env, verbose=1, **config)
        model.learn(total_timesteps=train_timesteps)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model.save(f"models/{model_name.lower()}_latency_optimizer.{timestamp}")
        print(f"Trained and saved {model_name} model.")

    elif model_name == "PPO":
        model_cls = PPO
        model = model_cls("MlpPolicy", env, verbose=1, **config)
        model.learn(total_timesteps=train_timesteps)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model.save(f"models/{model_name.lower()}_latency_optimizer.{timestamp}")
        print(f"Trained and saved {model_name} model.")

    elif model_name == "GRPO":
        # Now GRPO also uses the Monitor-wrapped environment
        model = GRPO(env, learning_rate=config.get("learning_rate", 3e-4),
                     batch_size=config.get("batch_size", 32))
        model.train(total_episodes=config.get("total_episodes", 1000),
                    steps_per_episode=config.get("steps_per_episode", 200))
        model.save("grpo_latency_optimizer")
        print(f"Trained and saved {model_name} model.")

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"Monitor logs saved to: {monitor_file}.monitor.csv")


def train_debug(model_name, data_path, config, train_timesteps):
    """Debug version of train function to diagnose Monitor issues"""
    df = pd.read_csv(data_path)
    df['Resolution_Binary'] = [int(res in ['1440p', '2160p']) for res in df["Resolution"].values]
    base_env = LatencyOptimizationEnv(df)

    # Create log directory if it doesn't exist
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with Monitor for ALL models (including GRPO)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    monitor_file = os.path.join(log_dir, f"{model_name.lower()}_monitor_{timestamp}")

    print(f"Monitor file will be: {monitor_file}.monitor.csv")

    env = Monitor(
        base_env,
        filename=monitor_file,
        info_keywords=('resolution', 'throughput')  # Track these info dict keys
    )

    if model_name == "GRPO":
        print("Starting GRPO training with Monitor debugging...")

        # Test a few episodes manually first
        print("Testing environment behavior...")
        for test_episode in range(3):
            print(f"\n--- Test Episode {test_episode + 1} ---")
            obs_info = env.reset()
            if isinstance(obs_info, tuple):
                obs, info = obs_info
            else:
                obs = obs_info
            print(f"Reset successful, obs shape: {obs.shape}")

            total_reward = 0
            for step in range(10):  # Just 10 steps for testing
                action = env.action_space.sample()  # Random action
                step_result = env.step(action)

                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result

                total_reward += reward
                print(f"  Step {step}: action={action}, reward={reward}, done={done}")

                if done:
                    print(f"  Episode ended at step {step}")
                    break

                obs = next_obs

            print(f"Test episode reward: {total_reward}")

            # Check if monitor file was created
            if os.path.exists(f"{monitor_file}.monitor.csv"):
                print(f"Monitor file exists, size: {os.path.getsize(f'{monitor_file}.monitor.csv')} bytes")
                # Read and print the contents
                try:
                    with open(f"{monitor_file}.monitor.csv", 'r') as f:
                        content = f.read()
                        print(f"Monitor file content:\n{content}")
                except Exception as e:
                    print(f"Error reading monitor file: {e}")
            else:
                print("Monitor file does not exist yet")

        # Now run actual GRPO training
        print("\nStarting actual GRPO training...")
        model = GRPO(env, learning_rate=config.get("learning_rate", 3e-4),
                     batch_size=config.get("batch_size", 32))
        model.train(total_episodes=config.get("total_episodes", 10),  # Reduced for testing
                    steps_per_episode=config.get("steps_per_episode", 50))  # Reduced for testing
        model.save("grpo_latency_optimizer")
        print(f"Trained and saved {model_name} model.")

        # Check final monitor file
        if os.path.exists(f"{monitor_file}.monitor.csv"):
            print(f"Final monitor file size: {os.path.getsize(f'{monitor_file}.monitor.csv')} bytes")
            try:
                with open(f"{monitor_file}.monitor.csv", 'r') as f:
                    content = f.read()
                    print(f"Final monitor file content:\n{content}")
            except Exception as e:
                print(f"Error reading final monitor file: {e}")
        else:
            print("Final monitor file does not exist!")

    else:
        raise ValueError(f"Debug function only supports GRPO, got: {model_name}")

    print(f"Monitor logs should be at: {monitor_file}.monitor.csv")

if __name__ == "__main__":
    print("Starting training")
    # Load dataset
    data_path = "../data/simulations/throughput_bitrate/throughput_bitrate.20250209181231.csv"

    with open("model_config.yaml") as f:
        config = yaml.safe_load(f)
    print(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PPO", help="Choose DQN, PPO, or GRPO")
    args = parser.parse_args()
    model_name = args.model.upper()
    model_params = config["model_params"].get(model_name, {})
    train_timesteps = config["train_params"]["total_timesteps"]
    #train(model_name, data_path, model_params, train_timesteps)
    train(model_name, data_path, model_params, train_timesteps)