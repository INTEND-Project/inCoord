import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneObjectDetectionEnv(gym.Env):
    # Constants for clarity
    MAX_LATENCY = 200.0
    MAX_SERVICE_LATENCY = 80.0
    MIN_SERVICE_THROUGHPUT = 30.0
    MAX_NETWORK_LATENCY = 100.0
    MIN_NETWORK_THROUGHPUT = 4.27
    DELTA_SERVICE_LATENCY = 10.0
    DELTA_SERVICE_THROUGHPUT = 5.0
    DELTA_NETWORK_LATENCY = 10.0
    DELTA_NETWORK_THROUGHPUT = 0.5

    def __init__(self):
        super().__init__()
        low = np.array([50.0, 10.0, 80.0, 2.0], dtype=np.float32)
        high = np.array([self.MAX_SERVICE_LATENCY, 50.0, self.MAX_NETWORK_LATENCY, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])

        # Initial state randomized within bounds
        self.state = np.random.uniform(low=[self.MAX_SERVICE_LATENCY, self.MIN_SERVICE_THROUGHPUT, 80.0, self.MIN_NETWORK_THROUGHPUT],
                                        high=[self.MAX_LATENCY, 70.0, self.MAX_NETWORK_LATENCY, 10.0],
                                        size=(4,))
        self.steps = 0
        self.max_steps = 50

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=[self.MAX_SERVICE_LATENCY, self.MIN_SERVICE_THROUGHPUT, 80.0, self.MIN_NETWORK_THROUGHPUT],
                                        high=[self.MAX_LATENCY, 70.0, self.MAX_NETWORK_LATENCY, 10.0],
                                        size=(4,))
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        new_state = self.state.copy()
        
        # Service Latency adjustment
        if action[0] == 0:
            new_state[0] = max(new_state[0] - self.DELTA_SERVICE_LATENCY, 50.0)
        elif action[0] == 2:
            new_state[0] = min(new_state[0] + self.DELTA_SERVICE_LATENCY, 300.0)
        
        # Service Throughput adjustment
        if action[1] == 0:
            new_state[1] = max(new_state[1] - self.DELTA_SERVICE_THROUGHPUT, 10.0)
            new_state[0] = max(new_state[0] - 2, 50.0)
        elif action[1] == 2:
            new_state[1] = min(new_state[1] + self.DELTA_SERVICE_THROUGHPUT, 70.0)
            new_state[0] = min(new_state[0] + 2, 300.0)
        
        # Network Latency adjustment
        if action[2] == 0:
            new_state[2] = max(new_state[2] - self.DELTA_NETWORK_LATENCY, 80.0)
        elif action[2] == 2:
            new_state[2] = min(new_state[2] + self.DELTA_NETWORK_LATENCY, 300.0)
        
        # Network Throughput adjustment
        if action[3] == 0:
            new_state[3] = max(new_state[3] - self.DELTA_NETWORK_THROUGHPUT, 2.0)
            new_state[2] = min(new_state[2] + 2, 300.0)
        elif action[3] == 2:
            new_state[3] = min(new_state[3] + self.DELTA_NETWORK_THROUGHPUT, 10.0)
            new_state[2] = max(new_state[2] - 2, 80.0)

        self.state = new_state
        self.steps += 1

        # Reward calculation
        done = False
        reward = 0.0

        end_to_end_latency = self.state[0] + self.state[2]
        if end_to_end_latency > 200.0:
            reward -= 2.0 * ((end_to_end_latency - 200.0) / 10)
        
        if self.state[0] > self.MAX_SERVICE_LATENCY:
            reward -= 1.0 * ((self.state[0] - self.MAX_SERVICE_LATENCY) / 10)
        
        if self.state[1] < self.MIN_SERVICE_THROUGHPUT:
            reward -= 2.0 * ((self.MIN_SERVICE_THROUGHPUT - self.state[1]) / 5)
        
        if self.state[2] > self.MAX_NETWORK_LATENCY:
            reward -= 1.0 * ((self.state[2] - self.MAX_NETWORK_LATENCY) / 10)
        
        if self.state[3] < self.MIN_NETWORK_THROUGHPUT:
            reward -= 2.0 * ((self.MIN_NETWORK_THROUGHPUT - self.state[3]) / 0.5)

        if (end_to_end_latency <= 200.0 
            and self.state[0] <= self.MAX_SERVICE_LATENCY
            and self.state[1] >= self.MIN_SERVICE_THROUGHPUT
            and self.state[2] <= self.MAX_NETWORK_LATENCY
            and self.state[3] >= self.MIN_NETWORK_THROUGHPUT):
            reward += 20.0

        reward -= 1.0

        if self.steps >= self.max_steps or not np.isfinite(reward):
            done = True

        return self._get_obs(), float(reward), done, False, {}

    def _get_obs(self):
        obs = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        return obs.astype(np.float32)