import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneObjectDetectionEnv(gym.Env):
    """
    A custom environment for simulating object detection metrics in a drone scenario.

    Attributes:
        action_space: Description of the action space and its configuration.
        observation_space: Description of the observation space and its configuration.
    """

    def __init__(self, initial_throughput=30, initial_latency=200):
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])

        low = np.array([0.0, 0.1, 256.0], dtype=np.float32)
        high = np.array([100.0, 1.0, 1024.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.service_throughput = initial_throughput   # req/s
        self.service_latency = initial_latency           # ms
        self.network_throughput = 4.27                  # MBps
        self.network_latency = 100                       # ms

        self.host_cpu_usage = 50.0                      # percent
        self.service_cpu_limit = 0.5                    # cores
        self.service_memory_limit = 512.0               # MiB

        self.throughput_bounds = (10, 100)
        self.latency_bounds = (100, 250)
        self.net_throughput_bounds = (1, 10)
        self.net_latency_bounds = (50, 150)
        self.cpu_usage_bounds = (20, 90)

        self.detection_payload_size_kb = 1  # KB per detection

        self.timestep = 0
        self.max_timesteps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.service_throughput = 30
        self.service_latency = 200
        self.network_throughput = 4.27
        self.network_latency = 100
        self.host_cpu_usage = 50.0
        self.service_cpu_limit = 0.5
        self.service_memory_limit = 512.0
        self.timestep = 0
        return self._get_obs(), {}

    def step(self, action):
        self.timestep += 1
        throughput_action, slat_action, net_tp_action, nlat_action = action

        self.service_throughput = self.adjust_metric(self.service_throughput, throughput_action, *self.throughput_bounds, 5)
        self.service_latency = self.adjust_metric(self.service_latency, slat_action, *self.latency_bounds, 20, invert=True)
        self.network_throughput = self.adjust_metric(self.network_throughput, net_tp_action, *self.net_throughput_bounds, 1)
        self.network_latency = self.adjust_metric(self.network_latency, nlat_action, *self.net_latency_bounds, 10, invert=True)

        cpu_base = 30.0
        cpu_per_throughput = 0.3 * (self.service_throughput - 10)
        cpu_per_latency = (250 - self.service_latency) * 0.15
        self.host_cpu_usage = np.clip(cpu_base + cpu_per_throughput + cpu_per_latency, *self.cpu_usage_bounds)

        max_possible_throughput = (self.network_throughput * 1024) / self.detection_payload_size_kb  
        if self.service_throughput > max_possible_throughput:
            self.service_latency = min(self.latency_bounds[1], self.service_latency + 30)

        possible_detection_rate = min(self.service_throughput, max_possible_throughput)
        dropped = max(0, self.service_throughput - possible_detection_rate)

        e2e_latency = self.service_latency + self.network_latency + 50

        reward = self.calculate_reward(dropped, e2e_latency)

        terminated = False
        truncated = self.timestep >= self.max_timesteps

        obs = self._get_obs()
        info = {
            "service_throughput": self.service_throughput,
            "service_latency": self.service_latency,
            "network_throughput": self.network_throughput,
            "network_latency": self.network_latency,
            "host_cpu_usage": self.host_cpu_usage,
            "e2e_latency": e2e_latency,
            "dropped": dropped
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        return np.array([
            self.host_cpu_usage,
            self.service_cpu_limit,
            self.service_memory_limit
        ], dtype=np.float32)

    def adjust_metric(self, current_value, action, lower_bound, upper_bound, step, invert=False):
        if action == 0:
            return max(lower_bound, current_value - step) if not invert else min(upper_bound, current_value + step)
        elif action == 2:
            return min(upper_bound, current_value + step) if not invert else max(lower_bound, current_value - step)
        return current_value

    def calculate_reward(self, dropped, e2e_latency):
        dropped_penalty = -5.0 * dropped
        latency_penalty = -0.05 * max(0, e2e_latency - 500)
        cpu_penalty = -2.0 if self.host_cpu_usage > 85.0 else 0.0

        if dropped > 0 or e2e_latency > 500:
            return -10.0 + dropped_penalty + latency_penalty + cpu_penalty
        reward = 1.0 + cpu_penalty
        if e2e_latency < 250:
            reward += 3.0
        return reward