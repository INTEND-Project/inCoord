import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ResizeImageLatencyEnv(gym.Env):
    """
    A custom Gym environment to manage service and network latencies for image processing.

    The environment includes observation metrics such as service latency, 
    network latency, bandwidth, CPU limits, memory limits, and host CPU usage.
    
    Action Space:
        - Decrease (-1), Maintain (0), or Increase (+1) service and network latencies.

    Observation Space:
        - A Box of continuous values representing the state of the environment.

    Rewards:
        - Positive rewards for meeting latency targets.
        - Penalties for overprovisioning and underperformance.
    """

    # Constants for latencies and penalties
    SERVICE_LATENCY_MIN = 150.0
    SERVICE_LATENCY_MAX = 600.0
    NETWORK_LATENCY_MIN = 20.0
    NETWORK_LATENCY_MAX = 300.0
    BANDWIDTH = 50.0
    SERVICE_CPU_LIMIT = 0.25
    SERVICE_MEMORY_LIMIT = 512.0
    HOST_CPU_USAGE = 10.0
    TARGET_SERVICE_LATENCY_UPPER = 300.0
    TARGET_SERVICE_LATENCY_LOWER = 200.0
    TARGET_NETWORK_LATENCY_UPPER = 150.0
    TARGET_NETWORK_LATENCY_LOWER = 50.0
    OVERPROVISION_PENALTY = -0.5  # Penalty for below-lower-bound latency
    UNDERPERF_PENALTY = -1.0      # Penalty for above-upper-bound latency
    EFFICIENT_REWARD = 1.0        # Reward for metrics in target band
    LATENCY_STEP = 25.0
    NETWORK_LATENCY_STEP = 25.0

    def __init__(self):
        super().__init__()
        self.obs_low = np.array([self.SERVICE_LATENCY_MIN, self.NETWORK_LATENCY_MIN, self.BANDWIDTH, 
                                  self.SERVICE_CPU_LIMIT, self.SERVICE_MEMORY_LIMIT, self.HOST_CPU_USAGE], dtype=np.float32)
        self.obs_high = np.array([self.SERVICE_LATENCY_MAX, self.NETWORK_LATENCY_MAX, self.BANDWIDTH, 
                                   self.SERVICE_CPU_LIMIT, self.SERVICE_MEMORY_LIMIT, self.HOST_CPU_USAGE], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.service_latency = 550.0
        self.network_latency = 250.0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Invalid action: {}".format(action))
        
        new_network_latency = self._update_network_latency(action[1])
        min_possible_service_latency = new_network_latency + 100.0
        new_service_latency = self._update_service_latency(action[0], min_possible_service_latency)

        self.network_latency = new_network_latency
        self.service_latency = new_service_latency
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, truncated = self._check_termination()
        return obs, float(reward), terminated, truncated, {}

    def _update_network_latency(self, action_value):
        if action_value == 0:
            return self.network_latency
        elif action_value == 1:
            return min(self.network_latency + self.NETWORK_LATENCY_STEP, self.NETWORK_LATENCY_MAX)
        else:
            return max(self.network_latency - self.NETWORK_LATENCY_STEP, self.NETWORK_LATENCY_MIN)

    def _update_service_latency(self, action_value, min_possible_service_latency):
        if action_value == 0:
            return self.service_latency
        elif action_value == 1:
            return min(self.service_latency + self.LATENCY_STEP, self.SERVICE_LATENCY_MAX)
        else:
            candidate = self.service_latency - self.LATENCY_STEP
            return max(candidate, min_possible_service_latency, self.SERVICE_LATENCY_MIN)

    def _get_obs(self):
        return np.array([
            self.service_latency,
            self.network_latency,
            self.BANDWIDTH,
            self.SERVICE_CPU_LIMIT,
            self.SERVICE_MEMORY_LIMIT,
            self.HOST_CPU_USAGE,
        ], dtype=np.float32)

    def _compute_reward(self):
        reward = 0.0

        if self.service_latency < self.TARGET_SERVICE_LATENCY_LOWER:
            reward += self.OVERPROVISION_PENALTY
        elif self.service_latency > self.TARGET_SERVICE_LATENCY_UPPER:
            reward += (self.UNDERPERF_PENALTY *
                       ((self.service_latency - self.TARGET_SERVICE_LATENCY_UPPER)
                        /50.0 + 1))
        else:
            reward += self.EFFICIENT_REWARD

        if self.network_latency < self.TARGET_NETWORK_LATENCY_LOWER:
            reward += self.OVERPROVISION_PENALTY * 0.5
        elif self.network_latency > self.TARGET_NETWORK_LATENCY_UPPER:
            reward += self.UNDERPERF_PENALTY * 0.5 * ((self.network_latency - self.TARGET_NETWORK_LATENCY_UPPER) / 50.0 + 1)
        else:
            reward += self.EFFICIENT_REWARD * 0.5

        return float(reward)

    def _check_termination(self):
        terminated = False
        truncated = False
        if self.service_latency > self.SERVICE_LATENCY_MAX or self.network_latency > self.NETWORK_LATENCY_MAX:
            terminated = True
        return terminated, truncated