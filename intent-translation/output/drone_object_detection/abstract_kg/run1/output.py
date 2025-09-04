import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneObjectDetectionEnv(gym.Env):
    """
    Gymnasium environment for tuning service and network metrics
    to ensure real-time, lossless object detection event delivery.
    Observable metrics:
        - Service Latency (ms)
        - Service Throughput (req/s)
        - Network Latency (ms)
        - Network Throughput (MBps)
    Configurable metrics (action space):
        - Service Latency (ms)
        - Network Latency (ms)
        - Network Throughput (MBps)
    Objectives:
        - Minimize end-to-end latency (Service Latency + Network Latency)
        - Keep Service Latency <= 100 ms (strict), Network Latency <= 100 ms, Network Throughput >= 1 MBps
        - Severe penalty for dropping below required throughput or exceeding latency threshold
    """

    def __init__(self, initial_state=None):
        super().__init__()
        # Observable metrics bounds
        self.service_latency_bounds = (10, 200)      # ms
        self.service_throughput_bounds = (10, 50)    # req/s (not tunable)
        self.network_latency_bounds = (20, 100)      # ms
        self.network_throughput_bounds = (1, 10)     # MBps

        # State: [service_latency, service_throughput, network_latency, network_throughput]
        low = np.array([
            self.service_latency_bounds[0],
            self.service_throughput_bounds[0],
            self.network_latency_bounds[0],
            self.network_throughput_bounds[0]
        ], dtype=np.float32)
        high = np.array([
            self.service_latency_bounds[1],
            self.service_throughput_bounds[1],
            self.network_latency_bounds[1],
            self.network_throughput_bounds[1]
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space (MultiDiscrete): Adjust configurable metrics.
        self.action_space = spaces.MultiDiscrete([3, 3, 3])  # [service_latency, network_latency, network_throughput]

        # Initial values (from system state)
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float32)
        else:
            self.state = np.array([200.0, 30.0, 100.0, 4.27], dtype=np.float32)

        # Step size for each action
        self.service_latency_step = 10.0
        self.network_latency_step = 5.0
        self.network_throughput_step = 0.5

        # Target thresholds
        self.target_service_latency = 100.0
        self.target_network_latency = 100.0
        self.min_network_throughput = 1.0

        # End-to-end latency hard limit
        self.end_to_end_latency_limit = 500.0  # ms

        # Reward penalties
        self.penalty_latency_exceed = -10.0
        self.severe_penalty = -100.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([200.0, 30.0, 100.0, 4.27], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        service_latency, service_throughput, network_latency, network_throughput = self.state

        # Apply action to configurable metrics
        service_latency = self._adjust_service_latency(service_latency, action[0])
        network_latency = self._adjust_network_latency(network_latency, action[1])
        network_throughput = self._adjust_network_throughput(network_throughput, action[2])

        # Simulate dependencies
        service_throughput = self._update_service_throughput(service_latency, service_throughput, network_throughput)

        # Compose new state
        self.state = np.array([
            service_latency,
            service_throughput,
            network_latency,
            network_throughput
        ], dtype=np.float32)

        # Compute end-to-end latency
        end_to_end_latency = service_latency + network_latency

        # Reward logic
        reward = self._calculate_reward(end_to_end_latency, service_latency, network_latency, network_throughput, service_throughput)

        # Done condition
        done = self._check_done(service_latency, network_latency, network_throughput, service_throughput)

        return self.state.copy(), float(reward), done, False, {}

    def _adjust_service_latency(self, service_latency, action):
        if action == 0:
            return max(self.service_latency_bounds[0], service_latency - self.service_latency_step)
        elif action == 2:
            return min(self.service_latency_bounds[1], service_latency + self.service_latency_step)
        return service_latency

    def _adjust_network_latency(self, network_latency, action):
        if action == 0:
            return max(self.network_latency_bounds[0], network_latency - self.network_latency_step)
        elif action == 2:
            return min(self.network_latency_bounds[1], network_latency + self.network_latency_step)
        return network_latency

    def _adjust_network_throughput(self, network_throughput, action):
        if action == 0:
            return max(self.network_throughput_bounds[0], network_throughput - self.network_throughput_step)
        elif action == 2:
            return min(self.network_throughput_bounds[1], network_throughput + self.network_throughput_step)
        return network_throughput

    def _update_service_throughput(self, service_latency, service_throughput, network_throughput):
        if service_latency < self.state[0]:
            service_throughput = min(self.service_throughput_bounds[1], service_throughput + 1.0)
        elif service_latency > self.state[0]:
            service_throughput = max(self.service_throughput_bounds[0], service_throughput - 1.0)

        # Adjust service throughput based on network throughput
        if network_throughput < 1.0:
            service_throughput = max(self.service_throughput_bounds[0], service_throughput - 5.0)

        return service_throughput

    def _calculate_reward(self, end_to_end_latency, service_latency, network_latency, network_throughput, service_throughput):
        reward = 0.0

        if end_to_end_latency > self.end_to_end_latency_limit:
            reward += self.severe_penalty
        if service_latency > self.targe, st_service_latency:
            reward += self.penalty_latency_exceed * ((service_latency - self.target_service_latency) / 10)
        if network_latency > self.target_network_latency:
            reward += 5.0 * ((network_latency - self.target_network_latency) / 10)
        if network_throughput < self.min_network_throughput:
            reward += -50.0
        
        if service_latency <= self.target_service_latency and network_latency <= self.target_network_latency:
            reward += 10.0
        reward += np.clip((500.0 - end_to_end_latency) / 50.0, 0, 10)
        if network_throughput >= self.min_network_throughput:
            reward += 5.0
        reward += (service_throughput - self.service_throughput_bounds[0]) / 10.0

        return reward

    def _check_done(self, service_latency, network_latency, network_throughput, service_throughput):
        return bool(
            service_latency < self.service_latency_bounds[0]
            or service_latency > self.service_latency_bounds[1]
            or network_latency < self.network_latency_bounds[0]
            or network_latency > self.network_latency_bounds[1]
            or network_throughput < self.network_throughput_bounds[0]
            or network_throughput > self.network_throughput_bounds[1]
            or service_throughput < self.service_throughput_bounds[0]
            or service_throughput > self.service_throughput_bounds[1]
        )