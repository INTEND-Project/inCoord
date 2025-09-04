import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneObjectDetectionEnv(gym.Env):
    """
    Gymnasium environment for real-time, lossless object detection classification delivery
    from drone to dashboard via tunable network and service SLIs.
    """

    def __init__(self):
        super(DroneObjectDetectionEnv, self).__init__()

        # Observable metrics: [Service Latency (ms), Service Throughput (req/s), Network Latency (ms), Network Throughput (Mbps)]
        self.obs_low = np.array([100, 20, 40, 2], dtype=np.float32)
        self.obs_high = np.array([200, 80, 150, 30], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        # Action space: For each configurable metric, select one of [decrease, maintain, increase]
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])

        # Step sizes for each metric
        self.metric_steps = np.array([10, 5, 10, 1], dtype=np.float32)

        # Initial state (randomized within valid range)
        self.state = np.random.uniform(self.obs_low, self.obs_high)

        # Business requirements (hard constraints)
        self.target_total_latency = 300     # ms (soft upper bound)
        self.max_service_latency = 150      # ms (strict)
        self.max_network_latency = 120      # ms (strict)
        self.min_service_throughput = 30    # req/s (strict)
        self.min_network_throughput = 5     # Mbps (strict)
        self.penalty_large = -100.0
        self.penalty_small = -10.0
        self.reward_perfect = 10.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomize initial state within valid range
        self.state = np.random.uniform(self.obs_low, self.obs_high).astype(np.float32)
        return self.state.copy(), {}

    def step(self, action):
        action = np.array(action)
        prev_state = self.state.copy()

        # Calculate state adjustments based on action
        deltas = np.zeros_like(self.state)
        for i in range(len(action)):
            if action[i] == 0:
                deltas[i] = -self.metric_steps[i]
            elif action[i] == 2:
                deltas[i] = self.metric_steps[i]

        # Apply changes and clip to valid ranges
        self.state += deltas
        self.state = np.clip(self.state, self.obs_low, self.obs_high)

        # Update Service Latency based on Service Throughput
        base_latency = 200.0
        base_throughput = 30.0
        latency_alpha = 2.0  # ms per req/s above base
        throughput = self.state[1]
        self.state[0] = np.clip(base_latency - latency_alpha * (throughput - base_throughput), self.obs_low[0], self.obs_high[0])

        # Update Network Latency based on Network Throughput
        base_net_latency = 100.0
        base_bw = 4.27
        beta = 2.0  # ms per Mbps above base
        net_throughput = self.state[3]
        self.state[2] = np.clip(base_net_latency - beta * (net_throughput - base_bw), self.obs_low[2], self.obs_high[2])

        # Update Service Throughput based on Service Latency
        gamma = 0.1  # req/s per ms above base
        service_latency = self.state[0]
        min_throughput = base_throughput - gamma * (service_latency - base_latency)
        self.state[1] = np.clip(max(self.state[1], min_throughput), self.obs_low[1], self.obs_high[1])

        # --- Compute reward ---
        service_latency = self.state[0]
        service_throughput = self.state[1]
        network_latency = self.state[2]
        network_throughput = self.state[3]
        total_latency = service_latency + network_latency

        reward = 0.0

        # Constraint violation penalties (hard constraints)
        if service_latency > self.max_service_latency:
            reward += self.penalty_large
        if network_latency > self.max_network_latency:
            reward += self.penalty_large
        if service_throughput < self.min_service_throughput:
            reward += self.penalty_large
        if network_throughput < self.min_network_throughput:
            reward += self.penalty_large

        # Soft constraint: Prefer total latency < target
        if total_latency > self.target_total_latency:
            reward += self.penalty_small * ((total_latency - self.target_total_latency) / 10.0)

        # Bonus for hitting all targets
        if (
            service_latency <= self.max_service_latency
            and network_latency <= self.max_network_latency
            and service_throughput >= self.min_service_throughput
            and network_throughput >= self.min_network_throughput
            and total_latency <= self.target_total_latency
        ):
            reward += self.reward_perfect

        # Episode termination if any hard constraints are violated badly
        terminated = bool(
            service_latency > self.obs_high[0]
            or service_throughput < self.obs_low[1]
            or network_latency > self.obs_high[2]
            or network_throughput < self.obs_low[3]
        )
        truncated = False

        return self.state.copy(), float(reward), terminated, truncated, {}