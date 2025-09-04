import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneObjectDetectionEnv(gym.Env):
    """
    Gymnasium environment for configuring SLIs to ensure real-time, lossless object detection classification transmission from drones to dashboard.
    """

    PENALTY_FACTOR = 2.0
    BONUS_FOR_CONSTRAINT_SATISFACTION = 5.0

    def __init__(self, initial_state=None):
        super().__init__()
        self.service_throughput_range = (10, 200)
        self.service_latency_range = (10, 200)
        self.network_latency_range = (50, 100)
        self.network_throughput_range = (1.0, 4.27)

        self.state = initial_state if initial_state is not None else np.array([30, 200, 100, 4.27], dtype=np.float32)
        
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])

        self.observation_space = spaces.Box(
            low=np.array([
                self.service_throughput_range[0],
                self.service_latency_range[0],
                self.network_latency_range[0],
                self.network_throughput_range[0]
            ], dtype=np.float32),
            high=np.array([
                self.service_throughput_range[1],
                self.service_latency_range[1],
                self.network_latency_range[1],
                self.network_throughput_range[1]
            ], dtype=np.float32),
            dtype=np.float32
        )

        self.svc_throughput_step = 10
        self.svc_latency_step = 10
        self.net_latency_step = 10
        self.net_throughput_step = 0.2

        self.target_service_throughput_min = 30
        self.target_service_throughput_max = 200
        self.target_service_latency_max = 50
        self.target_network_latency_max = 100
        self.target_network_throughput_min = 1.0
        self.e2e_latency_max = 200

        self.max_steps = 30
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([30, 200, 100, 4.27], dtype=np.float32)
        self.current_step = 0
        return self.state.copy(), {}

    def step(self, action):
        svc_throughput, svc_latency, net_latency, net_throughput = self.state

        svc_throughput = self._apply_action(svc_throughput, action[0], self.service_throughput_range, self.svc_throughput_step)
        svc_latency = self._apply_action(svc_latency, action[1], self.service_latency_range, self.svc_latency_step)
        net_latency = self._apply_action(net_latency, action[2], self.network_latency_range, self.net_latency_step)
        net_throughput = self._apply_action(net_throughput, action[3], self.network_throughput_range, self.net_throughput_step)

        min_service_latency = net_latency / 2
        if svc_latency < min_service_latency:
            svc_latency = min_service_latency

        self.state = np.array([svc_throughput, svc_latency, net_latency, net_throughput], dtype=np.float32)
        self.current_step += 1

        reward = self._calculate_reward(svc_latency, net_latency, svc_throughput, net_throughput)

        done = bool(self.current_step >= self.max_steps or reward >= 7.0)

        return self.state.copy(), float(reward), done, False, {}

    def _apply_action(self, current_value, action, value_range, step):
        if action == 0:
            return max(value_range[0], current_value - step)
        elif action == 2:
            return min(value_range[1], current_value + step)
        return current_value

    def _calculate_reward(self, svc_latency, net_latency, svc_throughput, net_throughput):
        reward = 0.0
        total_latency = svc_latency + net_latency

        if total_latency > self.e2e_latency_max:
            reward -= self.PENALTY_FACTOR * ((total_latency - self.e2e_latency_max) / self.e2e_latency_max)

        if svc_latency > self.target_service_latency_max:
            reward -= self.PENALTY_FACTOR * ((svc_latency - self.target_service_latency_max) / self.target_service_latency_max)
        else:
            reward += 1.0

        if net_latency > self.target_network_latency_max:
            reward -= self.PENALTY_FACTOR * ((net_latency - self.target_network_latency_max) / self.target_network_latency_max)
        else:
            reward += 0.5

        if svc_throughput < self.target_service_throughput_min:
            reward -= self.PENALTY_FACTOR * ((self.target_service_throughput_min - svc_throughput) / self.target_service_throughput_min)
        else:
            reward += 1.0

        if net_throughput < self.target_network_throughput_min:
            reward -= self.PENALTY_FACTOR * ((self.target_network_throughput_min - net_throughput) / self.target_network_throughput_min)
        else:
            reward += 0.5

        if (total_latency <= self.e2e_latency_max and
            svc_latency <= self.target_service_latency_max and
            net_latency <= self.target_network_latency_max and
            svc_throughput >= self.target_service_throughput_min and
            svc_throughput <= self.target_service_throughput_max and
            net_throughput >= self.target_network_throughput_min):
            reward += self.BONUS_FOR_CONSTRAINT_SATISFACTION

        return reward