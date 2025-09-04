import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneObjectDetectionEnv(gym.Env):
    """
    Gymnasium-compatible environment for tuning system metrics to ensure real-time, lossless object detection classification delivery from drones to dashboard.
    
    Action Space:
    - [Service Latency, Service Throughput, Network Latency, Network Throughput]
    - Each can decrease (-1), maintain (0), or increase (+1) by step.
    
    Observation Space:
    - [Service Latency (ms), Service Throughput (req/s), Network Latency (ms), Network Throughput (MBps), Host CPU Usage (%), Switch Bandwidth (Mbps)]
    - All floats, normalized to their actual values.
    """
    
    # Define constants
    REWARD_LATENCY_THRESHOLD = 100
    REWARD_THROUGHPUT_THRESHOLD = 30
    REWARD_NET_LATENCY_THRESHOLD = 50
    REWARD_NET_THROUGHPUT_THRESHOLD = 4.27
    MAX_LATENCY = 200
    MIN_LATENCY = 50
    MAX_THROUGHPUT = 60
    MIN_THROUGHPUT = 10
    MAX_NET_LATENCY = 100
    MIN_NET_LATENCY = 10
    MAX_NET_THROUGHPUT = 10
    MIN_NET_THROUGHPUT = 1

    def __init__(self):
        super().__init__()

        self.obs_low = np.array([self.MIN_LATENCY, self.MIN_THROUGHPUT, self.MIN_NET_LATENCY, self.MIN_NET_THROUGHPUT, 30, 10], dtype=np.float32)
        self.obs_high = np.array([self.MAX_LATENCY, self.MAX_THROUGHPUT, self.MAX_NET_LATENCY, self.MAX_NET_THROUGHPUT, 90, 100], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])  # 0=decrease, 1=maintain, 2=increase

        self.svc_latency_step = 10     # ms
        self.svc_throughput_step = 2   # req/s
        self.net_latency_step = 10      # ms
        self.net_throughput_step = 0.5  # MBps

        self.max_steps = 50
        self.current_step = 0
        self.reset()

    def initialize_state(self):
        return np.array([
            200,  # Service Latency (ms)
            30,   # Service Throughput (req/s)
            100,  # Network Latency (ms)
            4.27, # Network Throughput (MBps)
            50,   # Host CPU Usage (%)
            50    # Switch Bandwidth (Mbps)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initialize_state()
        self.current_step = 0
        return self.state.copy(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Invalid action: {}".format(action))
        
        svc_latency, svc_throughput, net_latency, net_throughput, host_cpu, sw_bw = self.state

        # Apply action to configurable metrics
        svc_latency = self.adjust_metric(svc_latency, action[0], self.svc_latency_step, self.MIN_LATENCY, self.MAX_LATENCY)
        svc_throughput = self.adjust_metric(svc_throughput, action[1], self.svc_throughput_step, self.MIN_THROUGHPUT, self.MAX_THROUGHPUT)
        net_latency = self.adjust_metric(net_latency, action[2], self.net_latency_step, self.MIN_NET_LATENCY, self.MAX_NET_LATENCY)
        net_throughput = self.adjust_metric(net_throughput, action[3], self.net_throughput_step, self.MIN_NET_THROUGHPUT, self.MAX_NET_THROUGHPUT)

        # ---- Metric dependencies ----
        svc_latency += 0.1 * (net_latency - self.state[2])
        svc_latency = np.clip(svc_latency, self.MIN_LATENCY, self.MAX_LATENCY)

        max_possible_throughput = net_throughput * 8
        svc_throughput = min(svc_throughput, max_possible_throughput)

        host_cpu = np.clip(30 + 0.7 * ((svc_throughput - self.MIN_THROUGHPUT) / (self.MAX_THROUGHPUT - self.MIN_THROUGHPUT)) * 60, 30, 90)
        sw_bw = 50

        self.state = np.array([svc_latency, svc_throughput, net_latency, net_throughput, host_cpu, sw_bw], dtype=np.float32)

        reward, done = self.calculate_reward(svc_latency, svc_throughput, net_latency, net_throughput)
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self.state.copy(), float(reward), done, False, {}

    def adjust_metric(self, metric, action, step, min_value, max_value):
        if action == 0:
            return max(min_value, metric - step)
        elif action == 2:
            return min(max_value, metric + step)
        return metric

    def calculate_reward(self, svc_latency, svc_throughput, net_latency, net_throughput):
        reward = 0.0
        done = False

        if svc_latency <= self.REWARD_LATENCY_THRESHOLD:
            reward += 2.0
        else:
            reward -= 0.02 * (svc_latency - self.REWARD_LATENCY_THRESHOLD)

        if svc_throughput >= self.REWARD_THROUGHPUT_THRESHOLD:
            reward += 2.0
        else:
            reward -= 0.5 * (self.REWARD_THROUGHPUT_THRESHOLD - svc_throughput)

        if net_latency <= self.REWARD_NET_LATENCY_THRESHOLD:
            reward += 1.0
        else:
            reward -= 0.01 * (net_latency - self.REWARD_NET_LATENCY_THRESHOLD)

        if net_throughput >= self.REWARD_NET_THROUGHPUT_THRESHOLD:
            reward += 1.0
        else:
            reward -= 0.2 * (int(np.ceil((self.REWARD_NET_THROUGHPUT_THRESHOLD - net_throughput) * 10)))

        if svc_latency > self.MAX_LATENCY or svc_throughput < self.MIN_THROUGHPUT:
            reward -= 10.0
            done = True

        return reward, done

    def close(self):
        pass  # Implement resource release if needed