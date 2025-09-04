import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ObjectDetectionRealTimeEnv(gym.Env):
    """A custom OpenAI Gym environment for simulating an object detection system with real-time constraints."""
    
    # Constant for Mbps per detection
    DETECTION_BANDWIDTH = 0.05
    
    def __init__(self):
        super(ObjectDetectionRealTimeEnv, self).__init__()
        
        # Observation space: [Service Latency (ms), Service Throughput (infer/sec), Network Latency (ms), Network Throughput (Mbps), Packet Loss (%)]
        low_obs = np.array([20.0, 10.0, 20.0, 1.0, 0.0], dtype=np.float32)
        high_obs = np.array([200.0, 50.0, 200.0, 10.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Action space: for each configurable metric, 0=decrease, 1=maintain, 2=increase
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])

        # Metric indices
        self.idx_srv_lat = 0
        self.idx_srv_thr = 1
        self.idx_net_lat = 2
        self.idx_net_thr = 3
        self.idx_loss = 4
        
        # Ranges and steps
        self.srv_lat_range = (20.0, 200.0)
        self.srv_thr_range = (10.0, 50.0)
        self.net_lat_range = (20.0, 200.0)
        self.net_thr_range = (1.0, 10.0)

        self.srv_lat_step = 10.0
        self.srv_thr_step = 2.0
        self.net_lat_step = 10.0
        self.net_thr_step = 1.0

        # Initial state
        self.state = np.array([200.0, 30.0, 100.0, 4.0, 0.0], dtype=np.float32)
        self.max_steps = 50
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        self.state = np.random.uniform(low=[20.0, 10.0, 20.0, 1.0, 0.0], 
                                        high=[200.0, 50.0, 200.0, 10.0, 1.0]).astype(np.float32)
        self.current_step = 0
        return self.state.copy(), {}

    def step(self, action):
        """Take an action in the environment."""
        self.current_step += 1
        done = False

        srv_lat, srv_thr, net_lat, net_thr, loss = self.state

        # Apply actions to metrics
        srv_lat = self.update_metric(srv_lat, action[0], self.srv_lat_range, self.srv_lat_step)
        srv_thr = self.update_metric(srv_thr, action[1], self.srv_thr_range, self.srv_thr_step)
        net_lat = self.update_metric(net_lat, action[2], self.net_lat_range, self.net_lat_step)
        net_thr = self.update_metric(net_thr, action[3], self.net_thr_range, self.net_thr_step)

        # Update packet loss
        loss = self.update_packet_loss(loss, srv_thr, net_thr, net_lat)

        # Calculate service throughput cap based on network throughput
        actual_srv_thr_cap = min(srv_thr, (net_thr / self.DETECTION_BANDWIDTH))
        srv_thr = min(srv_thr, actual_srv_thr_cap)

        # Enforce observation bounds
        self.state = np.array([
            np.clip(srv_lat, *self.srv_lat_range),
            np.clip(srv_thr, *self.srv_thr_range),
            np.clip(net_lat, *self.net_lat_range),
            np.clip(net_thr, *self.net_thr_range),
            np.clip(loss, 0.0, 1.0)
        ], dtype=np.float32)

        # Calculate reward
        reward = self.calculate_reward(srv_lat, net_lat, loss, srv_thr, net_thr)

        # Determine if episode is done
        if loss >= 1.0 or self.current_step >= self.max_steps:
            done = True

        return self.state.copy(), float(reward), done, False, {}

    def update_metric(self, metric, action, metric_range, step):
        """Update a given metric based on the action taken."""
        if action == 0:
            return max(metric_range[0], metric - step)
        elif action == 2:
            return min(metric_range[1], metric + step)
        return metric

    def update_packet_loss(self, loss, srv_thr, net_thr, net_lat):
        """Update the packet loss based on throughput and latency metrics."""
        min_required_bandwidth = max(1.0, srv_thr * self.DETECTION_BANDWIDTH)
        if net_thr < min_required_bandwidth:
            loss = min(1.0, loss + 0.05 + 0.1 * (min_required_bandwidth - net_thr) / 10)
        else:
            loss = max(0.0, loss - 0.03)
        if net_lat > 100.0:
            loss = min(1.0, loss + 0.01 * ((net_lat - 100.0) / 100))
        return loss

    def calculate_reward(self, srv_lat, net_lat, loss, srv_thr, net_thr):
        """Calculate the reward based on various performance metrics."""
        e2e_latency = srv_lat + net_lat
        reward = 0.0

        if loss > 0.0:
            reward -= 100.0 * loss  
        if e2e_latency <= 150.0:
            reward += 5.0
        elif e2e_latency <= 300.0:
            reward += 1.0
        else:
            reward -= 5.0 + 0.1 * (e2e_latency - 300.0)
        if srv_thr >= 30.0:
            reward += 3.0
        else:
            reward -= 2.0 * (30.0 - srv_thr) / 30.0
        if net_thr >= 2.0:
            reward += 1.0
        else:
            reward -= 1.0
        reward -= 0.01 * (srv_lat + net_lat)
        if loss == 0.0 and e2e_latency <= 150.0 and srv_thr >= 30.0 and net_thr >= 2.0:
            reward += 10.0

        return reward