import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneObjectDetectionRealTimeEnv(gym.Env):
    HARD_CONSTRAINT_PENALTY = 100.0
    THROUGHPUT_PENALTY = 50.0
    LATENCY_PENALTY = 0.5
    MAX_STEPS = 50
    CLASSIFICATION_SIZE = 2 / 1024  # 2 KB in MB
    CLASSIFICATION_RATE = 30  # req/s
    
    def __init__(self):
        super().__init__()

        self.sli_names = [
            "service_latency",      
            "service_throughput",   
            "network_latency",      
            "network_throughput",   
        ]

        low = np.array([10, 10, 10, 0.01], dtype=np.float32)
        high = np.array([200, 60, 100, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])

        self.state = None
        self.reset()

        self.sli_steps = {
            "service_latency":  [-10, 0, 10],       
            "service_throughput": [-5, 0, 5],       
            "network_latency":  [-10, 0, 10],       
            "network_throughput": [-0.5, 0, 0.5],   
        }

        self.target_service_latency = 20     
        self.target_network_latency = 20     
        self.target_service_throughput = 30  
        self.target_network_throughput = 0.06 

        self.max_service_latency = 200       
        self.min_service_latency = 10        
        self.max_service_throughput = 60     
        self.min_service_throughput = 10     
        self.max_network_latency = 100       
        self.min_network_latency = 10        
        self.max_network_throughput = 10.0   
        self.min_network_throughput = 0.01   

        self.latency_constraint = 200        
        self.critical_latency = 50           
        self.lossless_delivery = True        

        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=[10, 10, 10, 0.01], high=[200, 60, 100, 10.0]).astype(np.float32)
        self.current_step = 0
        return self.state.copy(), {}

    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}")

        done = False
        info = {}

        for idx, sli in enumerate(self.sli_names):
            delta = self.sli_steps[sli][action[idx]]
            self.state[idx] += delta

        self.state[0] = np.clip(self.state[0], self.min_service_latency, self.max_service_latency)
        self.state[1] = np.clip(self.state[1], self.min_service_throughput, self.max_service_throughput)
        self.state[2] = np.clip(self.state[2], self.min_network_latency, self.max_network_latency)
        self.state[3] = np.clip(self.state[3], self.min_network_throughput, self.max_network_throughput)

        if self.state[1] > 45:
            self.state[0] = min(self.state[0] + 2, self.max_service_latency)
        if self.state[3] > 8.0:
            self.state[2] = min(self.state[2] + 5, self.max_network_latency)

        expected_bandwidth = self.CLASSIFICATION_SIZE * self.CLASSIFICATION_RATE
        net_utilization = expected_bandwidth / self.state[3]
        packet_loss = min(1.0, (net_utilization - 0.9) * 10) if net_utilization > 0.9 else 0.0

        end_to_end_latency = self.state[0] + 2 * self.state[2]

        reward = 0.0
        constraint_violated = False

        if self.state[1] < self.target_service_throughput:
            reward -= self.THROUGHPUT_PENALTY
            constraint_violated = True
        if self.state[3] < self.target_network_throughput:
            reward -= self.THROUGHPUT_PENALTY
            constraint_violated = True
        if packet_loss > 0.001:
            reward -= self.HARD_CONSTRAINT_PENALTY
            constraint_violated = True
        if end_to_end_latency > self.latency_constraint:
            reward -= self.HARD_CONSTRAINT_PENALTY
            constraint_violated = True

        reward -= self.LATENCY_PENALTY * abs(self.state[0] - self.target_service_latency)
        reward -= self.LATENCY_PENALTY * abs(self.state[2] - self.target_network_latency)
        if self.state[1] >= self.target_service_throughput:
            reward += 0.2 * (self.state[1] - self.target_service_throughput)
        if self.state[3] >= self.target_network_throughput:
            reward += 0.2 * (self.state[3] - self.target_network_throughput)

        if all([
            self.state[0] <= self.target_service_latency,
            self.state[2] <= self.target_network_latency,
            self.state[1] >= self.target_service_throughput,
            self.state[3] >= self.target_network_throughput,
            packet_loss == 0.0,
            end_to_end_latency <= self.latency_constraint
        ]):
            reward += 50.0

        self.current_step += 1
        if self.current_step >= self.MAX_STEPS:
            done = True

        return self.state.copy(), float(reward), done, False, {
            "end_to_end_latency": end_to_end_latency,
            "packet_loss": packet_loss,
            "net_utilization": net_utilization,
        }