import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneObjectDetectionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])
        low = np.array([0.0, 0.5, 512.0, 50.0, 10.0, 10.0, 1.0], dtype=np.float32)
        high = np.array([1.0, 0.5, 512.0, 200.0, 120.0, 200.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.state = None
        self.step_count = 0
        self.reset()

        self.LATENCY_TARGET = 300.0
        self.LATENCY_UPPER = 500.0
        self.NET_THR_TARGET = 5.0
        self.SERV_THR_TARGET = 30.0
        self.SERV_LAT_TARGET = 150.0
        self.NET_LAT_TARGET = 80.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        host_cpu = np.random.uniform(0.05, 0.25)
        service_cpu_limit = 0.5
        service_mem_limit = 512.0
        service_latency = 200.0
        service_throughput = 30.0
        network_latency = 100.0
        network_throughput = 4.27
        self.state = np.array([host_cpu, service_cpu_limit, service_mem_limit,
                               service_latency, service_throughput,
                               network_latency, network_throughput], dtype=np.float32)
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        host_cpu, service_cpu_limit, service_mem_limit, serv_lat, serv_thr, net_lat, net_thr = self.state

        serv_lat_step, serv_thr_step, net_lat_step, net_thr_step = 10.0, 10.0, 10.0, 0.5
        
        actions = [serv_lat, serv_thr, net_lat, net_thr]
        steps = [serv_lat_step, serv_thr_step, net_lat_step, net_thr_step]
        limits = [(50.0, 200.0), (10.0, 120.0), (10.0, 200.0), (1.0, 10.0)]
        
        for idx, action_value in enumerate(action):
            if action_value == 0:
                actions[idx] = max(limits[idx][0], actions[idx] - steps[idx])
            elif action_value == 2:
                actions[idx] = min(limits[idx][1], actions[idx] + steps[idx])

        serv_lat, serv_thr, net_lat, net_thr = actions

        if serv_lat < 150.0 and serv_thr < 30.0:
            penalty = 30.0 - serv_thr
            serv_thr = max(serv_thr, 30.0 - penalty * 0.5)

        host_cpu = min(1.0, 0.10 + 0.005 * (serv_thr - 30.0) + np.random.uniform(-0.01, 0.01))

        if net_lat < 80.0 and net_thr < 5.0:
            net_thr = max(net_thr, 5.0 - (80.0 - net_lat) / 80.0)

        host_cpu = min(1.0, host_cpu + 0.002 * (net_thr - 4.27))

        serv_lat = np.clip(serv_lat + np.random.normal(0, 3), 50.0, 200.0)
        serv_thr = np.clip(serv_thr + np.random.normal(0, 1), 10.0, 120.0)
        net_lat = np.clip(net_lat + np.random.normal(0, 3), 10.0, 200.0)
        net_thr = np.clip(net_thr + np.random.normal(0, 0.1), 1.0, 10.0)
        host_cpu = np.clip(host_cpu, 0.0, 1.0)

        self.state = np.array([host_cpu, service_cpu_limit, service_mem_limit,
                               serv_lat, serv_thr, net_lat, net_thr], dtype=np.float32)

        end_to_end_latency = serv_lat + net_lat
        reward, terminated = 0.0, False

        if end_to_end_latency > 500:
            reward -= 100.0
            terminated = True
        else:
            latency_penalty = max(0, (end_to_end_latency - 300.0) / 10.0)
            reward -= latency_penalty
            if end_to_end_latency <= 300.0:
                reward += 10.0
            else:
                reward -= (end_to_end_latency - 300.0) * 0.1

        if serv_lat > self.SERV_LAT_TARGET:
            reward -= (serv_lat - self.SERV_LAT_TARGET) * 0.05
        if serv_thr >= self.SERV_THR_TARGET:
            reward += 2.0
        else:
            reward -= (self.SERV_THR_TARGET - serv_thr) * 0.2
        if net_lat > self.NET_LAT_TARGET:
            reward -= (net_lat - self.NET_LAT_TARGET) * 0.05
        if net_thr >= self.NET_THR_TARGET:
            reward += 2.0
        else:
            reward -= (self.NET_THR_TARGET - net_thr) * 0.2
        if host_cpu > 0.95:
            reward -= 5.0

        self.step_count += 1
        if self.step_count >= 50:
            terminated = True

        return self.state.copy(), float(reward), terminated, False, {}