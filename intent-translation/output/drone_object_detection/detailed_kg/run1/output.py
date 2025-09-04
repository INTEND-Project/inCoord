import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneObjectDetectionEnv(gym.Env):
    SERVICE_THROUGHPUT_LOW = 30
    SERVICE_THROUGHPUT_HIGH = 120
    SERVICE_LATENCY_LOW = 50
    SERVICE_LATENCY_HIGH = 200
    NETWORK_THROUGHPUT_LOW = 4.27
    NETWORK_THROUGHPUT_HIGH = 20.0
    NETWORK_LATENCY_LOW = 50
    NETWORK_LATENCY_HIGH = 100
    MAX_PODS = 12
    MIN_PODS = 6
    MAX_STEPS = 30

    def __init__(self):
        super().__init__()

        low = np.array([self.SERVICE_THROUGHPUT_LOW, self.SERVICE_LATENCY_LOW,
                        self.NETWORK_THROUGHPUT_LOW, self.NETWORK_LATENCY_LOW,
                        20, self.MIN_PODS], dtype=np.float32)
        high = np.array([self.SERVICE_THROUGHPUT_HIGH, self.SERVICE_LATENCY_HIGH,
                         self.NETWORK_THROUGHPUT_HIGH, self.NETWORK_LATENCY_HIGH,
                         100, self.MAX_PODS], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.n_metrics = 4
        self.action_space = spaces.MultiDiscrete([3] * self.n_metrics)

        self.state = np.array([self.SERVICE_THROUGHPUT_LOW, self.SERVICE_LATENCY_HIGH, 
                               self.NETWORK_THROUGHPUT_LOW, self.NETWORK_LATENCY_HIGH, 
                               50, 6], dtype=np.float32)
        self.steps = 0

        self.svc_throughput_step = 5
        self.svc_latency_step = 10
        self.net_throughput_step = 1.0
        self.net_latency_step = 5

        self.cpu_base = 30
        self.cpu_per_req = 0.2
        self.cpu_per_pod = 2.5

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(
            [self.SERVICE_THROUGHPUT_LOW, self.SERVICE_LATENCY_LOW, self.NETWORK_THROUGHPUT_LOW,
             self.NETWORK_LATENCY_LOW, 20, self.MIN_PODS],
            [self.SERVICE_THROUGHPUT_HIGH, self.SERVICE_LATENCY_HIGH, self.NETWORK_THROUGHPUT_HIGH,
             self.NETWORK_LATENCY_HIGH, 100, self.MAX_PODS]).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        svc_thr, svc_lat, net_thr, net_lat, cpu, pods = self.state
        d = [-1, 0, 1]
        svc_thr += d[action[0]] * self.svc_throughput_step
        svc_lat += d[action[1]] * self.svc_latency_step
        net_thr += d[action[2]] * self.net_throughput_step
        net_lat += d[action[3]] * self.net_latency_step

        svc_thr = float(np.clip(svc_thr, self.SERVICE_THROUGHPUT_LOW, self.SERVICE_THROUGHPUT_HIGH))
        svc_lat = float(np.clip(svc_lat, self.SERVICE_LATENCY_LOW, self.SERVICE_LATENCY_HIGH))
        net_thr = float(np.clip(net_thr, self.NETWORK_THROUGHPUT_LOW, self.NETWORK_THROUGHPUT_HIGH))
        net_lat = float(np.clip(net_lat, self.NETWORK_LATENCY_LOW, self.NETWORK_LATENCY_HIGH))

        if svc_thr > pods * 20 and pods < self.MAX_PODS:
            pods += 1
        elif svc_thr < pods * 10 and pods > self.MIN_PODS:
            pods -= 1

        cpu = self.cpu_base + self.cpu_per_req * (svc_thr - self.SERVICE_THROUGHPUT_LOW) + self.cpu_per_pod * (pods - self.MIN_PODS)
        cpu = float(np.clip(cpu, 20, 100))

        self.state = np.array([svc_thr, svc_lat, net_thr, net_lat, cpu, pods], dtype=np.float32)
        self.steps += 1

        constraints = {
            "svc_thr": svc_thr >= 40,
            "svc_lat": svc_lat <= 150,
            "net_thr": net_thr >= 5.0,
            "net_lat": net_lat <= 80,
        }
        all_constraints_met = all(constraints.values())

        out_of_bounds = (
            svc_thr < self.SERVICE_THROUGHPUT_LOW or svc_thr > self.SERVICE_THROUGHPUT_HIGH or
            svc_lat < self.SERVICE_LATENCY_LOW or svc_lat > self.SERVICE_LATENCY_HIGH or
            net_thr < self.NETWORK_THROUGHPUT_LOW or net_thr > self.NETWORK_THROUGHPUT_HIGH or
            net_lat < self.NETWORK_LATENCY_LOW or net_lat > self.NETWORK_LATENCY_HIGH or
            pods < self.MIN_PODS or pods > self.MAX_PODS
        )

        reward = 0.0
        if all_constraints_met:
            reward = 1.0
        else:
            reward -= 0.5 * (4 - sum(constraints.values()))
            total_latency = svc_lat + net_lat
            reward -= 0.1 * (total_latency / 1000)

        if cpu > 80:
            reward -= 0.2 * ((cpu - 80) / 10)

        if out_of_bounds:
            reward -= 2.0

        terminated = bool(self.steps >= self.MAX_STEPS or out_of_bounds)

        return self.state.copy(), float(reward), terminated, False, {}

    def render(self):
        print(f"Obs: {self.state}")

    def close(self):
        pass