import csv
import os
import time
import numpy as np
import requests
from stable_baselines3 import PPO

from central_management import (
    increase_k8s_latency,
    decrease_k8s_latency,
    increase_network_latency,
    decrease_network_latency
)

# === Load trained model ===
model = PPO.load("./models/resize/ppo_resize_model")

# === Config ===
LOOP_INTERVAL_SECONDS = 30
switch_delay_ms = 20  # Default network delay

deployments = {
    "microservice1-deployment": {
        "replicas": 1,
        "cpu_limit_m": 250,
        "memory_limit": "512Mi"
    }
}

LOG_FILE = "rl_runtime_log.csv"

# === Sensor functions ===
def get_service_latency_from_api(avg_n=10):
    try:
        url = "http://172.17.0.1:5005/get_processing_times/microservice1"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        times = data.get("processing_times", [])[-avg_n:]
        if not times:
            return 600.0
        latencies_ms = [parse_timedelta_str(t["processing_time"]) for t in times]
        latencies_ms = [l for l in latencies_ms if l is not None]
        return np.mean(latencies_ms) if latencies_ms else 600.0
    except Exception as e:
        print(f"[!] Failed to get service latency: {e}")
        return 600.0

def parse_timedelta_str(t_str):
    try:
        h, m, s = t_str.split(":")
        seconds = float(s)
        minutes = int(m)
        hours = int(h)
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds * 1000  # ms
    except Exception as e:
        print(f"[!] Failed to parse time string '{t_str}': {e}")
        return None

def get_total_latency(avg_n=10):
    try:
        with open("response_times.csv") as f:
            lines = f.readlines()[1:]
        if len(lines) < avg_n:
            return 850.0
        latencies = [float(row.strip().split(",")[3]) for row in lines[-avg_n:]]
        return np.mean(latencies)
    except Exception as e:
        print(f"[!] Failed to parse response_times.csv: {e}")
        return 850.0

# === Build state vector for ResizeImageLatencyEnv ===
def build_state_vector():
    service_latency_raw = get_service_latency_from_api()
    total_latency_raw = get_total_latency()
    network_latency_raw = total_latency_raw - service_latency_raw

    # Clip to env's bounds
    service_latency = np.clip(service_latency_raw, 150.0, 600.0)
    network_latency = np.clip(network_latency_raw, 20.0, 300.0)

    # Static values from env constants
    BANDWIDTH = 50.0
    SERVICE_CPU_LIMIT = 0.25
    SERVICE_MEMORY_LIMIT = 512.0
    HOST_CPU_USAGE = 10.0

    state = np.array([
        service_latency,
        network_latency,
        BANDWIDTH,
        SERVICE_CPU_LIMIT,
        SERVICE_MEMORY_LIMIT,
        HOST_CPU_USAGE
    ], dtype=np.float32)

    print(f"[State] Svc: {service_latency:.2f} ms | Net: {network_latency:.2f} ms | "
          f"BW: {BANDWIDTH} | CPU_Lim: {SERVICE_CPU_LIMIT} | Mem_Lim: {SERVICE_MEMORY_LIMIT} | Host_CPU: {HOST_CPU_USAGE}")
    return state, service_latency_raw, network_latency_raw, total_latency_raw

# === Apply MultiDiscrete action ===
def apply_action(action):
    global switch_delay_ms
    svc_action, net_action = action  # Each ∈ {-1,0,1} mapped from {0,1,2}

    # Service latency control
    if svc_action == 2:
        decrease_k8s_latency(deployments)
    elif svc_action == 0:
        pass  # maintain
    elif svc_action == 1:
        increase_k8s_latency(deployments)

    # Network latency control
    if net_action == 2:
        switch_delay_ms = decrease_network_latency(switch_delay_ms)
    elif net_action == 0:
        pass  # maintain
    elif net_action == 1:
        switch_delay_ms = increase_network_latency(switch_delay_ms)

    print(f"[Action] Service={svc_action} | Network={net_action}")

# === Logging ===
def log_step(step, service_latency, network_latency, total_latency, action):
    header = ["step", "service_latency_ms", "network_latency_ms", "total_latency_ms", "svc_action", "net_action"]
    write_header = not os.path.exists(LOG_FILE)
    try:
        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([
                step,
                round(service_latency, 2),
                round(network_latency, 2),
                round(total_latency, 2),
                action[0],
                action[1]
            ])
    except Exception as e:
        print(f"[!] Failed to write log: {e}")

# === Main loop ===
def main_loop():
    print("RL Runtime Adapter for ResizeImageLatencyEnv Started.")
    step = 0
    while True:
        state, svc_lat_raw, net_lat_raw, total_lat_raw = build_state_vector()
        action, _ = model.predict(state, deterministic=True)
        apply_action(action)
        step += 1
        log_step(step, svc_lat_raw, net_lat_raw, total_lat_raw, action)
        print("-" * 50)
        time.sleep(LOOP_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_loop()
