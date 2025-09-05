import paramiko
import threading
import os
import requests

K8S_API_URL = "http://128.131.172.96:5050"

NETWORK_API_URL = "http://128.131.172.95:5050/network/delay"

def increase_k8s_latency(deployments):
    """Reduce CPU allocation (simulate worse service latency)"""
    for name, config in deployments.items():
        if config["cpu_limit_m"] > 100:
            config["cpu_limit_m"] -= 10
            _post_cpu_update(name, config["cpu_limit_m"], config["memory_limit"])

def decrease_k8s_latency(deployments):
    """Increase CPU allocation (simulate better service latency)"""
    for name, config in deployments.items():
        if config["cpu_limit_m"] <500:
            config["cpu_limit_m"] += 10
            _post_cpu_update(name, config["cpu_limit_m"], config["memory_limit"])

def _post_cpu_update(deployment_name, cpu_m, memory_limit):
    cpu_str = f"{cpu_m}m"
    payload = {
        "deployment_name": deployment_name,
        "new_cpu_limit": cpu_str,
        "new_memory_limit": memory_limit,
        "namespace": "default"
    }
    try:
        response = requests.post(f"{K8S_API_URL}/k8s/resources", json=payload)
        if response.status_code == 200:
            print(f"[+] Updated {deployment_name} to {cpu_str} CPU")
        else:
            print(f"[!] Failed to update CPU: {response.text}")
    except Exception as e:
        print(f"[!] Error updating CPU for {deployment_name}: {e}")

def update_network_latency(delay_ms):
    try:
        response = requests.post(NETWORK_API_URL, json={"delay_ms": delay_ms})
        if response.status_code == 200:
            print(f"[+] Network delay updated to {delay_ms}ms")
        else:
            print(f"[!] Failed to update delay: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"[!] HTTP error while updating delay: {e}")

def increase_network_latency(switch_delay_ms):
    """Increase network latency by 5ms."""
    switch_delay_ms += 1
    update_network_latency(switch_delay_ms)
    return switch_delay_ms

def decrease_network_latency(switch_delay_ms):
    """Decrease network latency by 5ms."""
    switch_delay_ms = max(0, switch_delay_ms - 1)  # Ensure delay does not go below 1
    update_network_latency(switch_delay_ms)
    return switch_delay_ms

