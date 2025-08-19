import pandas as pd
import numpy as np
from tabulate import tabulate
import datetime
import argparse
from stable_baselines3 import DQN, PPO
from grpo import GRPO
from thresholds_finder_train import LatencyOptimizationEnv

def load_model(model_type, model_path, env):
    if model_type in ['DQN', 'PPO']:
        model_cls = DQN if model_type == 'DQN' else PPO
        return model_cls.load(model_path)
    elif model_type == 'GRPO':
        model = GRPO(env)
        model.load(model_path)
        return model
    else:
        raise ValueError("Unsupported model type!")

def predict_action(model, model_type, obs):
    if model_type == 'GRPO':
        return model.predict(obs[0])  # GRPO expects 1D obs
    else:  # PPO, DQN
        action, _ = model.predict(obs)
        return action

def adaptive_latency_step(action, original_network_latency, original_computing_latency, original_throughput, target_throughput=15000, max_reduction=5, alpha=3):
    scaling_factor = np.exp(-alpha * (original_throughput / target_throughput))
    adaptive_step = max_reduction * scaling_factor

    new_network_latency = original_network_latency
    new_computing_latency = original_computing_latency

    if action == 1:
        new_network_latency = max(original_network_latency - adaptive_step, 0)
    elif action == 2:
        new_computing_latency = max(original_computing_latency - adaptive_step, 0)
    elif action == 3:
        new_network_latency = max(original_network_latency - adaptive_step, 0)
        new_computing_latency = max(original_computing_latency - adaptive_step, 0)

    return new_network_latency, new_computing_latency

def select_bitrate(throughput, bitrates):
    valid_bitrates = bitrates[bitrates <= throughput]
    return max(valid_bitrates) if valid_bitrates.size > 0 else min(bitrates)

def select_resolution(bitrate):
    bitrates_to_res = {
        300: '240p', 700: '240p', 1000: '360p', 2000: '480p',
        4000: '720p', 8000: '1080p', 15000: '1440p', 30000: '2160p'
    }
    return bitrates_to_res[bitrate]

def test_model(model_type, model_path, data_path, num_samples=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    df = pd.read_csv(data_path)
    env = LatencyOptimizationEnv(df)

    if num_samples is None:
        num_samples = df.shape[0]

    model = load_model(model_type, model_path, env)

    # low_res_samples = df[df["Selected Bitrate (kbps)"] < 15000]
    # high_res_samples = df[df["Selected Bitrate (kbps)"] >= 15000]

    # test_samples = pd.concat([
    #     low_res_samples.sample(num_samples // 2, random_state=42),
    #     high_res_samples.sample(num_samples - num_samples // 2, random_state=42)
    # ]).sample(frac=1, random_state=42)

    actions_map = {
        0: "Keep Latency",
        1: "Decrease Network Latency",
        2: "Decrease Computing Latency",
        3: "Decrease Both Latencies",
    }

    results = []
    bitrates = np.array([300, 700, 1000, 2000, 4000, 8000, 15000, 30000])
    chunk_size = 4 * 1024 * 1024 * 8

    for _, row in df.iterrows():
        original_network_latency = row["Total Network Latency (s)"]
        original_computing_latency = row["Total Computing Latency (s)"]
        original_throughput = row["Throughput (kbps)"]

        obs = np.array([
            original_network_latency,
            original_computing_latency,
            original_throughput,
            row["Selected Bitrate (kbps)"]
        ], dtype=np.float32).reshape(1, -1)

        optimal_action = predict_action(model, model_type, obs)
        optimal_action = int(optimal_action)

        new_network_latency, new_computing_latency = adaptive_latency_step(
            optimal_action, original_network_latency, original_computing_latency, original_throughput
        )

        new_throughput = chunk_size / (new_network_latency + new_computing_latency) / 1000
        new_bitrate = select_bitrate(new_throughput, bitrates)
        new_resolution = select_resolution(new_bitrate)

        results.append({
            "Original Network Latency": original_network_latency,
            "Original Computing Latency": original_computing_latency,
            "Original Throughput (kbps)": original_throughput,
            "Original Selected Bitrate (kbps)": row["Selected Bitrate (kbps)"],
            "Original Resolution": row["Resolution"],
            "Recommended Action": actions_map[optimal_action],
            "New Network Latency": new_network_latency,
            "New Computing Latency": new_computing_latency,
            "New Throughput (kbps)": new_throughput,
            "New Selected Bitrate (kbps)": new_bitrate,
            "New Resolution": new_resolution,
        })

    results_df = pd.DataFrame(results)
    results_path = f'results/threshold_finder_test.{model_type.lower()}.{timestamp}.csv'
    results_df.to_csv(results_path, index=False)

    print(tabulate(results_df, headers='keys', tablefmt='psql'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=['DQN', 'PPO', 'GRPO'], help="Model type")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--data_path", type=str, default="../data/simulations/throughput_bitrate/throughput_bitrate.20250210074918.csv", help="Path to test data")
    args = parser.parse_args()

    test_model(args.model_type, args.model_path, args.data_path)