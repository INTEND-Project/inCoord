import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
# Run multiple tests to see how latency values change over time


timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# # Number of test cases
# num_tests = 100

# # Store results
# results = []
#
# for _ in range(num_tests):
#     # Generate random latencies for each test case
#     edge_to_cloud_latency = np.random.normal(200, 200) / 1000  # Convert to seconds
#     edge_to_client_latency = np.random.normal(50, 600) / 1000  # Convert to seconds
#     network_latency = edge_to_cloud_latency + edge_to_client_latency
#
#     fetch_latency = np.random.normal(20, 120) / 1000  # Convert to seconds
#     processing_latency = np.random.normal(100, 600) / 1000  # Convert to seconds
#     queueing_latency = np.random.normal(120, 100) / 1000  # Convert to seconds
#     computing_latency = fetch_latency + processing_latency + queueing_latency
#
#     # Print results
#     print(f"Edge-to-Cloud Latency: {edge_to_cloud_latency:.3f} sec")
#     print(f"Edge-to-Client Latency: {edge_to_client_latency:.3f} sec")
#     print(f"Total Network Latency: {network_latency:.3f} sec\n")
#
#     print(f"Fetch Latency: {fetch_latency:.3f} sec")
#     print(f"Processing Latency: {processing_latency:.3f} sec")
#     print(f"Queueing Latency: {queueing_latency:.3f} sec")
#     print(f"Total Computing Latency: {computing_latency:.3f} sec")
#
#     # Store the results
#     results.append([
#         edge_to_cloud_latency, edge_to_client_latency, network_latency,
#         fetch_latency, processing_latency, queueing_latency, computing_latency
#     ])
#
# # Create a DataFrame for better visualization
# columns = [
#     "Edge-to-Cloud Latency (s)", "Edge-to-Client Latency (s)", "Total Network Latency (s)",
#     "Fetch Latency (s)", "Processing Latency (s)", "Queueing Latency (s)", "Total Computing Latency (s)"
# ]
# df_results = pd.DataFrame(results, columns=columns)

# Step 1: Fit a gamma distribution to match the observed shape
shape, loc, scale = 5, 1, 0.3  # Initial estimates based on histogram observation

# Generate synthetic total latency values
num_samples = 10000
total_latency_samples = np.random.gamma(shape, scale, num_samples) + loc  # Ensure shift to match range

# Step 2: Split total latency into network and computing latency
# Assume network latency is typically 40-60% of total latency
network_latency_fraction = np.random.uniform(0.4, 0.6, num_samples)
network_latency_samples = total_latency_samples * network_latency_fraction
computing_latency_samples = total_latency_samples - network_latency_samples

df_results = pd.DataFrame({"Total Network Latency (s)": network_latency_samples, "Total Computing Latency (s)": computing_latency_samples})

# Step 2: Compute throughput for each test case
chunk_size = 4 * 1024 * 1024 * 8

# Compute throughput based on total latency
df_results["Throughput (kbps)"] = chunk_size / (df_results["Total Network Latency (s)"] + df_results["Total Computing Latency (s)"]) / 1000  # Convert bps to kbps




# Plot the throughput results
plt.figure(figsize=(12, 6))
plt.plot(df_results.index, df_results["Throughput (kbps)"], label="Estimated Throughput", marker='o', linestyle='dashed')
plt.axhline(y=df_results["Throughput (kbps)"].mean(), color='r', linestyle='--', label=f"Avg Throughput: {df_results['Throughput (kbps)'].mean():.2f} kbps")

plt.xlabel("Test Case")
plt.ylabel("Throughput (kbps)")
plt.title("Throughput Variability Across Test Cases")
plt.legend()
plt.grid(True)
#plt.show()

file_path = 'figures/throughput_variability'
versioned_file = f"{file_path}.{timestamp}"
plt.savefig(f"{versioned_file}.pdf", format="pdf")

# Step 3: Select the highest bitrate that fits within the available throughput
#bitrates = np.array([235, 375, 560, 750, 1050, 1750, 2350, 3000])  # Available bitrates (kbps)
bitrates = np.array([300, 700, 1000, 2000, 4000, 8000, 15000, 30000])
# Function to select the best bitrate
def select_bitrate(throughput, bitrates):
    valid_bitrates = bitrates[bitrates <= throughput]
    return max(valid_bitrates) if len(valid_bitrates) > 0 else min(bitrates)  # Fallback to lowest bitrate


# Apply bitrate selection
df_results["Selected Bitrate (kbps)"] = df_results["Throughput (kbps)"].apply(lambda x: select_bitrate(x, bitrates))

bitrates_to_res = {
    300: '240p',
    700: '240p',
    1000: '360p',
    2000: '480p',
    4000: '720p',
    8000: '1080p',
    15000: '1440p',
    30000: '2160p'
}

def select_resolution(bitrate, bitrates_to_res):
    resolution = bitrates_to_res[bitrate]
    return resolution

df_results["Resolution"] = df_results["Selected Bitrate (kbps)"].apply(lambda x: select_resolution(x, bitrates_to_res))


print(df_results)
input()
# Create a figure and a primary y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot throughput on the primary y-axis
ax1.plot(df_results.index, df_results["Throughput (kbps)"], label="Estimated Throughput", linestyle="dashed", alpha=0.7, color='b')
ax1.set_xlabel("Test Case")
ax1.set_ylabel("Throughput (kbps)", color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Create a secondary y-axis for the selected bitrate
ax2 = ax1.twinx()
ax2.plot(df_results.index, df_results["Selected Bitrate (kbps)"], label="Selected Bitrate", linewidth=2, marker='o', color='g')
ax2.axhline(y=df_results["Selected Bitrate (kbps)"].mean(), color='r', linestyle='--', label=f"Avg Bitrate: {df_results['Selected Bitrate (kbps)'].mean():.2f} kbps")
ax2.set_ylabel("Selected Bitrate (kbps)", color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Add title and legend
plt.title("Bitrate Adaptation Based on Throughput")
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
#plt.show()
file_path = 'figures/throughput_bitrate_AR'
versioned_file = f"{file_path}.{timestamp}"
plt.savefig(f"{versioned_file}.pdf", format="pdf")


file_path = '../data/simulations/throughput_bitrate/throughput_bitrate_AR'
versioned_file = f"{file_path}.{timestamp}"
df_results.to_csv(f'{versioned_file}.csv', index=False)