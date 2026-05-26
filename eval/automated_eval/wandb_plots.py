import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np
import os

if os.path.exists("./results"):
    for f in os.listdir("results"):
        os.remove(os.path.join("results", f))
else:
    print("The file does not exist")
    os.mkdir("results")

api = wandb.Api()
names = ["new coordinator tests"]


def plot_all_runs_actions(all_runs_data, title, filename):
    plt.figure(figsize=(7, 3.5))
    desired_order = ["inCoord", "Coord", "Expert", "DAs", "Reactive"]
    all_runs_data.sort(key=lambda x: desired_order.index(x[0]))
    added_label = {"inCoord": False, "Coord": False, "Expert": False, "DAs": False, "Reactive": False}
    method_colors = {
        "inCoord": ("inCoord", "magenta", 1.2),
        "Coord": ("Coord", "cyan", 1),
        "Expert": ("Expert", "orange", 1),
        "DAs": ("DAs", "green", 0.8),
        "Reactive": ("Reactive", "purple", 0.8)
    }

    for run_name, df in all_runs_data:
        print(df)
        df.dropna().reset_index(drop=True)
        if "inCoord" in run_name:
            label, color, width = method_colors["inCoord"]
        elif "Coord" in run_name:
            label, color, width = method_colors["Coord"]
        elif "Expert" in run_name:
            label, color, width = method_colors["Expert"]
        elif "DAs" in run_name:
            label, color, width = method_colors["DAs"]
        else:
            label, color, width = method_colors["Reactive"]

        plot_label = label if not added_label[label] else None
        added_label[label] = True

        plt.plot(df.index, df["action"], linewidth=width, color=color, alpha=0.9, label=plot_label)

    plt.xlabel("Timesteps")
    plt.ylabel("Upper SLO limit")
    plt.grid(True)
    plt.xlim(0, 52)
    test_len = 164/ 3 / 4
    plt.axvline(1 * test_len, color="black", linestyle="--", linewidth=1)
    plt.axvline(2 * test_len, color="black", linestyle="--", linewidth=1)
    plt.axvline(3 * test_len, color="black", linestyle="--", linewidth=1)
    plt.xticks(
        [1 * test_len, 2 * test_len, 3 * test_len],
        ["e3", "e4", "e5"]
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def load_run_data(run, latency_key, action_key):
    df = run.history(samples=10000)
    lat = df[latency_key].dropna().reset_index(drop=True)[0:164]
    act = df[action_key].dropna().reset_index(drop=True)[0:164]
    act = act.iloc[1:].reset_index(drop=True)
    result = pd.concat([lat, act], axis=1)
    result.columns = ["lat", "action"]
    result2 = pd.DataFrame()
    result2["lat"] = df["total_latency.total_latency"].dropna().reset_index(drop=True)
    return result, result2


def plot_ma(key, title, df, name):
    plt.figure(figsize=(14, 7))
    values = df[key]

    plt.plot(values, label=f"{key}")

    low_idx = values[values < 2000].index
    high_idx = values[values > 3000].index

    plt.scatter(low_idx, values.loc[low_idx], color="blue", marker="x", s=50)
    plt.scatter(high_idx, values.loc[high_idx], color="orange", marker="x", s=50)

    plt.axhline(2000, color="red", linestyle="--", linewidth=2)
    plt.axhline(3000, color="red", linestyle="--", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Latency (MA)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scatter_1000_and_1500ms_{key}_{title}_{name}.pdf")
    plt.close()


def plot_violation_count_over_time(values, lower, upper, title, filename):
    violations = ((values < lower) | (values > upper)).astype(int)
    cumulative = violations.cumsum()

    plt.figure(figsize=(7, 3.5))
    plt.plot(cumulative, color="purple", linewidth=2)

    # plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Violations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compute_action_violations(df):
    lower1 = (
            (df["lat"][0:164] < 2000 / 2) &
            (df["action"][0:164] == 2) &
            (df["lat"][0:164].shift(-1) > 3000 / 2)
    ).sum()
    upper1 = (
            (df["lat"][0:164] > 3000 / 2) &
            (df["action"][0:164] == 0) &
            (df["lat"][0:164].shift(-1) < 2000 / 2)
    ).sum()

    return upper1, lower1


def compute_violations(df, devider):
    lower = (df["lat"] < 2000 / devider).sum()

    upper = (df["lat"] > 3000 / devider).sum()

    return upper, lower


def plot_violations(df, title, filename):
    plt.figure(figsize=(7, 4))
    plt.plot(df["lat"], label="Latency", color="blue")

    plt.scatter(df.index[df["upper_violation"]],
                df["lat"][df["upper_violation"]],
                color="red", label="Upper violation", s=50)

    plt.scatter(df.index[df["lower_violation"]],
                df["lat"][df["lower_violation"]],
                color="green", label="Lower violation", s=50)

    plt.axhline(2000, color="gray", linestyle="--")
    plt.axhline(3000, color="gray", linestyle="--")

    # plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Latency")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_all_runs_boxplot(all_runs_data, title, filename,
                          LOWER_BOUND=2000, UPPER_BOUND=3000):
    plt.figure(figsize=(6, 2))
    methods = [
        ("inCoord", "inCoord", "magenta"),
        ("Coord", "Coord", "cyan"),
        ("Expert", "Expert", "orange"),
        ("DAs", "DAs", "green"),
        ("Reactive", "Reactive", "purple")
    ]

    method_lookup = {m[0]: (m[1], m[2]) for m in methods}
    q_data = {1: {}, 2: {}, 3: {}, 4: {}}
    for run_name, df in all_runs_data:
        label, color = method_lookup[run_name]
        n = len(df)
        qsize = n // 4

        quartiles = [
            df["lat"].iloc[0:qsize],
            df["lat"].iloc[qsize:2 * qsize],
            df["lat"].iloc[2 * qsize:3 * qsize],
            df["lat"].iloc[3 * qsize:4 * qsize]
        ]

        for q in range(4):
            if label not in q_data[q + 1]:
                q_data[q + 1][label] = []
            q_data[q + 1][label].append(quartiles[q])

    for q in range(1, 5):
        for method in q_data[q]:
            q_data[q][method] = pd.concat(q_data[q][method])

    plot_data = []
    plot_labels = []
    plot_colors = []

    for q in range(1, 5):
        for key, label, color in methods:
            plot_data.append(q_data[q][label])
            plot_labels.append(label)  # if q == 1 else "")  # show name only once
            plot_colors.append(color)
    for i, data in enumerate(plot_data):
        color = plot_colors[i]

        box = plt.boxplot(
            data,
            widths=0.5,
            positions=[i + 1],
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color="black", linewidth=1.5)
        )

        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_alpha(0.5)
    plt.axvline(5 + 0.5, color="black", linestyle="--", linewidth=1)
    plt.axvline(10 + 0.5, color="black", linestyle="--", linewidth=1)
    plt.axvline(15 + 0.5, color="black", linestyle="--", linewidth=1)

    plt.plot([0, len(plot_data) + 1], [LOWER_BOUND, LOWER_BOUND],
             color="blue", linestyle="--", linewidth=1)
    plt.plot([0, len(plot_data) + 1], [UPPER_BOUND, UPPER_BOUND],
             color="blue", linestyle="--", linewidth=1)

    plt.xticks(range(1, len(plot_data) + 1), plot_labels,
               rotation=40, ha="right")

    y_top = plt.ylim()[1] * 1.02
    plt.text(5.5, y_top, "e3", ha="center", fontsize=10)
    plt.text(10.5, y_top, "e4", ha="center", fontsize=10)
    plt.text(15.5, y_top, "e5", ha="center", fontsize=10)

    plt.ylim(bottom=0)
    plt.xlim([0, len(plot_data) + 1])
    plt.ylabel("Response time (ms)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()


def plot_all_runs_scatter(all_runs_data, title, filename):
    plt.figure(figsize=(7, 3.5))

    added_label = {"inCoord": False, "Coord": False, "Expert": False, "DAs": False, "Reactive": False}
    method_colors = {
        "inCoord": ("inCoord", "magenta", "solid", 1.2),
        "Coord": ("Coord", "cyan", "solid", 1),
        "Expert": ("Expert", "orange", "solid", 1),
        "DAs": ("DAs", "green", "solid", 0.5),
        "Reactive": ("Reactive", "purple", "solid", 0.5)
    }
    for run_name, df in all_runs_data:
        test_len = len(df)
        print(run_name)
        print(df)
        if "inCoord" in run_name:
            label, color, style, width = method_colors["inCoord"]
        elif "Coord" in run_name:
            label, color, style, width = method_colors["Coord"]
        elif "Expert" in run_name:
            label, color, style, width = method_colors["Expert"]
        elif "DAs" in run_name:
            label, color, style, width = method_colors["DAs"]
        else:
            label, color, style, width = method_colors["Reactive"]

        plot_label = label if not added_label[label] else None
        added_label[label] = True
        plt.plot(df.index, df["lat"], linewidth=width, alpha=0.8, color=color, label=plot_label, linestyle=style)

        first_mask_low = (df.index <= 164) & (df["lat"] < 2000)
        first_mask_high = (df.index <= 164) & (df["lat"] > 3000)
        plt.scatter(df.index[first_mask_low], df["lat"][first_mask_low], s=5, color=color)
        plt.scatter(df.index[first_mask_high], df["lat"][first_mask_high], s=5, color=color)

    plt.plot([0, 164], [2000, 2000], color="blue", linestyle="--", linewidth=1)
    plt.plot([0, 164], [3000, 3000], color="blue", linestyle="--", linewidth=1)

    plt.xlim([0, 164])
    test_len = test_len / 4
    plt.axvline(1 * test_len, color="black", linestyle="--", linewidth=1)
    plt.axvline(2 * test_len, color="black", linestyle="--", linewidth=1)
    plt.axvline(3 * test_len, color="black", linestyle="--", linewidth=1)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Timesteps (min)", fontsize=16)
    plt.ylabel("Response time (ms)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


for name in names:
    PROJECT_PATH = "[name]/" + name
    NAMES = ["no_coord agents: reactive:", "TD3_context32500010 agents: reactive", "TD3_context32500010 agents: PPO",
             "no_coord agents: PPO:", "reactive agents: PPO:"]

    labels = ["inCoord", "Coord", "Expert", "DAs", "Reactive"]

    colors = ["magenta", "cyan", "orange", "green", "purple"]
    alphas = [0.5, 0.5, 1, 0.5, 0.5]

    runs = api.runs(PROJECT_PATH)

    all_network_runs = []
    all_compute_runs = []
    all_latency_runs = []

    for run in runs:
        if not any(name in run.display_name for name in NAMES):
            continue
        if "final" not in run.name:
            continue
        hist = run.history(samples=100000)
        len_hist = len(hist["reward"].dropna().reset_index(drop=True))
        if 'no_coord agents: reactive' in run.display_name:
            run.display_name = 'Reactive'
            run.name = 'Reactive'
        elif "TD3_context32500010 agents: reactive" in run.display_name:
            run.display_name = "Coord"
            run.name = 'Coord'
        elif "TD3_context32500010 agents: PPO" in run.display_name:
            run.display_name = "inCoord"
            run.name = 'inCoord'
        elif "no_coord agents: PPO" in run.display_name:
            run.display_name = "DAs"
            run.name = 'DAs'
        elif "reactive agents: PPO" in run.display_name:
            run.display_name = "Expert"
            run.name = 'Expert'

        # NETWORK LATENCY
        df_net, result2 = load_run_data(
            run,
            latency_key="total_latency.network_latency",
            action_key="network_delay upper_limit"
        )
        all_network_runs.append((run.display_name, df_net))

        upper, lower = compute_violations(df_net, 2)
        with open("./results/network_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        upper, lower = compute_action_violations(df_net)
        with open("./results/network_action_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        action_counts = df_net["action"].value_counts().sort_index()
        with open("./results/network_action.txt", "a+") as f:
            f.write(f"{run.display_name}:{action_counts}\n")

        df_comp, global_result = load_run_data(
            run,
            latency_key="total_latency.compute_latency",
            action_key="compute_cpu upper_limit"
        )
        all_compute_runs.append((run.display_name, df_comp))
        all_latency_runs.append((run.display_name, global_result))
        action_counts = df_comp["action"].value_counts().sort_index()
        with open("./results/compute_action.txt", "a+") as f:
            f.write(f"{run.display_name}:{action_counts}\n")
        upper, lower = compute_violations(df_comp, 2)
        with open("./results/compute_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        upper, lower = compute_violations(result2, 1)

        with open("./results/global_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        upper, lower = compute_action_violations(df_comp)

        with open("./results/compute_action_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")


    plot_all_runs_actions(all_network_runs,
                          title=f"Network Actions",
                          filename=f"{name}_network_actions.pdf")

    plot_all_runs_actions(all_compute_runs,
                          title=f"Compute Actions",
                          filename=f"{name}_compute_actions.pdf")
    plot_all_runs_scatter(
        all_network_runs,
        title="Network Response time Scatter",
        filename=f"{name}_all_runs_network_scatter.pdf"
    )

    plot_all_runs_scatter(
        all_compute_runs,
        title="Compute Response time Scatter",
        filename=f"{name}_all_runs_compute_scatter.pdf"
    )

    plot_all_runs_scatter(
        all_latency_runs,
        title="Response time Violations",
        filename=f"{name}_all_runs_total_scatter.pdf"
    )


    def normalize_values(values, low=2000, high=3000):
        values = np.array(values)

        normalized = np.where(
            (values >= low) & (values <= high),
            0,
            np.where(values < low, low - values, values - high)
        )

        return normalized


    def compute_cdf(values):
        norm_vals = normalize_values(values)
        sorted_vals = np.sort(norm_vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        return sorted_vals, cdf


    def plot_cdfs_from_df(df, filename):
        plt.figure(figsize=(5, 3))
        df = df[labels]
        for col, color in zip(df.columns, colors):
            x, y = compute_cdf(df[col].values)
            plt.plot(x, y, label=col, color=color, alpha=0.8)

        plt.xlabel("Normalized Response Time (ms)")
        plt.ylabel("CDF")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


    def merge_runs_by_method(all_runs_data):
        method_groups = {}
        for run_name, df in all_runs_data:
            if run_name not in method_groups:
                method_groups[run_name] = []
            method_groups[run_name].append(df["lat"])
        merged = {m: pd.concat(vals).reset_index(drop=True)
                  for m, vals in method_groups.items()}

        return pd.DataFrame(merged)


    merged_df = merge_runs_by_method(all_latency_runs)
    plot_cdfs_from_df(merged_df, filename=f"global_latency_cdf.pdf")

    plot_all_runs_boxplot(all_network_runs,
                          title="Network Response time Distribution",
                          filename=f"{name}_all_runs_network_boxplot.pdf")
    plot_all_runs_boxplot(
        all_compute_runs,
        title="Compute Response time Distribution",
        filename=f"{name}_all_runs_compute_boxplot.pdf"
    )

    plot_all_runs_boxplot(
        all_latency_runs,
        title="Response time Distribution",
        filename=f"{name}_all_runs_total_boxplot.pdf"
    )
