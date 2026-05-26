import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np
import os


api = wandb.Api()
names = ["new coordinator tests"]


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

    return upper1 , lower1


def compute_violations(df, devider):
    lower = (df["lat"] < 2000 / devider).sum()

    upper = (df["lat"]> 3000 / devider).sum()

    return upper , lower




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
            plot_labels.append(label)# if q == 1 else "")  # show name only once
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
        test_len=len(df)
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

        # Add label only once per method
        plot_label = label if not added_label[label] else None
        added_label[label] = True
        plt.plot(df.index, df["lat"], linewidth= width, alpha=0.8, color=color, label=plot_label, linestyle=style)

        first_mask_low = (df.index <= 164) & (df["lat"] < 2000)
        first_mask_high = (df.index <= 164) & (df["lat"] > 3000)
        plt.scatter(df.index[first_mask_low], df["lat"][first_mask_low], s=5, color=color)
        plt.scatter(df.index[first_mask_high], df["lat"][first_mask_high], s=5, color=color)

    plt.plot([0, 164], [2000, 2000], color="blue", linestyle="--", linewidth=1)
    plt.plot([0, 164], [3000, 3000], color="blue", linestyle="--", linewidth=1)

    plt.xlim([0, 164])
    test_len=test_len/4
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

def plot_actions(df, title, filename):
    plt.figure(figsize=(10, 4))
    plt.step(df.index, df["action"], where="post", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.yticks(sorted(df["action"].unique()))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


for name in names:
    PROJECT_PATH="tuw_lacki/"+name
    NAMES = ["no_coord agents: reactive:","TD3_context32500010 agents: reactive", "TD3_context32500010 agents: PPO","no_coord agents: PPO:", "reactive agents: PPO:"]

    labels = ["inCoord","Coord", "Expert", "DAs","Reactive"]

    colors = ["magenta", "cyan", "orange", "green", "purple"]
    alphas = [0.5, 0.5, 1, 0.5, 0.5]

    runs = api.runs(PROJECT_PATH)

    all_network_runs = []
    all_compute_runs = []
    all_latency_runs = []

    for run in runs:
        if not any(name in run.display_name for name in NAMES ):
            continue
        if "final" not in run.name:
            continue
        hist = run.history(samples=100000)
        len_hist=len(hist["reward"].dropna().reset_index(drop=True))
        # if len_hist<142:
        #     print(run.name)
        #     continue
        if 'no_coord agents: reactive' in run.display_name:
            run.display_name='Reactive'
            run.name='Reactive'
        elif "TD3_context32500010 agents: reactive" in run.display_name:
            run.display_name = "Coord"
            run.name='Coord'
        elif "TD3_context32500010 agents: PPO" in run.display_name:
            run.display_name = "inCoord"
            run.name='inCoord'
        elif "no_coord agents: PPO" in run.display_name:
            run.display_name = "DAs"
            run.name='DAs'
        elif "reactive agents: PPO" in run.display_name:
            run.display_name = "Expert"
            run.name='Expert'

        # NETWORK LATENCY
        df_net, result2 = load_run_data(
            run,
            latency_key="total_latency.network_latency",
            action_key="network_delay action"
        )
        all_network_runs.append((run.display_name, df_net))

        upper, lower = compute_violations(df_net,2)
        # print(f"{upper} upper violations, {lower} lower violations")
        with open("./results/network_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        upper, lower = compute_action_violations(df_net)
        # print(f"[NETWORK] action {run.name}: {upper} upper violations, {lower} lower violations")
        with open("./results/network_action_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        action_counts = df_net["action"].value_counts().sort_index()
        # print(f"Network action counts {action_counts}")
        with open("./results/network_action.txt", "a+") as f:
            f.write(f"{run.display_name}:{action_counts}\n")


        df_comp, global_result = load_run_data(
            run,
            latency_key="total_latency.compute_latency",
            action_key="compute_cpu action"
        )
        all_compute_runs.append((run.display_name, df_comp))
        all_latency_runs.append((run.display_name, global_result))
        action_counts = df_comp["action"].value_counts().sort_index()
        with open("./results/compute_action.txt", "a+") as f:
            f.write(f"{run.display_name}:{action_counts}\n")
        upper, lower = compute_violations(df_comp,2)
        with open("./results/compute_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        upper, lower = compute_violations(result2,1)

        # print(f"GLOBAL {upper} upper violations, {lower} lower violations")
        with open("./results/global_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

        upper, lower = compute_action_violations(df_comp)

        # print(f"[COMPUTE] action {run.name}: {upper} upper violations, {lower} lower violations")
        with open("./results/compute_action_violation.txt", "a+") as f:
            f.write(f"{run.display_name}:{lower}:{upper}\n")

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

