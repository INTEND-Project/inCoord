import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()

entity = ""
project = ""
names=["no_direction", "no_fairness", "no_slo", "all"]
runs = api.runs(f"{entity}/{project}")
filtered_runs = []
for name in names:
    plt.figure(figsize=(9, 4))
    first=0
    for run in runs:
        if name in run.name:
            print(run.name)
            df = run.history(keys=["a2 percentage"]).dropna().reset_index(drop=True)
            df[name] = run.name
            print(len(df))
            if "_step" not in df.columns:
                df["_step"] = range(len(df))
            if first==0:
                plt.plot(df["_step"], df["a2 percentage"], label=run.name)
            else:
                plt.plot(df["_step"], df["a2 percentage"], label=run.name, alpha=0.3)

            first += 1
    plt.xlabel("Step")
    plt.ylabel("State")
    plt.tight_layout()

    plt.title("State over time for runs with no_direction")
    plt.savefig(f"{name}_percentage_plot.png")
    plt.close()
