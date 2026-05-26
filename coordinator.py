import threading
import time
from domain_agent import Agent
from utils import get_network_latency, get_compute_latency, get_total_latency_value, get_real_cpu_usage, \
    apply_compute_action, apply_network_action, get_total_delay
import wandb
import numpy as np
from flask import Flask, request, jsonify
import argparse
import requests
from stable_baselines3 import TD3, PPO
import torch
from stable_baselines3.common.logger import configure
import random

# === Config ===
LOOP_INTERVAL_SECONDS = 60

app = Flask(__name__)


def set_seed(seed):
    torch.manual_seed(seed)  # Set the seed for CPU
    torch.cuda.manual_seed(seed)  # Set the seed for the current GPU
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    np.random.seed(seed)  # Set the seed for NumPy
    random.seed(seed)
    # Ensures deterministic behavior with some operations
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SLO():
    def __init__(self, mode):
        self.mode = mode
        # =============== High level Metrics ===============
        if self.mode == "TD3":
            self.models = [TD3.load(f"./models/coord_256_0.0001_100000_60000_fr1")]
        elif self.mode == "TD3_context":
            self.models = [TD3.load(f"./models/coord_context_256_0.0001_100000_60000_fr1")]
        elif self.mode == "TD3_context5100001":
            self.models = [TD3.load(f"./models/context_reward_relative_5_256_0.0001_10000_1000_fr1")]

        elif self.mode == "TD3_context5100002":
            self.models = [TD3.load(f"./models/context_reward_relative_5_256_0.0001_10000_2000_fr1")]

        elif self.mode == "TD3_context510000":
            self.models = [TD3.load(f"./models/context_reward_relative_5_256_0.0001_10000_6000_fr1")]
        elif self.mode == "TD3_context5100000":
            self.models = [TD3.load(f"./models/context_reward_relative_5_256_0.0001_100000_60000_fr1")]
        elif self.mode == "TD3_dynamic":
            self.models = [TD3.load(f"./models/dynamic_coord_256_0.0001_100000_60000")]
        elif self.mode == "TD3_context_dynamic":
            self.models = [TD3.load(f"./models/dynamic_coord_context_256_0.0001_100000_60000")]
        elif self.mode == "PPOs":
            self.models = []
            for i in range(2):
                self.models.append(PPO.load(f"./models/PPO_coord_16_0.0001_100000_reward_action1.zip"))

        self.percentages = [0.5, 0.5]
        self.old_percentages = [0.5, 0.5]
        self.lower_bound = 2000
        self.upper_bound = 3000
        # We get the reward definitions from inSwitch
        self.goals = {"network_latency": {"lower": self.lower_bound * self.percentages[0],
                                          "upper": self.upper_bound * self.percentages[0]},
                      "compute_latency": {"lower": self.lower_bound * self.percentages[1],
                                          "upper": self.upper_bound * self.percentages[1]},
                      "total_latency": {"lower": self.lower_bound, "upper": self.upper_bound}}

        # First thing that we update to get the current state
        self.state = {"network_latency": None,
                      "compute_latency": None,
                      "total_latency": None}
        self.state_update = {"network_latency": get_network_latency,
                             "compute_latency": get_compute_latency,
                             "total_latency": get_total_latency_value}
        # =============== Low level Metrics ===============
        self.action_space = {"network_delay": {"lower": 50, "upper": 0},
                             "compute_cpu": {"lower": 100, "upper": 900}}
        self.state_action_mapping = {"network_latency": "network_delay", "compute_latency": "compute_cpu"}
        self.action_update = {
            "network_delay": {"func": apply_network_action, "params": {"action": None, "current_resources": None}},
            "compute_cpu": {"func": apply_compute_action, "params": {"action": None, "current_resources": None}}}
        self.average_state = []

    def normalize_one_state(self, x, lower, upper):
        if upper - lower == 0:
            return 0
        return (x - lower) / (upper - lower)

    def norm_obs(self):
        norm_states = []
        for key, update_fn in self.state_update.items():
            new_state = self.state[key]
            lower = self.goals[key]["lower"]
            upper = self.goals[key]["upper"]
            normalized_state = self.normalize_one_state(new_state, lower, upper)
            norm_states.append(normalized_state)
        return norm_states

    def update_state(self):
        time.sleep(20)  # wait until service is back up :)
        for key, update_fn in self.state_update.items():
            self.state[key] = update_fn()

    def update_SLOs(self, high_level_resource_metric, data):
        self.goals[high_level_resource_metric]["lower"] = data.get("lower",
                                                                   self.goals[high_level_resource_metric]["lower"])
        self.goals[high_level_resource_metric]["upper"] = data.get("upper",
                                                                   self.goals[high_level_resource_metric]["upper"])

    def update_goals(self):
        self.goals = {"network_latency": {"lower": self.lower_bound * self.percentages[0],
                                          "upper": self.upper_bound * self.percentages[0]},
                      "compute_latency": {"lower": self.lower_bound * self.percentages[1],
                                          "upper": self.upper_bound * self.percentages[1]},
                      "total_latency": {"lower": self.lower_bound, "upper": self.upper_bound}}

    def rule_reward(self, value, lower=None, upper=None):

        # # Case 1: Only upper bound
        if upper is not None and lower is None:
            print("CASE 1")
            if value > upper:
                return -(value - upper) / upper  # proportional
            return 2 - (value / upper)

            # Case 2: Only lower bound
        if lower is not None and upper is None:
            print("CASE 2")
            if value < lower:
                return -(lower - value) / lower  # proportional
            return 2 - (lower / value)  # if I am more above it is slightly better

            # Case 3: Range with midpoint target
        if lower is not None and upper is not None:
            midpoint = (lower + upper) / 2
            half_range = (upper - lower) / 2
            print(f"CASE 3 {lower}<{value}<{upper}")

            if value < lower:
                print("value < lower")
                return -0.5
                # return -(lower - value) / lower #min negative reward
            if value > upper:
                print("value > upper")
                return -0.5
                # return -(value - upper) / upper #min negative reward
            # Inside the range → reward increases near midpoint
            return 3 - abs(value - midpoint) / half_range  # Max positive reward
        return 0

    def compute_reward(self):
        total = 0
        new_rewards = {}
        for key, rule in self.goals.items():
            r = self.rule_reward(
                value=self.state[key],
                lower=rule.get("lower"),
                upper=rule.get("upper")
            )
            new_rewards[key] = r
            total += r
        return total, new_rewards


parser = argparse.ArgumentParser()

# Optional SLO parameters
parser.add_argument("--lower", type=float, default=1000)
parser.add_argument("--upper", type=float, default=1500)
parser.add_argument("--wandb_name", type=str, default="MVP tests")
parser.add_argument("--strategy", type=str, default="PPO")
parser.add_argument("--batch_size", type=int, default=0)
parser.add_argument("--epochs", type=int, default=0)
parser.add_argument("--save_model", type=int, default=0)
parser.add_argument("--mode", type=str, default="TD3")
parser.add_argument("--local_rounds", type=int, default=3)
parser.add_argument("--coord_epochs", type=int, default=10)

args = parser.parse_args()

SLOs = SLO(args.mode)


@app.route("/slo/update", methods=["POST"])
def update_slo():
    data = request.json
    metric = data.get("metric")

    if metric not in SLOs.goals:
        return jsonify({"error": "Metric not found"}), 404

    SLOs.update_SLOs(metric, data)

    return jsonify({"message": "SLO updated", "rewards": SLOs.goals})


def listen_to_SLO_updates():
    app.run(host='127.0.0.1', port=5000, threaded=True)


def normalize_percentage(action):
    raw = action
    total = np.sum(raw)
    if total < 1e-8:
        a1, a2 = 0.5, 0.5
    else:
        a1 = raw[0] / total
        a2 = raw[1] / total
    # min is 0.1
    scale = 0.8  # because (0.9 - 0.1) / 1 = 0.8
    offset = 0.1
    a1 = a1 * scale + offset
    a2 = a2 * scale + offset

    return [a1, a2]


def fairness_reward():
    a1, a2 = SLOs.percentages
    dist = ((a1 - 0.5) ** 2 + (a2 - 0.5) ** 2) ** 0.5
    fairness = 1 - dist * 2
    fairness = max(0, fairness)  # keep in [0,1]
    return fairness
def slo_reward(state):
    reward = 0.0
    for a in range(3):
        s = state[a]
        if s < 0:
            reward -= 1
        elif s > 1:
            reward -= 1
        else:
            reward += 1

    return reward


def calculate_coord_reward(s):
    reward = slo_reward(s) + fairness_reward()

    return reward


# === Main loop ===
def main_loop(wandb_name, strategy, args, local_rounds):
    wandb.init(project=f'{wandb_name}',
               name=f"Coordinator: {SLOs.mode} agents: {strategy}: bs:{args.batch_size} ep:{args.epochs} save? {args.save_model}")
    route_thread = threading.Thread(target=listen_to_SLO_updates)
    route_thread.start()
    time.sleep(LOOP_INTERVAL_SECONDS)

    print("starting the main loop")
    agents = []
    step = 1
    SLOs.update_state()
    for action in SLOs.action_update:
        state_key = next(k for k, v in SLOs.state_action_mapping.items() if v == action)
        agent = Agent(SLOs.goals[state_key], SLOs.state[state_key], state_key, SLOs.action_space[action],
                      SLOs.action_update[action], action, strategy, args.batch_size, args.epochs, args.save_model)
        agents.append(agent)
    # do this before we start
    for _ in range(2):
        wandb.log({"total_latency": SLOs.state})
        SLOs.update_state()  # update state
        for agent in agents:
            agent.update_state(SLOs.state)
            agent.perform_action()
        time.sleep(LOOP_INTERVAL_SECONDS)

    while True:
        obs = torch.tensor(list(SLOs.norm_obs()), dtype=torch.float32, device="cpu")
        if SLOs.mode == "TD3":
            action, e = SLOs.models[0].predict(obs,
                                               deterministic=True)  # after state update get new percentage recommendation
            SLOs.percentages = normalize_percentage(action)  # update percentages
        elif "TD3_context" in SLOs.mode:
            norm_obs = SLOs.norm_obs()
            context_obs = np.concatenate([SLOs.percentages, norm_obs])
            obs = torch.tensor(list(context_obs), dtype=torch.float32, device="cpu")
            action, e = SLOs.models[0].predict(obs,
                                               deterministic=True)  # after state update get new percentage recommendation
            SLOs.old_percentages = SLOs.percentages.copy()
            SLOs.percentages = normalize_percentage(action)  # update percentages
        elif SLOs.mode == "TD3_dynamic":
            action, e = SLOs.models[0].predict(obs,
                                               deterministic=True)  # after state update get new percentage recommendation
            SLOs.percentages = normalize_percentage(action)  # update percentages
        elif SLOs.mode == "PPOs":
            norm_obs = SLOs.norm_obs()
            for agent in range(2):
                a = np.atleast_1d(norm_obs[agent])
                b = np.atleast_1d(norm_obs[2])
                agent_obs = np.concatenate([a, b])
                obs = torch.tensor(list(agent_obs), dtype=torch.float32, device="cpu")
                action, e = SLOs.models[agent].predict(obs, deterministic=True)
                if action == 0:
                    SLOs.percentages[agent] -= 0.05
                elif action == 2:
                    SLOs.percentages[agent] += 0.05
                # CONSTRAINT
                total = SLOs.percentages[0] + SLOs.percentages[1]
                if total > 1:
                    SLOs.percentages[0] /= total
                    SLOs.percentages[1] /= total
        elif SLOs.mode == "no_coord":
            SLOs.percentages = [0.5, 0.5]
        elif SLOs.mode == "reactive":
            norm_obs = SLOs.norm_obs()
            if norm_obs[0] < 0 or norm_obs[1] > 1:
                SLOs.percentages[0] -= 0.05
                SLOs.percentages[1] += 0.05
            if norm_obs[0] > 1 or norm_obs[1] < 0:
                SLOs.percentages[0] += 0.05
                SLOs.percentages[1] -= 0.05

            # if norm_obs[0]<0 and norm_obs[1]>1:
            #     SLOs.percentages[0] -= 0.05
            #     SLOs.percentages[1] += 0.05
            # elif norm_obs[1]<0 and norm_obs[0]>1:
            #     SLOs.percentages[0] += 0.05
            #     SLOs.percentages[1] -= 0.05

        SLOs.update_goals()  # let agents know about new goals
        wandb.log({"action_network": SLOs.percentages[0]})
        wandb.log({"action_compute": SLOs.percentages[1]})
        wandb.log({"percentages": SLOs.percentages})
        for a in agents:
            a.update_limits(SLOs.goals)
        SLOs.average_state = []
        for _ in range(local_rounds):
            wandb.log({"total_latency": SLOs.state})
            SLOs.update_state()  # update state
            SLOs.average_state.append([
                SLOs.state["network_latency"],
                SLOs.state["compute_latency"],
                SLOs.state["total_latency"]
            ])
            for agent in agents:
                agent.update_state(SLOs.state)
                agent.perform_action()
            print("waiting")
            time.sleep(LOOP_INTERVAL_SECONDS)
        # clear data especially if we have many clients/long runs
        SLOs.update_state()  # update state
        SLOs.average_state.append([
            SLOs.state["network_latency"],
            SLOs.state["compute_latency"],
            SLOs.state["total_latency"]
        ])
        arr = np.array(SLOs.average_state)  # shape: (num_steps, 3)
        avg_network, avg_compute, avg_total = arr.mean(axis=0)
        with open("state_info.txt", "a") as f:
            f.write(
                f"{SLOs.old_percentages} - Observations: {avg_network, avg_compute, avg_total}, {SLOs.norm_obs()} -> before it was {obs}, so model predicted {SLOs.percentages}\n")

        # update the dict-based state
        SLOs.state["network_latency"] = avg_network
        SLOs.state["compute_latency"] = avg_compute
        SLOs.state["total_latency"] = avg_total

        requests.post("http://172.16.0.1:32663/clear_data")
        next_obs = torch.tensor(list(SLOs.norm_obs()), dtype=torch.float32)

        total_reward, rewards = SLOs.compute_reward()  # compute the reward
        wandb.log({"reward": total_reward})
        coord_reward = calculate_coord_reward(next_obs.numpy())
        wandb.log({"coord_reward": coord_reward})
        if "TD3" in SLOs.mode:
            final_action = np.asarray(action, dtype=np.float32)
            final_action = final_action.reshape(
                SLOs.models[0].replay_buffer.n_envs,
                SLOs.models[0].replay_buffer.action_dim
            )
            print("FINAL ACTION:", final_action)
            if "TD3_context" in SLOs.mode:
                print(next_obs)
                percentage_tensor = torch.tensor(SLOs.percentages)
                next_obs = torch.cat((percentage_tensor, next_obs), dim=0)
                print(next_obs)
            SLOs.models[0].replay_buffer.add(
                obs=obs.numpy(),  # old observation
                next_obs=next_obs.numpy(),
                action=final_action,
                reward=coord_reward,
                done=False,
                infos=[{}]
            )
        if True and "TD3" in SLOs.mode and args.coord_epochs != 0:  # step%16==0:
            new_logger = configure(folder=None, format_strings=["stdout"])
            SLOs.models[0].set_logger(new_logger)
            SLOs.models[0].train(batch_size=256, gradient_steps=args.coord_epochs)
        step += 1
        print("-" * 50)




if __name__ == "__main__":
    SLOs.lower_bound = args.lower
    SLOs.upper_bound = args.upper
    SLOs.goals = {"network_latency": {"lower": SLOs.lower_bound * SLOs.percentages[0],
                                      "upper": SLOs.upper_bound * SLOs.percentages[0]},
                  "compute_latency": {"lower": SLOs.lower_bound * SLOs.percentages[1],
                                      "upper": SLOs.upper_bound * SLOs.percentages[1]},
                  "total_latency": {"lower": SLOs.lower_bound,
                                    "upper": SLOs.upper_bound}}
    print(SLOs.goals)
    with open("state_info.txt", "a+") as f:
        f.write(f"*************{SLOs.mode}\n")
    set_seed(42)
    main_loop(args.wandb_name, args.strategy, args, args.local_rounds)
