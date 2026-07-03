import gymnasium as gym
from gymnasium import spaces

import wandb
import random
from stable_baselines3 import PPO
import torch

import numpy as np


def set_seed(seed):
    pass
    # random.seed(seed) - this caused regular events, limited training ability! :(


counter = 0


# The resource agents have x rounds to adjust their resources
class ResourceAgent:
    def __init__(self, lower_bound, upper_bound, remainder):
        self.rng = random.Random(seed_counter)
        self.percent = self.rng.uniform(0.05, 0.95)

        self.define_latency_range(remainder)
        self.current_latency = (1 - self.percent) * self.service_latency_range[1] + self.percent * \
                               self.service_latency_range[0]

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step_size = self.rng.uniform(0.05, 0.1)  # this fixed issue, was also too general
        with open("a.txt", "a+") as f:
            f.write(f"step: {self.step_size}\n")

        self.model = PPO.load("../train_RA/train_for_latency/models/new_latency_model_32_0.0001_512")

    def define_latency_range(self, remainder):

        low_range = [0, 1000]  # random.randint(300, 1300)
        high_range = [1500, 2500]  # random.randint(1700, 2700)
        middle_range = [500, 1500]  # random.randint(1000, 2000)
        mini_low_range = [200, 700]  # random.randint(300, 400)
        mini_high_range = [1800, 2300]  # random.randint(2600, 2700)

        if remainder == 9:  # A1 has less than A2
            self.service_latency_range = mini_low_range
        elif remainder == 8:
            self.service_latency_range = high_range
        elif remainder == 7:  # A1 has more than A2
            self.service_latency_range = high_range
        elif remainder == 6:
            self.service_latency_range = mini_low_range
        elif remainder == 4 or remainder == 5:  # Balanced
            self.service_latency_range = middle_range
        elif remainder == 3:
            self.service_latency_range = mini_high_range
        elif remainder == 2:
            self.service_latency_range = low_range
        elif remainder == 1:
            self.service_latency_range = low_range
        elif remainder == 0:
            self.service_latency_range = mini_high_range

    def change_upper_lower_bound(self, lower, upper):
        self.lower_bound = lower
        self.upper_bound = upper

    def micro_decrease(self, step_size):
        self.percent = min(self.percent + step_size, 0.99)

    def micro_increase(self, step_size):
        self.percent = max(0.01, self.percent - step_size)

    def get_state_vector(self):
        state = [self.percent,
                 self.current_latency / self.upper_bound,
                 self.lower_bound / self.upper_bound,
                 self.upper_bound / self.upper_bound]
        return state

    def update_latency_range(self):
        drift = 0  # self.rng.uniform(-30, 30) # if we need more dynamic environment add drift
        self.service_latency_range[0] += drift
        self.service_latency_range[1] += drift
        # We can also add some spikes if needed, removed this, since the objective could not be reached with high spikes
        if self.rng.random() < 0.0:
            spike = self.rng.uniform(300, 600)
            self.service_latency_range[0] += spike
            self.service_latency_range[1] += spike
            with open("a.txt", "a+") as f:
                f.write(f"SPIKE: {spike}\n")


    def run_microsteps(self, n_steps=3):
        self.update_latency_range()
        for _ in range(n_steps):
            obs = torch.tensor(self.get_state_vector(), dtype=torch.float32, device="cpu")
            action, _ = self.model.predict(obs, deterministic=True)
            if action == 0:
                self.micro_decrease(self.step_size)
            elif action == 2:
                self.micro_increase(self.step_size)
            self.current_latency = (1 - self.percent) * self.service_latency_range[1] + self.percent * \
                                   self.service_latency_range[0]
        return self.current_latency


nr_agents = 2
seed_counter = 0


class General_Env(gym.Env):
    def __init__(self, microsteps):
        wandb.init(project=f'Train Coordinator')
        super().__init__()
        self.agents = []
        self.microsteps = microsteps
        self.target_efficiency_limit = (2000, 2000)  # this is what we want to achieve
        self.lower_limit = None
        self.upper_limit = None
        self.current_percentages = [0.5, 0.5]
        self.old_percentages = [0.5, 0.5]
        self.current_latencies = [0, 0]
        self.old_state = []
        self.steps = 0
        self.max_steps = 100
        low = np.array([
            0,  # last action
            0,
            -np.inf,  # local nw
            -np.inf,  # local comp
            -np.inf  # global
        ], dtype=np.float32)
        high = np.array([
            1,
            1,
            np.inf,
            np.inf,
            np.inf
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32))
        self.total_reward = 0

    def normalize_state(self, x, lower, upper):
        if upper - lower == 0:
            return 0
        return (x - lower) / (upper - lower)

    def reset(self, *, seed=None, options=None):
        global seed_counter
        global counter
        print(f"RESET {seed_counter}")
        self.steps = 0
        set_seed(seed_counter)
        self.state = self.observation_space.sample()
        low, high = self.target_efficiency_limit
        self.current_percentages = [0.5, 0.5]
        self.old_percentages = [0.5, 0.5]
        self.total_reward = 0
        self.agents = []
        self.lower_limit = random.randint(low, high)
        self.upper_limit = self.lower_limit + 800
        self.max_steps = 200
        for agent in range(nr_agents):
            seed_counter += 1
            set_seed(seed_counter)
            remainder = counter % 10
            new_agent = ResourceAgent(self.lower_limit * self.current_percentages[agent],
                                      self.upper_limit * self.current_percentages[agent], remainder)
            self.agents.append(new_agent)
            counter += 1
            self.current_latencies[agent] = self.agents[agent].current_latency
            self.state[agent + 2] = self.normalize_state(self.agents[agent].current_latency,
                                                         self.lower_limit * self.current_percentages[agent],
                                                         self.upper_limit * self.current_percentages[agent])
            with open("a.txt", "a+") as f:
                f.write(f"{remainder}: {new_agent.service_latency_range}\n")
        with open("a.txt", "a+") as f:
            f.write(f"{self.lower_limit}-{self.upper_limit}\n")
        self.state[nr_agents + 2] = self.normalize_state(sum(self.current_latencies), self.lower_limit,
                                                         self.upper_limit)
        self.state[0] = 0.5
        self.state[1] = 0.5
        self.old_state = self.state

        return self.state.copy(), {}

    def normalize_percentage(self, action):
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

        self.current_percentages = [a1, a2]

    def step(self, action):
        assert self.action_space.contains(action)
        self.normalize_percentage(action)
        old_state = self.state.copy()
        for a in range(nr_agents):
            self.agents[a].change_upper_lower_bound(self.lower_limit * self.current_percentages[a],
                                                    self.lower_limit * self.current_percentages[a] + (
                                                                self.upper_limit - self.lower_limit) / 2)
            self.current_latencies[a] = self.agents[a].run_microsteps(self.microsteps)
            self.state[a + 2] = self.normalize_state(self.current_latencies[a],
                                                     self.lower_limit * self.current_percentages[a],
                                                     self.lower_limit * self.current_percentages[a] + (
                                                                 self.upper_limit - self.lower_limit) / 2)

        self.state[nr_agents + 2] = self.normalize_state(sum(self.current_latencies), self.lower_limit,
                                                         self.upper_limit)
        # the next state is: what percentage we had in addition to what we have after this percentages
        self.state[0] = self.current_percentages[0]
        self.state[1] = self.current_percentages[1]
        self.steps += 1
        self.max_steps -= 1
        reward, done = self.calculate_reward()
        self.total_reward += reward
        wandb.log({"return": self.total_reward})
        wandb.log({"a1 state": self.state[0]})
        wandb.log({"a2 state": self.state[1]})
        wandb.log({"a1 percentage": self.current_percentages[0]})
        wandb.log({"a2 percentage": self.current_percentages[1]})
        wandb.log({"reward": reward})
        self.old_state = self.state.copy()
        self.old_percentages = self.current_percentages.copy()
        return self.state.copy(), float(reward), done, False, {}

    def fairness_reward(self):
        a1, a2 = self.current_percentages
        dist = ((a1 - 0.5) ** 2 + (a2 - 0.5) ** 2) ** 0.5
        fairness = 1 - dist * 2
        fairness = max(0, fairness)  # keep in [0,1]
        return fairness

    def slo_reward(self):
        s1 = self.state[2]
        s2 = self.state[3]

        diff = abs(s1 - s2)

        # Positive when balanced, negative when unbalanced
        # Smooth exponential decay
        reward = np.exp(-3 * diff) - 0.5 * diff

        return float(reward)

    def direction_reward(self):
        reward = 0.0
        for i in range(nr_agents):
            latency = self.state[i + 2]  # normalized latency
            pct = self.current_percentages[i]
            old_pct = self.old_percentages[i]
            delta = pct - old_pct

            # If latency too high -> we WANT delta > 0
            if latency > 1:
                reward += 5.0 * delta  # positive if we increased, negative if we decreased

            # If latency too low -> we WANT delta < 0
            elif latency < 0:
                reward += -5.0 * delta  # positive if we decreased, negative if we increased
        return reward

    def calculate_reward(self):
        c_slo_reward = self.slo_reward()
        c_fairness_reward = self.fairness_reward()
        c_direction_reward = self.direction_reward()
        osc_penalty = -abs(self.current_percentages[0] - self.old_percentages[0]) * 0.2
        # have we reached our goal? Are the resources distributed evenly + do they reach similar performance?
        reward = c_slo_reward + 0.5 * c_fairness_reward + 0.7 * c_direction_reward  # - 0.5*osc_penalty

        global counter

        done = False
        with open("a.txt", "a+") as f:
            f.write(f" {self.state} -> {reward} -> {self.steps}\n")

        if reward >= 0.5:
            if abs(self.state[2] - self.state[3]) <= 0.5:
                reward += 1
                done = True
                with open("a.txt", "a+") as f:
                    f.write(f" {self.state} -> {reward} -> {self.steps}\n")
        elif self.state[2] < 0 and self.state[3] < 0:
            if abs(self.state[2] - self.state[3]) <= 0.5:
                reward += 1
                done = True
                with open("a.txt", "a+") as f:
                    f.write(f"{self.state} -> {reward} -> {self.steps}\n")
        elif self.state[2] > 1 and self.state[3] > 1:
            if abs(self.state[2] - self.state[3]) <= 0.5:
                reward += 1
                done = True
                with open("a.txt", "a+") as f:
                    f.write(f"{self.state} -> {reward} -> {self.steps}\n")
        if self.max_steps <= 0:
            done = True
            with open("last.txt", "a+") as f:
                f.write(f"NOT done: {self.state} -> {reward} -> {self.steps}\n")
        if done == True:
            wandb.log({"last_return": self.total_reward})
            wandb.log({"efficiency_return": self.total_reward / self.steps})
            wandb.log({"steps": self.steps})
            with open("last.txt", "a+") as f:
                f.write(f"last: {self.state} -> {reward} -> {self.steps}\n")

        return reward, done
