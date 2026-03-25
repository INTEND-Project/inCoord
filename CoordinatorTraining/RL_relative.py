from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
import torch
from new_coord_env_relative import General_Env
import numpy as np
import os
import shutil
import wandb
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


steps=[256]
microsteps=[3] #Let's mimic actual env
learning_rates=[1e-4] #1e-5 too small
#agent will interact with env a total of this timestamps
total_timesteps=[25000] #Test different number of total timestamps
learning_starts=[10000] #Test this kind of exploration
for sigma in learning_starts:
    for microstep in microsteps:
        for step in steps:
            for learning_rate in learning_rates:
                for total_timestep in total_timesteps:

                    wandb.init(project=f'Train Coordinator TD3', name=f"ms{microstep}_st={step}_lr={learning_rate}_steps={total_timestep}_start_learning={sigma}_tf=1")
                    env = General_Env(microstep)
                    check_env(env, warn=True)
                    n_actions = env.action_space.shape[-1]
                    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
                    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
                    seed = 42
                    env.action_space.seed(seed)
                    env.observation_space.seed(seed)
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = TD3(
                        "MlpPolicy",
                        env,
                        #buffer_size=1000000, size of the replay buffer
                        # How many steps we explore before update is done
                        batch_size=step,  # The number of steps to run for each environment per update
                        verbose=1,  # logs for info messages (such as device or wrappers used),
                        #tau=0.005, gamma=0.99, I did not set these tau (float) – the soft update coefficient (“Polyak update”, between 0 and 1)
                        # gamma (float) – the discount factor
                        learning_rate=learning_rate,
                        train_freq=1, #update every round
                        seed=42,
                        learning_starts=sigma, # how many steps of the model to collect transitions for before learning starts
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    model.learn(total_timesteps=total_timestep) #The total number of samples (env steps) to train on Note: it is a lower bound,
                    model.save(f"../models/context_reward_relative_{microstep}_{step}_{learning_rate}_{total_timestep}_{sigma}_fr1")
                    wandb.finish()
                    path = os.path.expanduser('./wandb')
                    shutil.rmtree(path)

