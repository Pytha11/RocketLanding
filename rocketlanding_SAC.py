import gymnasium as gym
import PyFlyt.gym_envs
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

env_id = "PyFlyt/Rocket-Landing-v3"
env = gym.make(env_id)
RENDER = False
EP_MAX = 500 * 4  # Maximum training episodes

# Define Ornstein-Uhlenbeck Noise
ou_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(7),
    sigma=0.2 * np.ones(7),  # Moderate noise
    theta=0.15,  # Speed of reversion to mean
    dt=0.01  # Small time step for smooth changes
)
policy_kwargs = dict(
    net_arch=dict(
        pi=[1024, 256],       # Policy network layers
        qf=[1024, 256]        # Q-function (critic) network layers
    )
)
# Define SAC model
model = SAC("MlpPolicy", env, verbose=1, gamma=0.99, learning_rate=1e-4, buffer_size=10000, batch_size=256, action_noise=ou_noise,
            policy_kwargs=policy_kwargs, tensorboard_log="./sac_pyflyt_tensorboard/")

# Optional callbacks for evaluation and stopping training based on reward
eval_env = gym.make(env_id)
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0, verbose=1)
eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, eval_freq=5000, best_model_save_path="./sac_pyflyt_best_model/", verbose=1)

# Train the model
model.learn(total_timesteps=EP_MAX * 300, callback=eval_callback)
model.save("sac_pyflyt_model")
env.close()
