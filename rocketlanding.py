import gymnasium as gym
import PyFlyt.gym_envs
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

env_id = "PyFlyt/Rocket-Landing-v3"
env = gym.make(env_id)
RENDER = False
EP_MAX = 500  # Maximum training episodes

# Define SAC model
model = SAC("MlpPolicy", env, verbose=1, gamma=0.9, learning_rate=3e-4, batch_size=128, tensorboard_log="./sac_pyflyt_tensorboard/")

# Optional callbacks for evaluation and stopping training based on reward
eval_env = env
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-10, verbose=1)
eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, eval_freq=5000, best_model_save_path="./sac_pyflyt_best_model/", verbose=1)

# Train the model
model.learn(total_timesteps=EP_MAX * 1000, callback=eval_callback)
model.save("sac_pyflyt_model")
env.close()
