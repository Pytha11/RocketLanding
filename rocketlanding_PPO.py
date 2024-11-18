import gymnasium as gym
import PyFlyt.gym_envs
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import VecNormalize
import os
from wind_test import  simple_wind
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

env_id = "PyFlyt/Rocket-Landing-v3"
env = gym.make(env_id)

RENDER = False
EP_MAX = 500 # Maximum training episodes


# Define PPO model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-3, batch_size=256, ent_coef=0.02, tensorboard_log="./PPO_rocket_landing/")
# model = PPO.load("ppo_pyflyt_model")
model.set_env(env)
# Optional callbacks for evaluation and stopping training based on reward
eval_env = gym.make(env_id)
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1.5, verbose=1)
eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, eval_freq=500, best_model_save_path="./ppo_pyflyt_best_model/", verbose=1)

# Train the model
model.learn(total_timesteps=EP_MAX * 300, callback=eval_callback)
model.save("ppo_pyflyt_model")
env.close()
