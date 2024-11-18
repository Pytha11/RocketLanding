import gymnasium as gym
import PyFlyt.gym_envs
import numpy as np
from stable_baselines3 import PPO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Environment setup
env_id = "PyFlyt/Rocket-Landing-v3"
env = gym.make(env_id, render_mode='human')
# env = gym.make(env_id)
NUM_TEST_EPISODES = 1  # Number of test episodes

# Load the trained model
model = PPO.load("ppo_pyflyt_model")

# Function to test the model performance
def test_model(model, env, num_episodes=NUM_TEST_EPISODES, render=False):
    total_rewards = []  # List to store total rewards for each episode

    for episode in range(num_episodes):
        obs, _ = env.reset()  # Reset the environment at the start of each episode
        done = False
        episode_reward = 0
        reward_list = []
        c = 0
        info = {'line_vel': [0, 0, 0]}
        while not done:
            c += 1
            action, _states = model.predict(obs, deterministic=True)  # Get action from the model
            obs, reward, done, _, info = env.step(action)  # Step in the environment
            print(action, info['line_vel'], info['line_pos'])
            episode_reward += reward  # Accumulate the reward for the episode
            reward_list.append(reward)
        total_rewards.append(episode_reward)  # Store the total reward for this episode
        print(c)
    avg_reward = np.mean(total_rewards)  # Compute the average reward across episodes
    print(f"Average reward over {num_episodes} test episodes: {avg_reward}")

    return total_rewards, avg_reward


# Test the model
test_rewards, avg_test_reward = test_model(model, env, num_episodes=NUM_TEST_EPISODES)

# Save the test results (rewards)
np.save("test_rewards.npy", test_rewards)  # Save individual episode rewards
np.save("avg_test_reward.npy", avg_test_reward)  # Save the average test reward

# Optionally print out the test results
print(f"Test Results - Average Reward: {avg_test_reward}")
