import gym
from stable_baselines3 import PPO

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Test the agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
