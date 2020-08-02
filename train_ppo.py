import gym
import time
import numpy as np

# from gym_multigrid.envs import *
import gym_minigrid
from gym_minigrid.wrappers import *

from agents import PPOAgent

# Create environment
# env = gym.make('multigrid-empty-v0')
env = gym.make('MiniGrid-Empty-8x8-v0')
env = ImgObsWrapper(env)
# nb_agents = len(env.agents)

###################
# Hyperparameters #
###################
N_EPISODES = 300
LEARNING_RATE = 2.5e-4
HIDDEN_SIZES = [64,64]
TRAJECTORY_BUFFER_SIZE = 256
BATCH_SIZE = 32
# Other parameters
log_interval = 10

# Inputs and outputs of the envrionment
num_inputs = env.observation_space.sample().flatten().shape[0]
num_outputs = env.action_space.n 

# Create agent
agent = PPOAgent(num_inputs, num_outputs, HIDDEN_SIZES, LEARNING_RATE, TRAJECTORY_BUFFER_SIZE, BATCH_SIZE)

# Iterate over episodes
total_steps = 0
total_reward = 0

for episode_num in range(N_EPISODES):
    obs = env.reset()
    # obs = obs[0]  # Single agent
    done = False

    ep_steps = 0
    ep_reward = 0

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, done, _ = env.step(action)  # Single agent
        env.render(mode='human', highlight=True)
        # next_obs = next_obs[0]  # Single agent
        # reward = rewards [0]  # Single agent
        ep_reward += reward  # Single agent
        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done)

        obs = next_obs

        ep_steps +=1

    print(f'Episode num: {episode_num+1} | Episode steps: {ep_steps} | Episode Reward: {ep_reward}')
    if episode_num % log_interval == 0:
        pass

    # If buffer is full, then train agent
    if len(agent.replay_memory) >= TRAJECTORY_BUFFER_SIZE and len(agent.replay_memory) !=0:
        print('Training agent...')
        # for _ in range(TRAJECTORY_BUFFER_SIZE // BATCH_SIZE):
        agent.train()
        agent.replay_memory.clear()