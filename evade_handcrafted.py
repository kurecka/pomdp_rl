
from pomdp_rl.envs.torch.env_wrappers import RewardMonitor, SkrlAdapter
from pomdp_rl.envs.storm.storm_vec_env import StormVecEnv

import gymnasium as gym
import random
import numpy as np

from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

num_envs = 1
seed = 42
set_seed(seed)
sketch_name = 'models/evade_n=5_r=23'

def reward_function(rewards):
    return rewards["goal"] - rewards["fail"]

env = StormVecEnv(f'/opt/learning/rl_src/{sketch_name}/sketch.templ', reward_function, num_envs=num_envs, allow_wrong_actions=True)
env.reset()

print(env.model.observation_labels)

env.reset()
for i in range(10):
    obs, rew, done, trunc, info = env.step(np.array([2]))
    print("\n", i)
    print(obs)
    print(f'allowed actions: {info["allowed_actions"]}')
    for label in info['labels']:
        print(f'{label}: {info["labels"][label]}')
for i in range(10):
    obs, rew, done, trunc, info = env.step(np.array([6]))
    print("\n", i+10)
    print(f'{obs=}')
    print(f'{rew=}')
    print(f'{done=}')
    print(f'allowed actions: {info["allowed_actions"]}')
    for label in info['labels']:
        print(f'{label}: {info["labels"][label]}')
    if done.any():
        break
