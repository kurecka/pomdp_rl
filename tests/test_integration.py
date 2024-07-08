from pomdp_rl.agents.torch.ppo import build_ppo_lstm_agent, build_ppo_agent, build_ppo_mixed_agent
from pomdp_rl.agents.torch.dqn import build_dqn_agent
from pomdp_rl.agents.torch.ddqn import build_ddqn_agent

from pomdp_rl.envs.torch.gridworld import Evade, Refuel
from pomdp_rl.envs.torch.env_wrappers import RewardMonitor

from skrl.trainers.torch import SequentialTrainer


import pytest
import logging


# Decorator for disabling logging during tests


def disable_logging(fn):
    def wrapper(*args, **kwargs):
        print("Logging disabled")
        logging.getLogger("skrl").disabled = True
        fn(*args, **kwargs)
    return wrapper


@disable_logging
def test_ppo_agent():
    env = RewardMonitor(Evade([7,7], num_envs=2, device='cpu'), log_interval=100, step_punishment=0, max_steps=100)
    agent = build_ppo_agent(env)
    trainer = SequentialTrainer(env=env, agents=[agent], cfg={'timesteps': 300, 'headless': True})
    trainer.train()
    trainer.eval()


@disable_logging
def test_ppo_mixed_agent():
    env = RewardMonitor(Evade([7,7], num_envs=2, device='cpu'), log_interval=100, step_punishment=0, max_steps=100)
    agent = build_ppo_mixed_agent(env)
    trainer = SequentialTrainer(env=env, agents=[agent], cfg={'timesteps': 300, 'headless': True})
    trainer.train()
    trainer.eval()


@disable_logging
def test_ppo_lstm_agent():
    env = RewardMonitor(Evade([7,7], num_envs=2, device='cpu'), log_interval=100, step_punishment=0, max_steps=100)
    agent = build_ppo_lstm_agent(env)
    trainer = SequentialTrainer(env=env, agents=[agent], cfg={'timesteps': 300, 'headless': True})
    trainer.train()
    trainer.eval()


@disable_logging
def test_dqn_agent():
    env = RewardMonitor(Evade([7,7], num_envs=2, device='cpu'), log_interval=100, step_punishment=0, max_steps=100)
    agent = build_dqn_agent(env)
    trainer = SequentialTrainer(env=env, agents=[agent], cfg={'timesteps': 300, 'headless': True})
    trainer.train()
    trainer.eval()


@disable_logging
def test_ddqn_agent():
    env = RewardMonitor(Evade([7,7], num_envs=2, device='cpu'), log_interval=100, step_punishment=0, max_steps=100)
    agent = build_ddqn_agent(env)
    trainer = SequentialTrainer(env=env, agents=[agent], cfg={'timesteps': 300, 'headless': True})
    trainer.train()
    trainer.eval()


@disable_logging
def test_evade_env():
    env = RewardMonitor(Evade([7,7], num_envs=2, device='cpu'), log_interval=100, step_punishment=0, max_steps=100)
    agent = build_ppo_agent(env)
    trainer = SequentialTrainer(env=env, agents=[agent], cfg={'timesteps': 300, 'headless': True})
    trainer.train()
    trainer.eval()


@disable_logging
def test_refuel_env():
    env = RewardMonitor(Refuel(10, num_envs=3, device='cpu'), log_interval=100, step_punishment=0, max_steps=100)
    agent = build_ppo_agent(env)
    trainer = SequentialTrainer(env=env, agents=[agent], cfg={'timesteps': 300, 'headless': True})
    trainer.train()
    trainer.eval()