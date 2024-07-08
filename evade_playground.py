from pomdp_rl.envs.torch.gridworld import Evade
from pomdp_rl.envs.torch.env_wrappers import RewardMonitor
from pomdp_rl.agents.torch.ppo import build_ppo_agent, build_ppo_mixed_agent, build_ppo_lstm_agent
from pomdp_rl.agents.torch.dqn import build_dqn_agent

from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# set_seed(42)


N = 5
radius = 2


res = []
for seed in [42]:
    set_seed(seed)
    num_envs = 256
    env = RewardMonitor(Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu'), step_punishment=0.001)
    agent = build_ppo_lstm_agent(env, cfg_override={'rollouts': 64, 'mini_batches': 32, 'ratio_clip': 0.05, 'value_clip': 0.1, 'discount_factor': 0.9999, 'learning_epochs': 1})
    cfg_trainer = {"timesteps": 100 * 1024 * 30 // num_envs, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
    trainer.train()
    trainer.eval()
    res.append(env.terminal_rewards.get_mean())
print(res)


# res = []
# for seed in [42]:
#     set_seed(seed)
#     num_envs = 256
#     env = RewardMonitor(Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu'), step_punishment=0.001)
#     agent = build_ppo_agent(env, cfg_override={'rollouts': 64, 'mini_batches': 32, 'ratio_clip': 0.05, 'value_clip': 0.1, 'discount_factor': 0.9999, 'learning_epochs': 1})
#     cfg_trainer = {"timesteps": 100 * 1024 * 20 // num_envs, "headless": True}
#     trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
#     trainer.train()
#     trainer.eval()
#     res.append(env.terminal_rewards.get_mean())
# print(res)
