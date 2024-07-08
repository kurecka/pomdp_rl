from pomdp_rl.envs.torch.gridworld import Evade
from pomdp_rl.envs.torch.env_wrappers import RewardMonitor
from pomdp_rl.agents.torch.ppo import build_ppo_agent, build_ppo_mixed_agent, build_ppo_lstm_agent
from pomdp_rl.agents.torch.dqn import build_dqn_agent

from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


set_seed(44)


N = 7
radius = 2

# time: 00:24
# reward: 0.755, 0.735, 0.754
res = []
for seed in [42, 43, 44]:
    set_seed(seed)
    num_envs = 512
    env = RewardMonitor(Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu'), step_punishment=0.01)
    agent = build_ppo_lstm_agent(env, cfg_override={'rollouts': 8, 'mini_batches': 8, 'ratio_clip': 0.05, 'value_clip': 0.05})
    cfg_trainer = {"timesteps": 50 * 1024 * 10 // num_envs, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
    trainer.train()
    trainer.eval()
    res.append(env.terminal_rewards.get_mean())
print(res)


# # time: 00:24
# # reward: 0.754, 0.729, 0.762
# res = []
# for seed in [42, 43, 44]:
#     set_seed(seed)
#     num_envs = 512
#     env = RewardMonitor(Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu'), step_punishment=0.01)
#     agent = build_ppo_lstm_agent(env, cfg_override={'rollouts': 8, 'mini_batches': 8, 'ratio_clip': 0.025, 'value_clip': 0.025})
#     cfg_trainer = {"timesteps": 50 * 1024 * 10 // num_envs, "headless": True}
#     trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
#     trainer.train()
#     trainer.eval()
#     res.append(env.terminal_rewards.get_mean())
# print(res)


# # time: 00:24
# # reward: 0.768, 0.770, 0.750
# res = []
# for seed in [42, 43, 44]:
#     set_seed(seed)
#     num_envs = 512
#     env = RewardMonitor(Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu'), step_punishment=0.001)
#     agent = build_ppo_lstm_agent(env, cfg_override={'rollouts': 8, 'mini_batches': 8, 'ratio_clip': 0.025, 'value_clip': 0.025})
#     cfg_trainer = {"timesteps": 50 * 1024 * 10 // num_envs, "headless": True}
#     trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
#     trainer.train()
#     trainer.eval()
#     res.append(env.terminal_rewards.get_mean())
# print(res)


# # time: 01:04
# # reward: 0.793, 0.783, 0.787
# num_envs = 512
# env = RewardMonitor(Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu'), step_punishment=0.001)
# agent = build_ppo_lstm_agent(env, cfg_override={'rollouts': 8, 'mini_batches': 8, 'ratio_clip': 0.025, 'value_clip': 0.025})
# cfg_trainer = {"timesteps": 100 * 1024 * 10 // num_envs, "headless": True}
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
# trainer.train()
# trainer.eval()


# # time: 00:55
# # reward: 0.767, 0.750, 0.7857
# num_envs = 512
# env = RewardMonitor(Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu'), step_punishment=0.001)
# agent = build_ppo_lstm_agent(env, cfg_override={'rollouts': 8, 'mini_batches': 8, 'ratio_clip': 0.02, 'value_clip': 0.02})
# cfg_trainer = {"timesteps": 100 * 1024 * 10 // num_envs, "headless": True}
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
# trainer.train()
# trainer.eval()


# # time: 00:55
# # reward: 0.771
# num_envs = 512
# env = RewardMonitor(Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu'), step_punishment=0.01)
# agent = build_ppo_lstm_agent(env, cfg_override={'rollouts': 8, 'mini_batches': 8, 'ratio_clip': 0.01, 'value_clip': 0.01})
# cfg_trainer = {"timesteps": 100 * 1024 * 10 // num_envs, "headless": True}
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
# trainer.train()
# trainer.eval()


# # time: 00:27
# # reward: 0.704
# num_envs = 512//4
# env = RewardMonitor(Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu'), step_punishment=0.01)
# agent = build_ppo_lstm_agent(env, cfg_override={'rollouts': 8, 'mini_batches': 2})
# cfg_trainer = {"timesteps": 50 * 1024 * 10 // num_envs, "headless": True}
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
# trainer.train()
# trainer.eval()


# # time: 00:26
# # reward: 0.715
# num_envs = 512
# env = RewardMonitor(Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu'), step_punishment=0.01)
# agent = build_ppo_lstm_agent(env, cfg_override={'rollouts': 16, 'mini_batches': 2})
# cfg_trainer = {"timesteps": 100 * 1024 * 10 // num_envs, "headless": True}
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
# trainer.train()
# trainer.eval()
