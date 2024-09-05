def main(scatch_name, seed=42):
    from pomdp_rl.envs.torch.env_wrappers import RewardMonitor, SkrlAdapter
    from pomdp_rl.agents.torch.ppo import build_ppo_agent, build_ppo_mixed_agent, build_ppo_lstm_agent

    import gymnasium as gym
    import random

    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    num_envs = 32
    set_seed(seed)
    env_seed = random.randint(0, 1000)

    def get_storm_env(seed):
        from pomdp_rl.envs.storm.storm_env import StormEnv, ReachAvoidWrapper
        def fn():
            return ReachAvoidWrapper(StormEnv(f'/opt/learning/synthesis/rl_src/{scatch_name}', max_steps=200, seed=seed), reach_reward=1, fail_reward=-1)
        return fn
    env = gym.vector.AsyncVectorEnv([get_storm_env(env_seed + i) for i in range(num_envs)], daemon=True)
    env = SkrlAdapter(env)
    env = RewardMonitor(env, step_punishment=0.0)

    agent = build_ppo_lstm_agent(env, cfg_override={
        'rollouts': 256,
        'mini_batches': 32,
        'ratio_clip': 0.2,
        'value_clip': 0.2,
        'discount_factor': 0.99,
        'learning_epochs': 1,
        'experiment': {'directory': f'runs/large/{scatch_name}'}}
    )
    cfg_trainer = {"timesteps": 1500 * 1024 // num_envs, "headless": True, "log_interval": 100}
    # cfg_trainer = {"timesteps": 10, "headless": True, "log_interval": 100}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
    trainer.train()


if __name__ == '__main__':
    for seed in [42, 43, 44, 45]:
        for scatch_name in ['models/refuel-10','models_large/geo-2-8', 'models/refuel-20', 'models_large/drone-2-8-1']:
            main(scatch_name, seed)


# from pomdp_rl.envs.storm.storm_env import StormEnv, ReachAvoidWrapper
# env = ReachAvoidWrapper(StormEnv(f'/opt/learning/synthesis/rl_src/models/evade-n5-r2', max_steps=100), reach_reward=1, fail_reward=-1)

# print(env.env.index2label)


# actions = [2, 3, 4, 3, 0, 5, 0, 2, 1, 6]
# print(env.reset())
# for a in actions:
#     print(env.step(a))
#     if env.is_done():
#         break


# for j in range(10):
#     obs, info = env.reset()
#     print(env.env.index2label)
#     for i in range(10):
#         print(env.step(1))
#         if env.is_done():
#             break

#     if not env.is_done():
#         for i in range(10):
#             print(env.step(5))
#             if env.is_done():
#                 break
