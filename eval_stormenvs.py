def main(sketch_name, seed=42):
    from pomdp_rl.envs.torch.env_wrappers import RewardMonitor, SkrlAdapter
    from pomdp_rl.agents.torch.ppo import build_ppo_agent, build_ppo_mixed_agent, build_ppo_lstm_agent

    from pomdp_rl.envs.storm.storm_vec_env import StormVecEnv

    import gymnasium as gym
    import random

    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    num_envs = 32
    set_seed(seed)

    def reward_function(rewards):
        return rewards["goal"] #- rewards["fail"]
    
    env = StormVecEnv(f'/opt/learning/rl_src/{sketch_name}/sketch.templ', reward_function, num_envs=num_envs, allow_wrong_actions=True)
    env = SkrlAdapter(env)

    # def get_storm_env(seed):
    #     from pomdp_rl.envs.storm.storm_env import StormEnv, ReachAvoidWrapper
    #     def fn():
    #         return ReachAvoidWrapper(StormEnv(f'/opt/learning/rl_src/{sketch_name}', max_steps=100, seed=seed), reach_reward=1, fail_reward=-1)
    #     return fn
    # env = gym.vector.AsyncVectorEnv([get_storm_env(seed + i) for i in range(num_envs)], daemon=True)
    # env = SkrlAdapter(env)

    # from pomdp_rl.envs.torch.gridworld import Evade
    # N = 5
    # radius = 2
    # env = Evade(dims=[N,N], radius=radius, num_envs=num_envs, slide_prob=0, render_mode=None, device='cpu', extra_actions=3)

    env = RewardMonitor(env, step_punishment=0.0)

    agent = build_ppo_lstm_agent(env, cfg_override={
        'rollouts': 128,
        'mini_batches': 32,
        'ratio_clip': 0.05,
        'value_clip': 0.2,
        'discount_factor': 0.99,
        'learning_epochs': 1,
        'experiment': {'directory': f'runs/large/{sketch_name}'}}
    )
    cfg_trainer = {"timesteps": 1500 * 1024 // num_envs, "headless": True, "log_interval": 100}
    # cfg_trainer = {"timesteps": 10, "headless": True, "log_interval": 100}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
    trainer.train()


if __name__ == '__main__':
    for seed in [42]: #, 43, 44, 45]:
        for sketch_name in [
            'models/evade-n7-r2',
            # 'models/refuel-10',
            # 'models/evade-n6-r2',
            # 'models_large/geo-2-8',
            # 'models/refuel-20',
            # 'models_large/drone-2-8-1',
        ]:
            main(sketch_name, seed)
