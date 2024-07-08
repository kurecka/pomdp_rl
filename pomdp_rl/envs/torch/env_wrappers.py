import gymnasium as gym
import torch


class RunningMean:
    def __init__(self, num_envs, device='cpu'):
        self.N = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.mean = torch.zeros(num_envs, device=device)
    
    def update(self, values, dones):
        self.N += dones
        self.mean[dones] += (values[dones] - self.mean[dones]) / self.N[dones]
    
    def get_mean(self):
        weights = self.N / self.N.sum()
        return (self.mean * weights).sum()


class RewardMonitor(gym.Wrapper):
    def __init__(self, env, log_interval=100, step_punishment=0, max_steps=100):
        super(RewardMonitor, self).__init__(env)
        self.cumulative_rewards = torch.zeros(env.num_envs, device=env.device)
        self.episodic_rewards = RunningMean(env.num_envs, device=env.device)
        self.terminal_rewards = RunningMean(env.num_envs, device=env.device)
        self.num_steps = 0
        self.env_num_steps = torch.zeros(env.num_envs, device=env.device)
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.step_punishment = step_punishment

    @property
    def state_space(self) -> gym.Space:
        return self.env.single_observation_space

    @property
    def observation_space(self) -> gym.Space:
        return self.env.single_observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.single_action_space

    @property
    def num_agents(self) -> int:
        return self.env.num_agents if hasattr(self.env, "num_agents") else 1

    def reset(self):
        self.cumulative_rewards[:] = 0
        self.episodic_rewards = RunningMean(self.env.num_envs, device=self.env.device)
        self.terminal_rewards = RunningMean(self.env.num_envs, device=self.env.device)

        return self.env.reset()

    def step(self, action):
        obs, rewards, dones, trunc, infos = self.env.step(action)
        rewards = rewards - self.step_punishment * ~dones
        self.cumulative_rewards += rewards.view(-1)

        self.episodic_rewards.update(self.cumulative_rewards, dones.view(-1))
        self.terminal_rewards.update(rewards.view(-1), dones.view(-1))

        self.cumulative_rewards[dones.view(-1)] = 0

        self.num_steps += 1

        if self.num_steps % self.log_interval == 0:
            print(f"Mean episodic reward: {self.episodic_rewards.get_mean()}")
            print(f"Mean terminal reward: {self.terminal_rewards.get_mean()}")
        
        self.env_num_steps += 1
        if self.max_steps is not None:
            trunc = (self.env_num_steps >= self.max_steps).reshape(dones.shape)
            self.env_num_steps[trunc.flatten()] = 0
            dones = dones | trunc
            self.env.reset(trunc.view(-1))
        self.env_num_steps[dones.flatten()] = 0

        return obs, rewards, dones, trunc, infos

    def get_episode_rewards(self):
        return self.episode_rewards