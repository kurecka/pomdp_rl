import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class GridWorld(gym.vector.VectorEnv):
    MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, STAY = 0, 1, 2, 3, 4

    def __init__(self, dims, observation_space, action_space, num_envs=1, render_mode=None, device='cpu', slide_prob=0.0, slide_scale=0):
        super(GridWorld, self).__init__(num_envs, observation_space, action_space)
        self.metadata = {"render_modes": ["human"]}
        self.render_mode = render_mode

        self.device = device
        self.num_agents = 1
        self.dims = torch.tensor(dims, device=device)
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reward = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.slide_prob = slide_prob
        self.slide_scale = slide_scale

        self.agent_pos = self._agent_initial_pos()

        self.directions = torch.tensor([
            [0, -1],  # MOVE_LEFT
            [0, 1],   # MOVE_RIGHT
            [-1, 0],  # MOVE_UP
            [1, 0],   # MOVE_DOWN
            [0, 0],   # STAY
        ], device=device)
    
    def _agent_initial_pos(self):
        return torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)

    def _get_obs(self):
        return self.agent_pos.to(torch.float32)

    def _get_info(self):
        return {}

    def reset(self, env_idx=...):
        self.agent_pos[env_idx] = self._agent_initial_pos()[env_idx]
        return self._get_obs(), self._get_info()

    def _env_map(self, env_idx=0):
        grid = np.full((self.dims[0], self.dims[1]), '.')
        grid[self.agent_pos[env_idx, 0], self.agent_pos[env_idx, 1]] = 'A'
        return grid

    def env_print(self, env_idx=0):
        for row in self._env_map(env_idx):
            print(' '.join(row))
        print(f"Done: {self.done[env_idx]}")
        print(f"Reward: {self.reward[env_idx]}")

    def _compute_new_position(self, position, delta, lbounds=None, ubounds=None):
        new_position = position + delta

        if lbounds is None:
            lbounds = torch.zeros_like(new_position)
        if ubounds is None:
            ubounds = self.dims - 1

        return torch.clamp(new_position, lbounds, ubounds)

    def _agent_move(self, action):
        slide = torch.rand(self.num_envs, device=self.device) < self.slide_prob
        delta = self.directions[torch.clip(action, 0, 4)] * (1 + slide * (self.slide_scale-1))[:, None]
        return self._compute_new_position(self.agent_pos, delta)

    def step(self, action):
        shape = action.shape
        self._move(action.view(-1))
        self._eval_move()
        self.reset(env_idx=self.done)
        if self.render_mode == "human":
            print("="*10)
            print("\n")
            self.env_print()
        return self._get_obs(), self.reward.reshape(shape), self.done.reshape(shape), torch.zeros(shape, dtype=bool, device=self.device), self._get_info()

    def _move(self, action):
        self.agent_pos[:] = self._agent_move(action.view(-1))
    
    def _eval_move(self):
        pass


class GridWorldReachAvoid(GridWorld):
    def __init__(self, dims, observation_space, action_space, num_envs=1, render_mode=None, device='cpu', slide_prob=0.0, slide_scale=0):
        super(GridWorldReachAvoid, self).__init__(dims, observation_space, action_space, num_envs, render_mode, device, slide_prob, slide_scale)

        self.target_pos = self._target_initial_pos()
        self.target_mask = torch.zeros((num_envs, *dims), dtype=torch.bool, device=device)
        self.trap_pos = self._trap_initial_pos()
        self.trap_mask = torch.zeros((num_envs, *dims), dtype=torch.bool, device=device)
    
    def _agent_initial_pos(self):
        return torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)

    def _target_initial_pos(self):
        return torch.tensor([[self.dims[0] - 1, self.dims[1] - 1]], device=self.device).repeat(self.num_envs, 1, 1)
    
    def _trap_initial_pos(self):
        return torch.tensor([[self.dims[0] // 2, self.dims[1] // 2]], device=self.device).repeat(self.num_envs, 1, 1)

    def reset(self, env_idx=...):
        super().reset(env_idx)
        self.trap_pos[env_idx] = self._trap_initial_pos()[env_idx]
        self.target_pos[env_idx] = self._target_initial_pos()[env_idx]
        self.trap_mask[env_idx] = False
        self.target_mask[env_idx] = False
        self._set_masks(self.target_mask, self.target_pos, True)
        self._set_masks(self.trap_mask, self.trap_pos, True)
        return self._get_obs(), self._get_info()

    def _env_map(self, env_idx=0):
        grid = np.full((self.dims[0], self.dims[1]), '.')
        grid[self.agent_pos[env_idx, 0], self.agent_pos[env_idx, 1]] = 'A'
        grid[self.target_pos[env_idx, :, 0], self.target_pos[env_idx, :, 1]] = 'T'
        grid[self.trap_pos[env_idx, :, 0], self.trap_pos[env_idx, :, 1]] = 'X'
        
        return grid

    def _trap_move(self):
        return self.trap_pos

    def _target_move(self):
        return self.target_pos

    def _query_masks(self, mask, idx):
        return mask[torch.arange(self.num_envs).unsqueeze(1), idx[..., 0], idx[..., 1]]
    
    def _set_masks(self, mask, idx, value):
        mask[torch.arange(self.num_envs).unsqueeze(1), idx[..., 0], idx[..., 1]] = value

    def _move(self, action):
        super()._move(action)
        self._set_masks(self.target_mask, self.target_pos, False)
        self._set_masks(self.trap_mask, self.trap_pos, False)
        self.trap_pos[:] = self._trap_move()
        self.target_pos[:] = self._target_move()
        self._set_masks(self.target_mask, self.target_pos, True)
        self._set_masks(self.trap_mask, self.trap_pos, True)

    def _eval_move(self):
        crash = self._query_masks(self.trap_mask, self.agent_pos.unsqueeze(1)).flatten()
        reach = self._query_masks(self.target_mask, self.agent_pos.unsqueeze(1)).flatten()

        self.done[:] = torch.logical_or(crash, reach)
        self.reward[:] = reach.to(torch.float32)
        self.reward[crash] = -1


class Evade(GridWorldReachAvoid):
    def __init__(self, dims, radius=2, num_envs=1, render_mode=None, device='cpu', slide_prob=0, slide_scale=0):
        observation_space = spaces.Box(np.array([0, 0, -1, -1]), np.concatenate([dims, dims])-1, shape=(4,), dtype=int)
        action_space = spaces.Discrete(5)
        super(Evade, self).__init__(dims, observation_space, action_space, num_envs, render_mode, device, slide_prob, slide_scale)

        self.radius = radius
        self.scanning = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _trap_initial_pos(self):
        return torch.tensor([[self.dims[0] - 2, self.dims[1] - 1]], device=self.device).repeat(self.num_envs, 1, 1)


    def _trap_move(self):
        positions = self.trap_pos
        moves = torch.tensor([
            [0, 1], [0, -1], [1, 0], [-1, 0],  # distance 1
            [0, 2], [0, -2], [2, 0], [-2, 0],  # distance 2 (straight)
            [1, 1], [1, -1], [-1, 1], [-1, -1] # distance 2 (diagonal)
        ], device=self.device)
        move_probs = torch.tensor([1/8] * 4 + [1/16] * 8, device=self.device)
        move_idx = torch.multinomial(move_probs, np.prod(positions.shape[:-1]), replacement=True).reshape(positions.shape[:-1])
        delta = moves[move_idx]
        return self._compute_new_position(positions, delta, lbounds=torch.tensor([0, 1], device=self.device))

    def _move(self, action):
        super()._move(action)
        self.scanning[:] = action == 4
    
    def _get_obs(self):
        trap_pos = self.trap_pos.reshape(-1, 2)

        visible = torch.logical_or(self.scanning, (torch.abs(trap_pos - self.agent_pos) <= self.radius).all(dim=-1))
        trap_obs = torch.where(visible[:, None], trap_pos, -torch.ones_like(trap_pos))
        return torch.cat([self.agent_pos, trap_obs], dim=-1).to(torch.float32)


class Refuel(GridWorldReachAvoid):
    def __init__(self, dim, num_envs=1, render_mode=None, device='cpu', slide_prob=0.3, slide_scale=2, max_fuel=None, num_refuel=3):
        dims = (dim, dim)
        observation_space = spaces.Box(np.array([0, 0, 0]), np.concatenate([dims, [2]])-1, shape=(3,), dtype=int)
        action_space = spaces.Discrete(5)
        super(Refuel, self).__init__(dims, observation_space, action_space, num_envs, render_mode, device, slide_prob, slide_scale)

        if max_fuel is None:
            max_fuel = dim - 2

        self.max_fuel = max_fuel
        self.fuel = torch.full((num_envs,), max_fuel, device=device)
        self.refuel_mask = torch.zeros(dims, dtype=torch.bool, device=device)
        for i in range(0, dim, np.ceil(dim / num_refuel).astype(int)):
            self.refuel_mask[i, i] = True

    def reset(self, env_idx=...):
        super().reset(env_idx)
        self.fuel[env_idx] = self.max_fuel
        return self._get_obs(), self._get_info()
    
    def _trap_initial_pos(self):
        return torch.tensor([[self.dims[0] - 2, self.dims[1] - 2]], device=self.device).repeat(self.num_envs, 1, 1)

    def _env_map(self, env_idx=0):
        grid = super()._env_map(env_idx)
        for i in range(self.dims[0]):
            if self.refuel_mask[i, i]:
                grid[i, i] = 'R'
        return grid

    def _can_refuel(self):
        return self.refuel_mask[self.agent_pos[..., 0], self.agent_pos[..., 1]]

    def _move(self, action):
        super()._move(action)

        can_refuel = self._can_refuel()
        self.fuel -= (action < 4).to(torch.int32)
        self.fuel[:] = torch.maximum(self.fuel, can_refuel * self.max_fuel * (action == 4).to(torch.int32))

    def _eval_move(self):
        no_fuel = (self.fuel <= 0) & ~self.done & ~self._can_refuel()
        self.done[:] = torch.logical_or(self.done, no_fuel)
        self.reward[self.done] = 1
        self.reward[no_fuel] = -1
    
    def _get_obs(self):
        return torch.cat([self.agent_pos, self._can_refuel()[:, None].to(torch.int)], dim=-1)
