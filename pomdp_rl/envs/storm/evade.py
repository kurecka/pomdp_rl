import gymnasium as gym
import random
import numpy as np


class Evade(gym.Env):
    ACT_ADV = 0
    ACT_EAST = 1
    ACT_NORTH = 2
    ACT_PLACE = 3
    ACT_SCAN = 4
    ACT_SOUTH = 5
    ACT_WEST = 6

    def __init__(self, N=5, RADIUS=2, slippery=0.0, max_steps=100, seed=None):
        self.N = N
        self.RADIUS = RADIUS
        self.slippery = slippery
        self.max_steps = max_steps

        self.agent_pos = np.array([0, 0])
        self.trap_pos = np.array([N-2, N-1])
        self.target_pos = np.array([N-1, N-1])
        self.justscanned = False
        self.start = False
        self.turn = False
        self.num_steps = 0
        self.seed = seed

        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.reset()

    def amdone(self):
        return np.all(self.agent_pos == self.target_pos)

    def is_done(self):
        return self.amdone() or self.hascrash() or self.num_steps >= self.max_steps
    
    def hascrash(self):
        return np.all(self.agent_pos == self.trap_pos)

    def get_observations(self):
        obs = np.array([
            self.amdone(),
            self.agent_pos[0],
            self.agent_pos[1],
            self.hascrash(),
            -1,
            -1,
            self.start,
            self.turn
        ])
    

        # print(np.max(np.abs(self.agent_pos - self.trap_pos)) <= self.RADIUS, self.agent_pos, self.trap_pos)
        if self.justscanned or np.max(np.abs(self.agent_pos - self.trap_pos)) <= self.RADIUS:
            obs[4:6] = self.trap_pos

        return obs
            
    def get_reward(self):
        if self.hascrash():
            return -1
        elif self.amdone():
            return 1
        else:
            return 0
    
    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.trap_pos = np.array([self.N-2, self.N-1])
        self.justscanned = False
        self.start = False
        self.turn = False
        self.num_steps = 0
        return self.get_observations(), {}

    def move_trap(self):
        movements = np.array([
            [1, 0],[0, 1],[-1, 0],[0, -1],
            [1, 1],[-1, 1],[1, -1],[-1, -1],
            [2, 0],[0, 2],[-2, 0],[0, -2],
        ])
        probs = np.array([1/8]*4 + [1/16]*8)
        move = movements[np.random.choice(len(probs), p=probs)]
        self.trap_pos = np.clip(self.trap_pos + move, [0, 1], [self.N-1, self.N-1])

    def playable_actions(self):
        acts = np.zeros(7)
        if not self.start:
            acts[self.ACT_PLACE] = 1
        else:
            if not self.turn:
                acts[self.ACT_ADV] = 1
            else:
                acts[self.ACT_EAST] = 1
                acts[self.ACT_WEST] = 1
                acts[self.ACT_NORTH] = 1
                acts[self.ACT_SOUTH] = 1
                acts[self.ACT_SCAN] = 1
        return acts

    def action_label(self, action):
        if action == self.ACT_ADV:
            return 'ADV'
        elif action == self.ACT_PLACE:
            return 'PLACE'
        elif action == self.ACT_EAST:
            return 'EAST'
        elif action == self.ACT_WEST:
            return 'WEST'
        elif action == self.ACT_NORTH:
            return 'NORTH'
        elif action == self.ACT_SOUTH:
            return 'SOUTH'
        elif action == self.ACT_SCAN:
            return 'SCAN'


    def step(self, action):
        self.num_steps += 1
        playable = self.playable_actions()
        if not playable[action]:
            action = np.nonzero(playable)[0][0]
        
        # print(self.seed, self.num_steps, self.action_label(action))
        # print(self.seed, self.num_steps, playable)

        if action == self.ACT_ADV:
            self.justscanned = False
            self.move_trap()
        elif action == self.ACT_PLACE:
            self.start = True
        elif action == self.ACT_EAST:
            self.agent_pos[0] += 1
        elif action == self.ACT_WEST:
            self.agent_pos[0] -= 1
        elif action == self.ACT_NORTH:
            self.agent_pos[1] -= 1
        elif action == self.ACT_SOUTH:
            self.agent_pos[1] += 1
        elif action == self.ACT_SCAN:
            self.justscanned = True
        
        self.agent_pos = np.clip(self.agent_pos, 0, self.N-1)

        if action not in [self.ACT_PLACE, self.ACT_SCAN]:
            self.turn = not self.turn

        obs = self.get_observations()

        # if not self.is_done() and not self.turn:
        #     self.num_steps -= 1
        #     self.step(self.ACT_ADV)
    
        return obs, self.get_reward(), self.is_done(), False, {}

    