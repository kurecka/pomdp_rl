import paynt.parser.sketch
import os
from stormpy import simulator
import json
import numpy as np
import logging
import gymnasium as gym
import random

logging.basicConfig(level=logging.INFO)


def load_sketch(env_path):
    env_path = os.path.abspath(env_path)
    sketch_path = os.path.join(env_path, "sketch.templ")
    properties_path = os.path.join(env_path, "sketch.props")    
    quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    return quotient


def observation_index2features(observation, stormpy_model):
    valuations_json = stormpy_model.observation_valuations.get_json(
        observation)
    parsed_valuations = json.loads(str(valuations_json))
    vector = []
    for key in parsed_valuations:
        if type(parsed_valuations[key]) == bool:
            if parsed_valuations[key]:
                vector.append(1.0)
            else:
                vector.append(0.0)
        else:
            vector.append(float(parsed_valuations[key]))
    return np.array(vector, dtype=np.float32)


class StormEnv(gym.Env):
    def __init__(self, env_path, max_steps=None, seed=None):
        self.quotient = load_sketch(env_path)
        self.simulator = simulator.create_simulator(self.quotient.pomdp, seed=seed)
        self.stormpy_model = self.quotient.pomdp
        self._compute_action_mappings()
        self.flags = []
        self.max_steps = max_steps
        self.seed = seed

        self.num_steps = 0

        self._detect_space()
    
    def close(self):
        pass
    
    def _detect_space(self):
        """Detects the observation and action space of the environment."""
        self.action_space = gym.spaces.Discrete(len(self.index2label))
        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)

    def step(self, action):
        self.num_steps += 1
        storm_action = self._convert_action(action)
        obs, rew, labels = self.simulator.step(storm_action)

        self.current_obs = obs
        self.flags = labels

        obs = observation_index2features(obs, self.quotient.pomdp)
        trunc = False  
        done = self.is_done()

        info = {} if done else {
            'action_mask': self._get_action_mask()
        }
        rew = rew[-1] if len(rew) > 0 else 0
        return obs, rew, done, trunc, info

    def is_done(self):
        return self.simulator.is_done() or (self.max_steps is not None and self.num_steps >= self.max_steps)
    
    def reset(self, seed=None, options=None):
        obs, _, labels = self.simulator.restart()
        self.num_steps = 0
        self.current_obs = obs
        self.flags = labels
        obs = observation_index2features(obs, self.quotient.pomdp)
        info = {
            'action_mask': self._get_action_mask()
        }
        return obs, info
    
    def _compute_action_mappings(self):
        """Computes the mapping between the action labels and their indices in the one-hot encoding used by the RL agent."""
        action_labels = set()
        for s_i in range(self.stormpy_model.nr_states):
            n_act = self.stormpy_model.get_nr_available_actions(s_i)
            for a_i in range(n_act):
                for label in self.stormpy_model.choice_labeling.get_labels_of_choice(self.stormpy_model.get_choice_index(s_i, a_i)):
                    action_labels.add(label)
        self.nr_actions = len(action_labels)
        self.index2label = list(sorted(action_labels))
        self.label2index = {
            label: i for i, label in enumerate(self.index2label)
        }
    
    def _get_action_choice_labels(self):
        """Converts the current legal actions to the keywords used by the Storm model."""
        labels = []
        current_state = self.simulator._report_state()
        for action_index in range(self.simulator.nr_available_actions()):
            choice_index = self.stormpy_model.get_choice_index(current_state, action_index)
            labels_of_choice = self.stormpy_model.choice_labeling.get_labels_of_choice(choice_index)
            label = labels_of_choice.pop()
            labels.append(label)
        return labels
    
    def _get_action_mask(self):
        """Returns a mask of the legal actions."""
        mask = np.zeros(self.nr_actions, dtype=bool)
        for label in self._get_action_choice_labels():
            #if label is not None:
            mask[self.label2index[label]] = 1
                
        return mask
    
    def _convert_action(self, action):
        """Converts the action from the RL agent to the action used by the Storm model."""
        act_label = self.index2label[int(action)]
        choice_list = self._get_action_choice_labels()
        try:
            action = choice_list.index(act_label)
            # print(self.seed, self.num_steps, action, choice_list[action])
        except ValueError:
            # action = random.randrange(0, len(choice_list))  #0 #None
            argsorted = np.argsort(choice_list)
            action = int(argsorted[0]) #None
            # print(self.seed, self.num_steps, action, choice_list[action])
            # raise ValueError(f"Action {act_label} not found in the list of legal actions: {choice_list}")
        return action

class ReachAvoidWrapper(gym.Wrapper):
    def __init__(self, env, reach_reward=100, fail_reward=-100):
        super().__init__(env)
        self.reach_reward = reach_reward
        self.fail_reward = fail_reward
    
    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        rew = 0
        if done:
            if 'goal' in self.env.flags:
                rew = self.reach_reward
            elif 'unexplored' in self.env.flags:
                rew = self.fail_reward
        return obs, rew, done, trunc, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed, options)

class MemoryWrapper(gym.Wrapper):
    def __init__(self, env, memory_size=5):
        super().__init__(env)
        self.memory_size = memory_size
        self.memory = 0

        self.action_space = gym.spaces.Discrete(self.env.action_space.n * memory_size)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.observation_space.shape[0] + self.memory_size,), dtype=np.float32)
    
    def extend_observation(self, obs):
        memory_obs = np.zeros(self.memory_size)
        memory_obs[self.memory] = 1
        return np.concatenate([obs, memory_obs])

    def step(self, memory_action):
        action = memory_action // self.memory_size
        memory = memory_action % self.memory_size
        obs, rew, done, trunc, info = self.env.step(action)
        self.memory = memory
        return self.extend_observation(obs), rew, done, trunc, info
    
    def reset(self):
        obs, info = self.env.reset()
        self.memory = 0
        return self.extend_observation(obs), info
