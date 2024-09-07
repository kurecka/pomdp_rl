import json

import paynt.parser.sketch
import paynt.synthesizer.synthesizer_pomdp
import gymnasium as gym
import numpy as np
from scipy.sparse import lil_matrix

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('paynt.rl')


def csr_row_sample(log_probs):
    """
        Sample an element (a column) from each row of a CSR matrix.
        Each row is treated as a probability distribution over the columns.

        We use gumbel-max trick to sample from the distributions simultaneously.
    """
    assert log_probs.getformat() == "csr"

    # Sample from Gumbel(0, 1)
    gumbel_data = np.random.gumbel(loc=log_probs.data, scale=1.0)
    gumbel_data -= gumbel_data.min()
    gumbel_data += 1e-6

    # Add Gumbel noise to the log-probabilities
    gumbel = log_probs.copy()
    gumbel.data = gumbel_data

    # Find the maximum value in each row
    max_values = gumbel.argmax(axis=1)

    return np.asarray(max_values).flatten()


class StormVecModel:
    NO_LABEL = "__no_label__"

    def __init__(self, sketch_path, properties_path=None):
        logger.info(f"Creating StormVecEnv with sketch: {sketch_path}")
        if properties_path is None:
            properties_path = sketch_path.replace(".templ", ".props")

        # Extract POMDP from sketch
        quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
        pomdp = quotient.pomdp

        num_states = pomdp.nr_states
        logger.info(f"Number of states: {num_states}")

        # Get allowed actions
        action_index2label, action_label2index = self._compute_action_mappings(pomdp)

        num_actions = len(action_index2label)
        logger.info(f"Number of actions: {num_actions}")
        logger.info(f"Actions: {action_index2label}")

        self.allowed_actions = np.zeros((num_states, num_actions), dtype=np.bool)
        state_action_list = []
        for state in range(num_states):
            n_act = pomdp.get_nr_available_actions(state)
            for a_i in range(n_act):
                for label in pomdp.choice_labeling.get_labels_of_choice(pomdp.get_choice_index(state, a_i)):
                    if label == self.NO_LABEL:
                        state_action_list.append((None, None))
                    else:
                        self.allowed_actions[state, action_label2index[label]] = 1
                        state_action_list.append((state, action_label2index[label]))
        
        # Transition table
        h = num_states * num_actions
        w = num_states
        self.transitions = lil_matrix((h, w), dtype=np.float32)
        
        transition_matrix = pomdp.transition_matrix
        for (state, action), row_idx in zip(state_action_list, range(transition_matrix.nr_rows)):
            if state is not None:
                row = transition_matrix.get_row(row_idx)
                for entry in row:
                    self.transitions[state*num_actions + action, entry.column] = entry.value()
        self.transitions = self.transitions.tocsr()
        self.initial_state = pomdp.initial_states[0]

        # Assign observation vectors to states
        valuations = pomdp.observation_valuations
        observations = pomdp.observations
        num_observations = len(json.loads(str(valuations.get_json(0))))
        self.observation_table = np.zeros((num_states, num_observations), dtype=np.float32)
        self.observation_labels = list(json.loads(str(valuations.get_json(0))).keys())
        for state in range(num_states):
            observation_id = observations[state]
            valuation_json = json.loads(str(valuations.get_json(observation_id)))
            self.observation_table[state] = np.array(list(valuation_json.values()), dtype=np.float32)
        
        
        # Assign labels to states
        labeling = pomdp.labeling
        self.labels = {}
        for label in labeling.get_labels():
            self.labels[label] = np.zeros(num_states, dtype=np.float32)
            for state in labeling.get_states(label):
                self.labels[label][state] = 1
        
        # Rewards
        self.rewards = {}
        for key, reward_model in pomdp.reward_models.items():
            if reward_model.has_state_rewards:
                raise NotImplementedError("State rewards are not supported")
            if reward_model.has_transition_rewards:
                raise NotImplementedError("Transition rewards are not supported")
            if reward_model.has_state_action_rewards:
                self.rewards[key] = np.zeros((num_states, num_actions), dtype=np.float32)
                for (state, action), reward in zip(state_action_list, reward_model.state_action_rewards):
                    if state is not None:
                        self.rewards[key][state, action] = reward

        # Gym env attributes
        self.num_observations = num_observations
        self.num_actions = num_actions

    
    @classmethod
    def _compute_action_mappings(cls, pomdp):
        """Computes the mapping between the action labels and their indices in the one-hot encoding used by the RL agent."""
        action_labels = set()
        for s_i in range(pomdp.nr_states):
            n_act = pomdp.get_nr_available_actions(s_i)
            for a_i in range(n_act):
                for label in pomdp.choice_labeling.get_labels_of_choice(pomdp.get_choice_index(s_i, a_i)):
                    action_labels.add(label)
        
        if "__no_label__" in action_labels:
            action_labels.remove(cls.NO_LABEL)
        index2label = list(sorted(action_labels))
        label2index = {
            label: i for i, label in enumerate(index2label)
        }
        return index2label, label2index


class StormEnv(gym.Env):
    def __init__(self, sketch_path, reward_function, properties_path=None, allow_wrong_actions=False):
        """
            Reward function accepts a dictionary of reward signals and returns a scalar reward.
            Values `goal` and `fail` are always present in the dictionary.
        """
        super().__init__()
        self.model = StormVecModel(sketch_path, properties_path)
        self.reward_function = reward_function
        self.allow_wrong_actions = allow_wrong_actions

        self.state = self.model.initial_state
        self.done = False

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.num_observations,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.model.num_actions)
    
    def state_info(self, state):
        return {
            # "labels": [label for label, value in self.model.labels.items() if value[state] == 1],
            "allowed_actions": self.model.allowed_actions[state],
        }

    def is_done(self):
        return self.done

    def reset(self):
        self.state = self.model.initial_state
        self.done = False
        return self.model.observation_table[self.state], self.state_info(self.state)
    
    def step(self, action):
        if self.done:
            raise ValueError("Cannot step in a terminal state")

        if not self.model.allowed_actions[self.state, action]:
            if not self.allow_wrong_actions:
                raise ValueError(f"Action {action} is not allowed in state {self.state}")
            else:
                action = self.model.allowed_actions[self.state].nonzero()[0]
        
        # Sample next state
        state_action = self.state * self.model.num_actions + action
        row = self.model.transitions[state_action]
        candidate_states = row.indices
        probabilities = row.data
        next_state = np.random.choice(candidate_states, p=probabilities)

        observation = self.model.observation_table[next_state]

        # Update done
        goal = float(self.model.labels["goal"][next_state])
        fail = float(1 - self.model.labels["notbad"][next_state])
        self.done = goal>0.99 or fail>0.99

        # Compute reward
        rewards = {key: reward[self.state, action] for key, reward in self.model.rewards.items()}
        rewards["goal"] = goal
        rewards["fail"] = fail

        reward = self.reward_function(rewards)

        self.state = next_state

        return observation, reward, self.done, False, self.state_info(next_state)

class StormVecEnv(gym.vector.VectorEnv):
    def __init__(self, sketch_path, reward_function, num_envs=10, properties_path=None, allow_wrong_actions=False):
        model = StormVecModel(sketch_path, properties_path)
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(model.num_observations,), dtype=np.float32)
        action_space = gym.spaces.Discrete(model.num_actions)

        super(StormVecEnv, self).__init__(num_envs, observation_space, action_space)

        self.model = model
        self.reward_function = reward_function
        self.allow_wrong_actions = allow_wrong_actions
        self.transition_log_probs = self.model.transitions.copy()
        self.transition_log_probs.data = np.log(self.transition_log_probs.data)

        self.states = np.array([model.initial_state] * num_envs)

    def state_info(self, states):
        return {
            "allowed_actions": self.model.allowed_actions[states],
            "labels": {label: self.model.labels[label][states] for label in self.model.labels},
        }

    def reset(self, seed=None, idx=...):
        self.states[idx] = self.model.initial_state
        return self.model.observation_table[self.states], self.state_info(self.states)
    
    def step(self, actions):
        wrong_actions = ~self.model.allowed_actions[self.states, actions]
        wrong_states = self.states[wrong_actions]
        if np.any(wrong_actions) and not self.allow_wrong_actions:
            idx = np.argmin(self.model.allowed_actions[self.states, actions])
            raise ValueError(f"Action {actions[idx]} is not allowed in state {self.states[idx]}")

        actions[wrong_actions] = np.argmax(self.model.allowed_actions[wrong_states], axis=1)
        
        
        # Sample next state
        state_actions = self.states * self.model.num_actions + actions
        rows = self.transition_log_probs[state_actions]
        next_states = csr_row_sample(rows)

        # Update done
        goals = self.model.labels["goal"][next_states] 
        fail = 1 - self.model.labels["notbad"][next_states]
        dones = np.maximum(goals, fail) > 0.99

        # Compute reward
        rewards_dict = {key: reward[self.states, actions] for key, reward in self.model.rewards.items()}
        rewards_dict["goal"] = goals
        rewards_dict["fail"] = fail
        rewards = self.reward_function(rewards_dict)

        # Update states and observations
        self.states[:] = next_states
        self.reset(idx=dones)
        observations = self.model.observation_table[next_states]

        return observations, rewards, dones, np.zeros_like(dones), self.state_info(next_states)


if __name__ == "__main__":
    # Test csr_row_sample
    probs = lil_matrix([[0.9, 0, 0.1], [0, 0.2, 0.8], [0, 0.6, 0.4]]).tocsr()
    log_probs = probs.copy()
    log_probs.data = np.log(log_probs.data)
    nr_samples = np.zeros(log_probs.shape, dtype=np.float64)
    n = 10000
    for i in range(n):
        samples = csr_row_sample(log_probs)
        nr_samples[np.arange(log_probs.shape[0]), samples] += 1/n
    assert (nr_samples - probs.toarray() < 0.05).all()
