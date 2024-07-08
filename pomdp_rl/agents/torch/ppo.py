import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model, CategoricalMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL


class PPOMixedModel(CategoricalMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.net_value = nn.Linear(64, 1)
        self.net_policy = nn.Linear(64, self.num_actions)

    def compute(self, inputs, role):
        if role == "policy":
            x = self.net(inputs["states"])
            return self.net_policy(x), {}
        elif role == "value":
            x = self.net(inputs["states"])
            return self.net_value(x), {}
    
    def act(self, inputs, role):
        if role == "policy":
            return CategoricalMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)


class PPOPolicy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class PPOValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class PPOMixedModelLSTM(CategoricalMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True,
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=10, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length
        self.clip_actions = clip_actions

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),)
        self.net_policy = nn.Linear(64, self.num_actions)
        self.net_value = nn.Linear(64, 1)
    
    def act(self, inputs, role):
        if role == "policy":
            return CategoricalMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def get_specification(self):
        # batch size (N) is the number of envs during rollout
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        if role == "policy":
            x = self.net(rnn_output)
            return self.net_policy(x), {"rnn": [rnn_states[0], rnn_states[1]]}
        elif role == "value":
            x = self.net(rnn_output)
            return self.net_value(x), {"rnn": [rnn_states[0], rnn_states[1]]}


def get_ppo_config(env, cfg_override=None):
    batch_size = 128
    mini_batches = 2
    rollouts = batch_size * mini_batches // env.num_envs
    rollouts = max(rollouts, 8)
    
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["learning_epochs"] = 1
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["grad_norm_clip"] = 0.5
    cfg["ratio_clip"] = 0.05
    cfg["value_clip"] = 0.05
    cfg["clip_predicted_values"] = False
    cfg["entropy_loss_scale"] = 0
    cfg["value_loss_scale"] = 2
    cfg["kl_threshold"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
    cfg["rollouts"] = rollouts
    cfg["mini_batches"] = rollouts * env.num_envs // batch_size

    if cfg_override is not None:
        cfg.update(cfg_override)
    
    print("PPO config:")
    for key, value in cfg.items():
        print(f"{key}: {value}")

    return cfg


def build_ppo_mixed_agent(env, cfg_override=None):
    cfg = get_ppo_config(env, cfg_override=cfg_override)

    memory = RandomMemory(memory_size=cfg['rollouts'], num_envs=env.num_envs, device=env.device)

    models = {}
    models["value"] = models["policy"] = PPOMixedModel(env.observation_space, env.action_space, device=env.device, unnormalized_log_prob=True)

    return PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device
    )


def build_ppo_agent(env, cfg_override=None):
    cfg = get_ppo_config(env, cfg_override=cfg_override)

    memory = RandomMemory(memory_size=cfg['rollouts'], num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = PPOPolicy(env.observation_space, env.action_space, unnormalized_log_prob=True, device=env.device)
    models["value"] = PPOValue(env.observation_space, env.action_space, device=env.device)

    return PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device
    )


def build_ppo_lstm_agent(env, cfg_override=None):
    cfg = get_ppo_config(env, cfg_override=cfg_override)

    memory = RandomMemory(memory_size=cfg['rollouts'], num_envs=env.num_envs, device=env.device)

    models = {}
    models["value"] = models["policy"] = PPOMixedModelLSTM(
        env.observation_space, env.action_space,
        device=env.device, unnormalized_log_prob=True,
        num_envs=env.num_envs, num_layers=1,
        hidden_size=128, sequence_length=32
    )

    return PPO_RNN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device
    )
