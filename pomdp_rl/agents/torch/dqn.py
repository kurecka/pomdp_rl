from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import Shape, deterministic_model


def build_dqn_agent(env, memory_size=3000):
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device, replacement=False)

    models = {}
    models["q_network"] = deterministic_model(observation_space=env.observation_space,
                                            action_space=env.action_space,
                                            device=env.device,
                                            clip_actions=False,
                                            input_shape=Shape.OBSERVATIONS,
                                            hiddens=[64, 64],
                                            hidden_activation=["relu", "relu"],
                                            output_shape=Shape.ACTIONS,
                                            output_activation=None,
                                            output_scale=1.0)
    models["target_q_network"] = deterministic_model(observation_space=env.observation_space,
                                                    action_space=env.action_space,
                                                    device=env.device,
                                                    clip_actions=False,
                                                    input_shape=Shape.OBSERVATIONS,
                                                    hiddens=[64, 64],
                                                    hidden_activation=["relu", "relu"],
                                                    output_shape=Shape.ACTIONS,
                                                    output_activation=None,
                                                    output_scale=1.0)


    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["learning_starts"] = 100
    cfg["exploration"]["final_epsilon"] = 0.05
    cfg["exploration"]["timesteps"] = 5000

    return DQN(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)
