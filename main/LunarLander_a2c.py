import torch
from torch import nn
from shared.gym_env import Environment, GameList
from A2C.a2c import a2c, A2CHyperParameters


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment(GameList.LunarLander, device)
    # Hyper parameters
    hyper_parameters = {
        A2CHyperParameters.TRAIN_EPISODES: 2000,
        A2CHyperParameters.ACTOR_LEARNING_RATE: 0.00001,
        A2CHyperParameters.CRITIC_LEARNING_RATE: 0.0002,
        A2CHyperParameters.GAMMA: 0.99,
        A2CHyperParameters.ACTOR_LAYERS: [nn.Linear(env.get_n_obs(), 256), nn.ReLU(),
                                          nn.Linear(256, 256), nn.ReLU(),
                                          nn.Linear(256, 128), nn.ReLU(),
                                          nn.Linear(256, env.get_n_actions())],
        A2CHyperParameters.CRITIC_LAYERS: [nn.Linear(env.get_n_obs(), 256), nn.ReLU(),
                                          nn.Linear(256, 256), nn.ReLU(),
                                          nn.Linear(256, 128), nn.ReLU(),
                                          nn.Linear(256, 1)],
    }

    a2c(env, hyper_parameters)

    return


if __name__ == "__main__":
    main()
