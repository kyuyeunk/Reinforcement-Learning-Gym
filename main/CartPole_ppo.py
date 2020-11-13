import torch
from torch import nn
from shared.gym_env import Environment, GameList
from PPO.ppo import ppo, PPOHyperParameters


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment(GameList.CartPole, device)
    # Hyper parameters
    hyper_parameters = {
        PPOHyperParameters.TRAIN_EPISODES: 1000,
        PPOHyperParameters.LEARNING_RATE: 0.004,
        PPOHyperParameters.LAYERS: [nn.Linear(env.get_n_obs(), 256), nn.ReLU(),
                                    nn.Linear(256, env.get_n_actions())],
        PPOHyperParameters.GAMMA: 0.99,
        PPOHyperParameters.LAMBDA: 0.95,
        PPOHyperParameters.EPS: 0.2,
        PPOHyperParameters.BATCH_SIZE: 32,
        PPOHyperParameters.K: 3
    }

    ppo(env, hyper_parameters)

    return


if __name__ == "__main__":
    main()
