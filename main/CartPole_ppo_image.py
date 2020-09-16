import torch
from torch import nn
from shared.gym_env import Environment, GameList
from PPO.ppo import ppo, PPOHyperParameters


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment(GameList.CartPole, device, force_image_input=True, n_frames=4)
    # Hyper parameters
    hyper_parameters = {
        PPOHyperParameters.TRAIN_EPISODES: 1000000,
        PPOHyperParameters.LEARNING_RATE: 0.000006,
        PPOHyperParameters.LAYERS: [nn.Conv2d(env.n_frames, 32, kernel_size=8, stride=4), nn.ReLU(),
                                    nn.Conv2d(32, 32, kernel_size=3, stride=2), nn.ReLU(),
                                    nn.Conv2d(32, 32, kernel_size=3, stride=2), nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, env.get_n_actions())],
        PPOHyperParameters.GAMMA: 0.99,
        PPOHyperParameters.LAMBDA: 0.95,
        PPOHyperParameters.EPS: 0.2,
        PPOHyperParameters.BATCH_SIZE: 128,
        PPOHyperParameters.K: 4
    }

    ppo(env, hyper_parameters)

    return


if __name__ == "__main__":
    main()
