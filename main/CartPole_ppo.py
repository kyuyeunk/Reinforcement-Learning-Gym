import torch
import numpy
from shared.gym_env import Environment
from PPO.ppo import ppo, HyperParameters


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment('CartPole-v1', device)
    n_inputs = numpy.prod(env.get_shape_observations())
    n_outputs = env.get_n_actions()

    # Hyper parameters
    hyper_parameters = {
        HyperParameters.TRAIN_EPISODES: 2000,
        HyperParameters.LEARNING_RATE: 0.004,
        HyperParameters.LAYERS: [n_inputs, 256, n_outputs],
        HyperParameters.GAMMA: 0.99,
        HyperParameters.LAMBDA: 0.95,
        HyperParameters.EPS: 0.1,
        HyperParameters.BATCH_SIZE: 20,
        HyperParameters.K: 3
    }

    ppo(env, hyper_parameters)

    return


if __name__ == "__main__":
    main()
