import torch
import numpy
from shared.gym_env import Environment
from A2C.a2c import a2c, HyperParameters


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment('CartPole-v1', device)
    n_inputs = numpy.prod(env.get_shape_observations())
    n_outputs = env.get_n_actions()

    # Hyper parameters
    hyper_parameters = {
        HyperParameters.TRAIN_EPISODES: 2000,
        HyperParameters.ACTOR_LEARNING_RATE: 0.00005,
        HyperParameters.CRITIC_LEARNING_RATE: 0.0003,
        HyperParameters.GAMMA: 0.99,
        HyperParameters.ACTOR_LAYERS: [n_inputs, 256, n_outputs],
        HyperParameters.CRITIC_LAYERS: [n_inputs, 256, 1]
    }

    a2c(env, hyper_parameters)


if __name__ == "__main__":
    main()
