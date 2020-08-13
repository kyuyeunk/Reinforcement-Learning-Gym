import torch
import numpy
from shared.gym_env import Environment
from DQN.dqn import dqn, HyperParameters


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment('CartPole-v1', device)
    n_inputs = numpy.prod(env.get_shape_observations())
    n_outputs = env.get_n_actions()

    # Hyper parameters
    hyper_parameters = {
        HyperParameters.TRAIN_SECONDS: 60 * 60,
        HyperParameters.BATCH_SIZE: 2048,
        HyperParameters.BUFFER_SIZE: 500000,
        HyperParameters.LEARNING_RATE: 0.00005,
        HyperParameters.TARGET_UPDATE_FREQUENCY: 2048,
        HyperParameters.GAMMA: 0.99,
        HyperParameters.P_DECAY: 0.001,
        HyperParameters.P_MIN: 0.05,
        HyperParameters.LAYERS: [n_inputs, 256, n_outputs]
    }

    dqn(env, hyper_parameters)

    return


if __name__ == "__main__":
    main()
