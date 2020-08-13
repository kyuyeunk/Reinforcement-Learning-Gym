import torch
import numpy
from shared.gym_env import Environment
from DQN.dqn import dqn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment('CartPole-v1', device)
    n_inputs = numpy.prod(env.get_shape_observations())
    n_outputs = env.get_n_actions()

    # Hyper parameters
    hyper_parameters = {
        'train_seconds': 60 * 60,
        'batch_size': 2048,
        'buffer_size': 500000,
        'learning_rate': 0.00005,
        'target_update_frequency': 2048,
        'gamma': 0.99,
        'p_decay': 0.001,
        'p_min': 0.05,
        'layers': [n_inputs, 256, n_outputs]
    }

    dqn(env, hyper_parameters)

    return


if __name__ == "__main__":
    main()
