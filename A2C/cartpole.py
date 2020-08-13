import torch
import numpy
from shared.gym_env import Environment
from A2C.a2c import a2c


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment('CartPole-v1', device)
    n_inputs = numpy.prod(env.get_shape_observations())
    n_outputs = env.get_n_actions()

    # Hyper parameters
    hyper_parameters = {
        'train_seconds': 60 * 60,
        'actor_learning_rate': 0.00005,
        'critic_learning_rate': 0.0003,
        'gamma': 0.99,
        'actor_layers': [n_inputs, 256, n_outputs],
        'critic_layers': [n_inputs, 256, 1]
    }

    a2c(env, hyper_parameters)


if __name__ == "__main__":
    main()
