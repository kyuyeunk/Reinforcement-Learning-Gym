from PPO.ppo import ppo, HyperParameters


def main():
    game = 'CartPole-v1'
    # Hyper parameters
    hyper_parameters = {
        HyperParameters.TRAIN_EPISODES: 2000,
        HyperParameters.LEARNING_RATE: 0.004,
        HyperParameters.LAYERS: [256],
        HyperParameters.GAMMA: 0.99,
        HyperParameters.LAMBDA: 0.95,
        HyperParameters.EPS: 0.2,
        HyperParameters.BATCH_SIZE: 32,
        HyperParameters.K: 3
    }

    ppo(game, hyper_parameters)

    return


if __name__ == "__main__":
    main()
