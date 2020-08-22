from PPO.ppo import ppo, HyperParameters


def main():
    game = 'LunarLander-v2'
    # Hyper parameters
    hyper_parameters = {
        HyperParameters.TRAIN_EPISODES: 10000,
        HyperParameters.LEARNING_RATE: 0.001,
        HyperParameters.LAYERS: [128, 128],
        HyperParameters.GAMMA: 0.99,
        HyperParameters.LAMBDA: 0.95,
        HyperParameters.EPS: 0.1,
        HyperParameters.BATCH_SIZE: 512,
        HyperParameters.K: 3
    }

    ppo(game, hyper_parameters)

    return


if __name__ == "__main__":
    main()
