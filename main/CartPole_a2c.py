from A2C.a2c import a2c, HyperParameters


def main():
    game = 'CartPole-v1'
    # Hyper parameters
    hyper_parameters = {
        HyperParameters.TRAIN_EPISODES: 2000,
        HyperParameters.ACTOR_LEARNING_RATE: 0.00005,
        HyperParameters.CRITIC_LEARNING_RATE: 0.0003,
        HyperParameters.GAMMA: 0.99,
        HyperParameters.ACTOR_LAYERS: [256],
        HyperParameters.CRITIC_LAYERS: [256]
    }

    a2c(game, hyper_parameters)

    return


if __name__ == "__main__":
    main()
