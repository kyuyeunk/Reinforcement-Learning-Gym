from A2C.a2c import a2c, A2CHyperParameters


def main():
    game = 'CartPole-v1'
    # Hyper parameters
    hyper_parameters = {
        A2CHyperParameters.TRAIN_EPISODES: 2000,
        A2CHyperParameters.ACTOR_LEARNING_RATE: 0.00005,
        A2CHyperParameters.CRITIC_LEARNING_RATE: 0.0003,
        A2CHyperParameters.GAMMA: 0.99,
        A2CHyperParameters.ACTOR_LAYERS: [256],
        A2CHyperParameters.CRITIC_LAYERS: [256]
    }

    a2c(game, hyper_parameters)

    return


if __name__ == "__main__":
    main()
