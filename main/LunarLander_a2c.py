from A2C.a2c import a2c, HyperParameters


def main():
    game = 'LunarLander-v2'
    # Hyper parameters
    hyper_parameters = {
        HyperParameters.TRAIN_EPISODES: 2000,
        HyperParameters.ACTOR_LEARNING_RATE: 0.00001,
        HyperParameters.CRITIC_LEARNING_RATE: 0.0002,
        HyperParameters.GAMMA: 0.99,
        HyperParameters.ACTOR_LAYERS: [256, 256, 128],
        HyperParameters.CRITIC_LAYERS: [256, 256, 128]
    }

    a2c(game, hyper_parameters)

    return


if __name__ == "__main__":
    main()
