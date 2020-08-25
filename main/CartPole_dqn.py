from DQN.dqn import dqn, HyperParameters


def main():
    game = 'CartPole-v1'
    # Hyper parameters
    hyper_parameters = {
        HyperParameters.TRAIN_EPISODES: 2000,
        HyperParameters.BATCH_SIZE: 2048,
        HyperParameters.BUFFER_SIZE: 500000,
        HyperParameters.LEARNING_RATE: 0.00005,
        HyperParameters.TARGET_UPDATE_FREQUENCY: 2048,
        HyperParameters.GAMMA: 0.99,
        HyperParameters.P_DECAY: 0.001,
        HyperParameters.P_MIN: 0.05,
        HyperParameters.LAYERS: [256]
    }

    dqn(game, hyper_parameters)

    return


if __name__ == "__main__":
    main()
