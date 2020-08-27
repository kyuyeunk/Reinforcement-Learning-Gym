from DQN.dqn import dqn, DQNHyperParameters


def main():
    game = 'CartPole-v1'
    # Hyper parameters
    hyper_parameters = {
        DQNHyperParameters.TRAIN_EPISODES: 2000,
        DQNHyperParameters.BATCH_SIZE: 2048,
        DQNHyperParameters.BUFFER_SIZE: 500000,
        DQNHyperParameters.LEARNING_RATE: 0.00005,
        DQNHyperParameters.TARGET_UPDATE_FREQUENCY: 2048,
        DQNHyperParameters.GAMMA: 0.99,
        DQNHyperParameters.P_DECAY: 0.001,
        DQNHyperParameters.P_MIN: 0.05,
        DQNHyperParameters.LAYERS: [256]
    }

    dqn(game, hyper_parameters)

    return


if __name__ == "__main__":
    main()
