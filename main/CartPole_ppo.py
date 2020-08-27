from PPO.ppo import ppo, PPOHyperParameters


def main():
    game = 'CartPole-v1'
    # Hyper parameters
    hyper_parameters = {
        PPOHyperParameters.TRAIN_EPISODES: 2000,
        PPOHyperParameters.LEARNING_RATE: 0.004,
        PPOHyperParameters.LAYERS: [256],
        PPOHyperParameters.GAMMA: 0.99,
        PPOHyperParameters.LAMBDA: 0.95,
        PPOHyperParameters.EPS: 0.2,
        PPOHyperParameters.BATCH_SIZE: 32,
        PPOHyperParameters.K: 3
    }

    ppo(game, hyper_parameters)

    return


if __name__ == "__main__":
    main()
