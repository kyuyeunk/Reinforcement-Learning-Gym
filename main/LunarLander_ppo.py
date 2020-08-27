from PPO.ppo import ppo, PPOHyperParameters


def main():
    game = 'LunarLander-v2'
    # Hyper parameters
    hyper_parameters = {
        PPOHyperParameters.TRAIN_EPISODES: 10000,
        PPOHyperParameters.LEARNING_RATE: 0.001,
        PPOHyperParameters.LAYERS: [128, 128],
        PPOHyperParameters.GAMMA: 0.99,
        PPOHyperParameters.LAMBDA: 0.95,
        PPOHyperParameters.EPS: 0.1,
        PPOHyperParameters.BATCH_SIZE: 512,
        PPOHyperParameters.K: 3
    }

    ppo(game, hyper_parameters)

    return


if __name__ == "__main__":
    main()
