import torch
from PPO.model import Agent
from enum import IntEnum
from shared import utils
from datetime import datetime
import pickle
import os

class PPOHyperParameters(IntEnum):
    TRAIN_EPISODES = 0
    LEARNING_RATE = 1
    LAYERS = 2
    GAMMA = 3
    BATCH_SIZE = 4
    LAMBDA = 5
    EPS = 6
    K = 7
    N_PARAMETERS = 8


def ppo(env, hyper_parameters=None, load_timestamp=None):
    start_time = datetime.now()
    if load_timestamp:
        hyper_parameters = utils.HyperParameterIO.load_hyperparameters(env.get_game_name(), 'ppo', load_timestamp)
    else:
        utils.HyperParameterIO.save_hyperparameters(env.get_game_name(), 'ppo', start_time, hyper_parameters)

    assert(len(hyper_parameters) == PPOHyperParameters.N_PARAMETERS)
    # Hyper parameters
    train_episodes = hyper_parameters[PPOHyperParameters.TRAIN_EPISODES]
    learning_rate = hyper_parameters[PPOHyperParameters.LEARNING_RATE]
    layers = hyper_parameters[PPOHyperParameters.LAYERS]
    gamma = hyper_parameters[PPOHyperParameters.GAMMA]
    batch_size = hyper_parameters[PPOHyperParameters.BATCH_SIZE]
    lmbda = hyper_parameters[PPOHyperParameters.LAMBDA]
    eps = hyper_parameters[PPOHyperParameters.EPS]
    k = hyper_parameters[PPOHyperParameters.K]

    # Initialize agent
    agent = Agent(layers, learning_rate, gamma, lmbda, eps, k, batch_size, env.device)
    if load_timestamp is not None:
        agent.load_model(env.get_game_name(), load_timestamp)

    # Initialize statistics related variables
    log_interval = 20
    tensorboard_writer, tq, stats = utils.initialize_logging(env.get_game_name(), utils.Algorithms.PPO,
                                                             log_interval, start_time, load_timestamp)

    if load_timestamp is not None:
        start_time = datetime.strptime(load_timestamp, "%y%m%d-%H%M")
    prev_state = env.reset()
    while stats.episode < train_episodes:
        tq.update(1)

        prob = agent.prob(prev_state)[0].detach()
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()

        next_state, reward, done, info = env.step(action.item())

        agent.store_history(prev_state, action.view(1, 1),
                            prob[action].view(1, 1), reward, next_state, done)

        if agent.is_buffer_full():
            actor_loss, critic_loss = agent.train()
            tensorboard_writer.save_scalar('ppo/loss_actor', actor_loss, stats.steps)
            tensorboard_writer.save_scalar('ppo/loss_critic', critic_loss, stats.steps)
        tensorboard_writer.save_scalar('ppo/probability', prob[action], stats.steps)

        stats.score += reward.item()
        stats.steps += 1

        if done:
            if stats.episode % log_interval == 0:
                agent.save_model(env.get_game_name(), start_time)
                stats.save_stat(env.get_game_name(), start_time)
            stats.done_update_stats(tensorboard_writer)
            prev_state = env.reset()
        else:
            prev_state = next_state

    return
