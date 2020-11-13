import torch
import random
from DQN.replay_buffer import ReplayBuffer
from DQN.model import Agent
from enum import IntEnum
from shared import utils
from datetime import datetime


class DQNHyperParameters(IntEnum):
    TRAIN_EPISODES = 0
    BATCH_SIZE = 1
    BUFFER_SIZE = 2
    LEARNING_RATE = 3
    TARGET_UPDATE_FREQUENCY = 4
    GAMMA = 5
    P_DECAY = 6
    P_MIN = 7
    LAYERS = 8
    N_PARAMETERS = 9


def dqn(env, hyper_parameters=None, load_timestamp=None):
    start_time = datetime.now()
    if load_timestamp:
        hyper_parameters = utils.HyperParameterIO.load_hyperparameters(env.get_game_name(), 'dqn', load_timestamp)
    else:
        utils.HyperParameterIO.save_hyperparameters(env.get_game_name(), 'dqn', start_time, hyper_parameters)

    assert(len(hyper_parameters) == DQNHyperParameters.N_PARAMETERS)
    # Hyper parameters
    train_episodes = hyper_parameters[DQNHyperParameters.TRAIN_EPISODES]
    batch_size = hyper_parameters[DQNHyperParameters.BATCH_SIZE]
    buffer_size = hyper_parameters[DQNHyperParameters.BUFFER_SIZE]
    learning_rate = hyper_parameters[DQNHyperParameters.LEARNING_RATE]
    target_update_frequency = hyper_parameters[DQNHyperParameters.TARGET_UPDATE_FREQUENCY]
    gamma = hyper_parameters[DQNHyperParameters.GAMMA]
    p_decay = hyper_parameters[DQNHyperParameters.P_DECAY]
    p_min = hyper_parameters[DQNHyperParameters.P_MIN]
    layers = hyper_parameters[DQNHyperParameters.LAYERS]

    # Initialize agent
    agent = Agent(layers, learning_rate, gamma, env.device)
    if load_timestamp is not None:
        agent.load_model(env.get_game_name(), load_timestamp)
    buffer = ReplayBuffer(buffer_size)

    # Initialize statistics related variables
    log_interval = 20
    tensorboard_writer, tq, stats = utils.initialize_logging(env.get_game_name(), utils.Algorithms.DQN,
                                                             log_interval, start_time, load_timestamp)

    if load_timestamp is not None:
        start_time = datetime.strptime(load_timestamp, "%y%m%d-%H%M")
    prev_state = env.reset()
    while stats.episode < train_episodes:
        tq.update(1)

        eps = pow((1 - p_min), p_decay * env.total_steps)
        if random.random() > eps:
            action = agent.predict(prev_state)
        else:
            action = random.randrange(env.get_n_actions())

        next_state, reward, done, info = env.step(action)
        buffer.insert(prev_state, utils.data_to_torch([[action]], torch.long, env.device), reward, next_state, done)

        samples = buffer.sample(batch_size)
        if samples:
            loss = agent.train(samples)
            tensorboard_writer.save_scalar('dqn/loss', loss, stats.steps)
            tensorboard_writer.save_scalar('dqn/eps', eps, stats.steps)

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

        if env.total_steps % target_update_frequency == 0:
            agent.update_target_model()

    return
