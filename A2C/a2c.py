import torch
from A2C.model import Agent
from enum import IntEnum
from shared import utils
from datetime import datetime


class A2CHyperParameters(IntEnum):
    TRAIN_EPISODES = 0
    ACTOR_LEARNING_RATE = 1
    CRITIC_LEARNING_RATE = 2
    GAMMA = 3
    ACTOR_LAYERS = 4
    CRITIC_LAYERS = 5
    N_PARAMETERS = 6


def a2c(env, hyper_parameters=None, load_timestamp=None):
    start_time = datetime.now()
    if load_timestamp:
        hyper_parameters = utils.HyperParameterIO.load_hyperparameters(env.get_game_name(), 'a2c', load_timestamp)
    else:
        utils.HyperParameterIO.save_hyperparameters(env.get_game_name(), 'a2c', start_time, hyper_parameters)

    assert(len(hyper_parameters) == A2CHyperParameters.N_PARAMETERS)
    # Hyper parameters
    train_episodes = hyper_parameters[A2CHyperParameters.TRAIN_EPISODES]
    actor_learning_rate = hyper_parameters[A2CHyperParameters.ACTOR_LEARNING_RATE]
    critic_learning_rate = hyper_parameters[A2CHyperParameters.CRITIC_LEARNING_RATE]
    gamma = hyper_parameters[A2CHyperParameters.GAMMA]
    actor_layers = hyper_parameters[A2CHyperParameters.ACTOR_LAYERS]
    critic_layers = hyper_parameters[A2CHyperParameters.CRITIC_LAYERS]

    # Initialize agent
    agent = Agent(actor_layers, critic_layers, actor_learning_rate, critic_learning_rate, gamma, env.device)
    if load_timestamp is not None:
        agent.load_model(env.get_game_name(), load_timestamp)

    # Initialize statistics related variables
    log_interval = 20
    tensorboard_writer, tq, stats = utils.initialize_logging(env.get_game_name(), utils.Algorithms.A2C,
                                                             log_interval, start_time, load_timestamp)

    if load_timestamp is not None:
        start_time = datetime.strptime(load_timestamp, "%y%m%d-%H%M")
    prev_state = env.reset()
    while stats.episode < train_episodes:
        tq.update(1)

        prob = agent.prob(prev_state)[0]
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()

        next_state, reward, done, info = env.step(action.item())

        actor_loss, critic_loss = agent.train(prev_state, prob[action].view(1, 1),
                                              reward, next_state, done)

        tensorboard_writer.save_scalar('a2c/loss_actor', actor_loss, stats.steps)
        tensorboard_writer.save_scalar('a2c/loss_critic', critic_loss, stats.steps)
        tensorboard_writer.save_scalar('a2c/probability', prob[action], stats.steps)

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
