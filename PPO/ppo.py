import torch
from PPO.model import Agent
from enum import IntEnum
import numpy
from shared.gym_env import Environment
from shared import utils


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


def ppo(game, hyper_parameters):
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

    # Initialize device, environment, and agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment(game, device)
    n_inputs = numpy.prod(env.get_shape_observations())
    n_outputs = env.get_n_actions()

    layers = [n_inputs] + layers + [n_outputs]
    agent = Agent(layers, learning_rate, gamma, lmbda, eps, k, batch_size, device)

    # Initialize statistics related variables
    tensorboard_writer, tq, stats = utils.initialize_logging(game, utils.Algorithms.PPO)

    prev_state = env.reset()
    while stats.episode < train_episodes:
        tq.update(1)

        prob = agent.prob(prev_state).detach()
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()

        next_state, reward, done, info = env.step(action.item())

        agent.store_history(prev_state, action.unsqueeze(0), prob[action].unsqueeze(0), reward, next_state, done)

        if agent.is_buffer_full():
            actor_loss, critic_loss = agent.train()
            tensorboard_writer.save_scalar('loss/ppo_actor', actor_loss, env.total_steps)
            tensorboard_writer.save_scalar('loss/ppo_critic', critic_loss, env.total_steps)

        stats.score += reward.item()
        stats.average_prob.append(prob[action])

        if done:
            stats.done_update_stats(tensorboard_writer)
            prev_state = env.reset()
        else:
            prev_state = next_state
            stats.steps += 1

    return
