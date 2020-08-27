import torch
from A2C.model import Agent
from enum import IntEnum
import numpy
from shared.gym_env import Environment
from shared import utils


class A2CHyperParameters(IntEnum):
    TRAIN_EPISODES = 0
    ACTOR_LEARNING_RATE = 1
    CRITIC_LEARNING_RATE = 2
    GAMMA = 3
    ACTOR_LAYERS = 4
    CRITIC_LAYERS = 5
    N_PARAMETERS = 6


def a2c(game, hyper_parameters):
    assert(len(hyper_parameters) == A2CHyperParameters.N_PARAMETERS)
    # Hyper parameters
    train_episodes = hyper_parameters[A2CHyperParameters.TRAIN_EPISODES]
    actor_learning_rate = hyper_parameters[A2CHyperParameters.ACTOR_LEARNING_RATE]
    critic_learning_rate = hyper_parameters[A2CHyperParameters.CRITIC_LEARNING_RATE]
    gamma = hyper_parameters[A2CHyperParameters.GAMMA]
    actor_layers = hyper_parameters[A2CHyperParameters.ACTOR_LAYERS]
    critic_layers = hyper_parameters[A2CHyperParameters.CRITIC_LAYERS]

    # Initialize device, environment, and agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment(game, device)
    n_inputs = numpy.prod(env.get_shape_observations())
    n_outputs = env.get_n_actions()

    actor_layers = [n_inputs] + actor_layers + [n_outputs]
    critic_layers = [n_inputs] + critic_layers + [1]
    agent = Agent(actor_layers, critic_layers, actor_learning_rate, critic_learning_rate, gamma, device)

    # Initialize statistics related variables
    tensorboard_writer, tq, stats = utils.initialize_logging(game, utils.Algorithms.A2C)

    prev_state = env.reset()
    while stats.episode < train_episodes:
        tq.update(1)

        prob = agent.prob(prev_state)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()

        next_state, reward, done, info = env.step(action.item())

        actor_loss, critic_loss = agent.train(prev_state, prob[action], reward, next_state, done)

        tensorboard_writer.save_scalar('loss/a2c_actor', actor_loss, env.total_steps)
        tensorboard_writer.save_scalar('loss/a2c_critic', critic_loss, env.total_steps)

        stats.score += reward.item()
        stats.average_prob.append(prob[action])

        if done:
            stats.done_update_stats(tensorboard_writer)
            prev_state = env.reset()
        else:
            prev_state = next_state
            stats.steps += 1

    return
