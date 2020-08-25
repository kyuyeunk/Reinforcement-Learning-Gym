import torch
from tqdm import tqdm
from datetime import datetime
from PPO.model import Agent
from shared.tensorboard_wrapper import TensorboardWrapper
from enum import IntEnum
from shared.gym_env import Environment
import numpy


class HyperParameters(IntEnum):
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
    assert(len(hyper_parameters) == HyperParameters.N_PARAMETERS)
    # Hyper parameters
    train_episodes = hyper_parameters[HyperParameters.TRAIN_EPISODES]
    learning_rate = hyper_parameters[HyperParameters.LEARNING_RATE]
    layers = hyper_parameters[HyperParameters.LAYERS]
    gamma = hyper_parameters[HyperParameters.GAMMA]
    batch_size = hyper_parameters[HyperParameters.BATCH_SIZE]
    lmbda = hyper_parameters[HyperParameters.LAMBDA]
    eps = hyper_parameters[HyperParameters.EPS]
    k = hyper_parameters[HyperParameters.K]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment(game, device)
    n_inputs = numpy.prod(env.get_shape_observations())
    n_outputs = env.get_n_actions()

    layers = [n_inputs] + layers + [n_outputs]
    agent = Agent(layers, learning_rate, gamma, lmbda, eps, k, batch_size, env.device)

    tq = tqdm()

    now = datetime.now()
    now_str = now.strftime("%y%m%d-%H%M")
    tensorboard_writer = TensorboardWrapper(game, "ppo", now_str)

    log_interval = 20
    scores = []
    score = 0
    episode = 0
    average_prob = []
    steps = 0
    start = now.timestamp()

    prev_state = env.reset()
    while episode < train_episodes:
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

        score += reward.item()
        average_prob.append(prob[action])

        if done:
            tensorboard_writer.save_scalar('score', score, episode)
            tensorboard_writer.save_scalar('exploration/prob', sum(average_prob)/len(average_prob), episode)
            tensorboard_writer.save_scalar('etc/steps', steps, episode)
            now = datetime.now().timestamp()
            tensorboard_writer.save_scalar('etc/elapsed_time', now - start, episode)

            scores.append(score)
            if episode % log_interval == 0:
                print("Episode: {} Rewards: {:.1f} Max: {:.1f} Min: {:.1f}".format(episode, sum(scores)/len(scores), max(scores), min(scores)))
                scores = []

            prev_state = env.reset()
            steps = 0
            average_prob = []
            episode += 1
            score = 0
        else:
            prev_state = next_state
            steps += 1

    return
