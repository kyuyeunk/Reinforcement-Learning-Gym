import torch
from tqdm import tqdm
from datetime import datetime
from A2C.model import Agent
from shared.tensorboard_wrapper import TensorboardWrapper
from enum import IntEnum
import numpy
from shared.gym_env import Environment


class HyperParameters(IntEnum):
    TRAIN_EPISODES = 0
    ACTOR_LEARNING_RATE = 1
    CRITIC_LEARNING_RATE = 2
    GAMMA = 3
    ACTOR_LAYERS = 4
    CRITIC_LAYERS = 5
    N_PARAMETERS = 6


def a2c(game, hyper_parameters):
    assert(len(hyper_parameters) == HyperParameters.N_PARAMETERS)
    # Hyper parameters
    train_episodes = hyper_parameters[HyperParameters.TRAIN_EPISODES]
    actor_learning_rate = hyper_parameters[HyperParameters.ACTOR_LEARNING_RATE]
    critic_learning_rate = hyper_parameters[HyperParameters.CRITIC_LEARNING_RATE]
    gamma = hyper_parameters[HyperParameters.GAMMA]
    actor_layers = hyper_parameters[HyperParameters.ACTOR_LAYERS]
    critic_layers = hyper_parameters[HyperParameters.CRITIC_LAYERS]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment(game, device)
    n_inputs = numpy.prod(env.get_shape_observations())
    n_outputs = env.get_n_actions()

    actor_layers = [n_inputs] + actor_layers + [n_outputs]
    critic_layers = [n_inputs] + critic_layers + [1]
    agent = Agent(actor_layers, critic_layers, actor_learning_rate, critic_learning_rate, gamma, env.device)

    tq = tqdm()

    now = datetime.now()
    now_str = now.strftime("%y%m%d-%H%M")
    tensorboard_writer = TensorboardWrapper(game, "a2c", now_str)

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

        prob = agent.prob(prev_state)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()

        next_state, reward, done, info = env.step(action.item())

        actor_loss, critic_loss = agent.train(prev_state, prob[action], reward, next_state, done)

        tensorboard_writer.save_scalar('loss/a2c_actor', actor_loss, env.total_steps)
        tensorboard_writer.save_scalar('loss/a2c_critic', critic_loss, env.total_steps)

        score += reward.item()
        average_prob.append(prob[action])

        if done:
            tensorboard_writer.save_scalar('score', score, episode)
            tensorboard_writer.save_scalar('exploration/prob', sum(average_prob) / len(average_prob), episode)
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
