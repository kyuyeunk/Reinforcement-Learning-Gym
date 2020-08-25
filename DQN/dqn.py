import torch
import random
from tqdm import tqdm
from datetime import datetime
from DQN.replay_buffer import ReplayBuffer
from DQN.model import Agent
from shared.tensorboard_wrapper import TensorboardWrapper
from shared.utils import data_to_torch
from enum import IntEnum
import numpy
from shared.gym_env import Environment


class HyperParameters(IntEnum):
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


def dqn(game, hyper_parameters):
    assert(len(hyper_parameters) == HyperParameters.N_PARAMETERS)
    # Hyper parameters
    train_episodes = hyper_parameters[HyperParameters.TRAIN_EPISODES]
    batch_size = hyper_parameters[HyperParameters.BATCH_SIZE]
    buffer_size = hyper_parameters[HyperParameters.BUFFER_SIZE]
    learning_rate = hyper_parameters[HyperParameters.LEARNING_RATE]
    target_update_frequency = hyper_parameters[HyperParameters.TARGET_UPDATE_FREQUENCY]
    gamma = hyper_parameters[HyperParameters.GAMMA]
    p_decay = hyper_parameters[HyperParameters.P_DECAY]
    p_min = hyper_parameters[HyperParameters.P_MIN]
    layers = hyper_parameters[HyperParameters.LAYERS]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment(game, device)
    n_inputs = numpy.prod(env.get_shape_observations())
    n_outputs = env.get_n_actions()

    layers = [n_inputs] + layers + [n_outputs]
    agent = Agent(layers, learning_rate, gamma, env.device)
    buffer = ReplayBuffer(buffer_size)

    tq = tqdm()

    now = datetime.now()
    now_str = now.strftime("%y%m%d-%H%M")
    tensorboard_writer = TensorboardWrapper(game, "dqn", now_str)

    log_interval = 20
    scores = []
    score = 0
    episode = 0
    steps = 0
    start = now.timestamp()

    prev_state = env.reset()
    while episode < train_episodes:
        tq.update(1)

        eps = pow((1 - p_min), p_decay * env.total_steps)
        if random.random() > eps:
            action = agent.predict(prev_state)
        else:
            action = random.randrange(env.get_n_actions())

        next_state, reward, done, info = env.step(action)
        buffer.insert(prev_state, data_to_torch([action], torch.long, env.device), reward, next_state, done)

        samples = buffer.sample(batch_size)
        if samples:
            loss = agent.train(samples)
            tensorboard_writer.save_scalar('loss/dqn', loss, env.total_steps)
            tensorboard_writer.save_scalar('exploration/eps', eps, env.total_steps)

        score += reward.item()

        if done:
            tensorboard_writer.save_scalar('score', score, episode)
            tensorboard_writer.save_scalar('etc/steps', steps, episode)
            now = datetime.now().timestamp()
            tensorboard_writer.save_scalar('etc/elapsed_time', now - start, episode)

            scores.append(score)
            if episode % log_interval == 0:
                print("Episode: {} Rewards: {:.1f} Max: {:.1f} Min: {:.1f}".format(episode, sum(scores) / len(scores), max(scores), min(scores)))
                scores = []

            prev_state = env.reset()
            steps = 0
            episode += 1
            score = 0
        else:
            prev_state = next_state
            steps += 1

        if env.total_steps % target_update_frequency == 0:
            agent.update_target_model()

    return
