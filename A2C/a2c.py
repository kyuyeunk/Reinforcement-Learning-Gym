from tqdm import tqdm
from datetime import datetime
import time
from A2C.model import Agent
from shared.tensorboard_wrapper import TensorboardWrapper
from enum import IntEnum


class HyperParameters(IntEnum):
    TRAIN_EPISODES = 0
    ACTOR_LEARNING_RATE = 1
    CRITIC_LEARNING_RATE = 2
    GAMMA = 3
    ACTOR_LAYERS = 4
    CRITIC_LAYERS = 5
    N_PARAMETERS = 6


def a2c(env, hyper_parameters):
    assert(len(hyper_parameters) == HyperParameters.N_PARAMETERS)
    # Hyper parameters
    train_episodes = hyper_parameters[HyperParameters.TRAIN_EPISODES]
    actor_learning_rate = hyper_parameters[HyperParameters.ACTOR_LEARNING_RATE]
    critic_learning_rate = hyper_parameters[HyperParameters.CRITIC_LEARNING_RATE]
    gamma = hyper_parameters[HyperParameters.GAMMA]
    actor_layers = hyper_parameters[HyperParameters.ACTOR_LAYERS]
    critic_layers = hyper_parameters[HyperParameters.CRITIC_LAYERS]

    agent = Agent(actor_layers, critic_layers, actor_learning_rate, critic_learning_rate, gamma, env.device)

    tq = tqdm()

    now = datetime.now()
    now_str = now.strftime("%y%m%d-%H%M")
    tensorboard_writer = TensorboardWrapper(now_str)

    score = 0
    episode = 0

    prev_state = env.reset()
    while episode < train_episodes:
        tq.update(1)

        action = agent.decide(prev_state)
        next_state, reward, done, info = env.step(action)

        actor_loss, critic_loss = agent.train(prev_state, reward, next_state, done)

        tensorboard_writer.save_scalar('actor_loss', actor_loss, env.total_steps)
        tensorboard_writer.save_scalar('critic_loss', critic_loss, env.total_steps)

        score += reward

        if done:
            prev_state = env.reset()
            tensorboard_writer.save_scalar('score', score, episode)

            episode += 1
            score = 0
        else:
            prev_state = next_state

    return
