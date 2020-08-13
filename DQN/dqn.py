import torch
import random
from tqdm import tqdm
from datetime import datetime
import time
from DQN.replay_buffer import ReplayBuffer
from DQN.model import Agent
from shared.tensorboard_wrapper import TensorboardWrapper
from shared.utils import data_to_torch


def dqn(env, hyper_parameters):
    # Hyper parameters
    train_seconds = hyper_parameters['train_seconds']
    batch_size = hyper_parameters['batch_size']
    buffer_size = hyper_parameters['buffer_size']
    learning_rate = hyper_parameters['learning_rate']
    target_update_frequency = hyper_parameters['target_update_frequency']
    gamma = hyper_parameters['gamma']
    p_decay = hyper_parameters['p_decay']
    p_min = hyper_parameters['p_min']
    layers = hyper_parameters['layers']

    buffer = ReplayBuffer(buffer_size)
    agent = Agent(layers, learning_rate, gamma, env.device)

    tq = tqdm()

    now = datetime.now()
    now_str = now.strftime("%y%m%d-%H%M")
    tensorboard_writer = TensorboardWrapper(now_str)

    score = 0
    epoch = 0

    prev_state = env.reset()
    now = time.time()
    start_time = now
    while now - start_time < train_seconds:
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
            tensorboard_writer.save_scalar('loss', loss, env.total_steps)
            tensorboard_writer.save_scalar('eps', eps, env.total_steps)
            tensorboard_writer.save_scalar('elapsed time', now - start_time, env.total_steps)

        score += reward

        if done:
            prev_state = env.reset()
            tensorboard_writer.save_scalar('score', score, epoch)

            epoch += 1
            score = 0
        else:
            prev_state = next_state

        if env.total_steps % target_update_frequency == 0:
            print("Updated Model")
            agent.update_target_model()

        now = time.time()
    return
