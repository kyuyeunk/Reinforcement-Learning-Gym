import torch
from shared.gym_env import Environment
import random
from tqdm import tqdm
from datetime import datetime
import time
from DQN.replay_buffer import ReplayBuffer
from DQN.model import Agent
from shared.tensorboard_wrapper import TensorboardWrapper
from shared.utils import data_to_torch


def main():
    # Hyper parameters
    train_seconds = 60 * 60
    batch_size = 2048
    buffer_size = 500000
    learning_rate = 0.00005
    target_update_frequency = 2048
    gamma = 0.99

    p_decay = 0.001
    p_min = 0.05

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = Environment('CartPole-v1', device)
    prev_state = env.reset()

    buffer = ReplayBuffer(buffer_size)
    agent = Agent(env.get_shape_observations(), env.get_n_actions(), learning_rate, gamma, device)

    tq = tqdm()

    now = datetime.now()
    now_str = now.strftime("%y%m%d-%H%M")
    tensorboard_writer = TensorboardWrapper(now_str)

    score = 0
    epoch = 0

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
        buffer.insert(prev_state, data_to_torch([action], torch.long, device), reward, next_state, done)

        samples = buffer.sample(batch_size)
        if samples:
            loss = agent.train(samples)
            tensorboard_writer.save_scalar('loss', loss, env.total_steps)
            tensorboard_writer.save_scalar('eps', eps, env.total_steps)
            tensorboard_writer.save_scalar('elapsed time', now - start_time, env.total_steps)

        if done:
            prev_state = env.reset()
            tensorboard_writer.save_scalar('score', score, epoch)

            epoch += 1
            score = 0
        else:
            prev_state = next_state

        score += 1

        if env.total_steps % target_update_frequency == 0:
            print("Updated Model")
            agent.update_target_model()

        now = time.time()
    return


if __name__ == "__main__":
    main()
