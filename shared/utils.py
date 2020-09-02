import torch
from datetime import datetime
from tqdm import tqdm
from shared.tensorboard_wrapper import TensorboardWrapper
from enum import Enum


class Algorithms(Enum):
    DQN = 'dqn'
    A2C = 'a2c'
    PPO = 'ppo'


class Stats:
    def __init__(self, algorithm, log_interval, precision, start):
        self.scores = []
        self.average_prob = []
        self.score = 0
        self.episode = 0
        self.steps = 0
        self.start = start
        self.algorithm = algorithm

        self.log_interval = log_interval
        self.precision = precision

    def done_update_stats(self, tensorboard_writer):
        self.scores.append(self.score)
        if self.episode % self.log_interval == 0:
            self.print_scores()

        now = datetime.now().timestamp()
        data = {}
        if self.algorithm == Algorithms.DQN:
            data = {'score': self.score, 'etc/steps': self.steps, 'etc/elapsed_time': now - self.start}
        elif self.algorithm == Algorithms.A2C or self.algorithm == Algorithms.PPO:
            data = {'score': self.score, 'etc/steps': self.steps, 'etc/elapsed_time': now - self.start,
                    'exploration/prob': sum(self.average_prob) / len(self.average_prob)}

        tensorboard_writer.save_scalars(data, self.episode)

        self.episode += 1
        self.steps = 0
        self.score = 0
        self.average_prob = []

    def print_scores(self):
        print("Episode: {} Rewards: {:.{prec}f} Max: {:.{prec}f} Min: {:.{prec}f}"
              .format(self.episode, sum(self.scores) / len(self.scores),
                      max(self.scores), min(self.scores), prec=self.precision))
        self.scores = []


def initialize_logging(game, algorithm):
    tq = tqdm()

    now = datetime.now()
    now_str = now.strftime("%y%m%d-%H%M")
    tensorboard_writer = TensorboardWrapper(game, algorithm.value, now_str)

    stats = Stats(algorithm=algorithm, log_interval=20, precision=1, start=now.timestamp())

    return tensorboard_writer, tq, stats


def data_to_torch(data, torch_type, device):
    data = torch.tensor(data, device=device, dtype=torch_type)
    return data
