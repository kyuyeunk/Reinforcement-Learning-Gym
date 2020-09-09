import torch
from datetime import datetime
from tqdm import tqdm
from shared.tensorboard_wrapper import TensorboardWrapper
from enum import Enum
import pickle


class Algorithms(Enum):
    DQN = 'dqn'
    A2C = 'a2c'
    PPO = 'ppo'


class Stats:
    def __init__(self, algorithm, log_interval, precision, start):
        self.scores = []
        self.score = 0
        self.episode = 0
        self.steps = 0
        self.previous_steps = 0
        self.previous_time = start
        self.passed_time = 0
        self.algorithm = algorithm

        self.log_interval = log_interval
        self.precision = precision

    def done_update_stats(self, tensorboard_writer):
        self.scores.append(self.score)
        if self.episode % self.log_interval == 0:
            self.print_scores()

        now = datetime.now().timestamp()
        self.passed_time += now - self.previous_time
        episode_steps = self.steps - self.previous_steps

        tensorboard_writer.save_scalar('score', self.score, self.episode)
        tensorboard_writer.save_scalar('etc/steps_per_episode', episode_steps, self.episode)
        tensorboard_writer.save_scalar('etc/elapsed_time', self.passed_time, self.episode)

        self.previous_time = now
        self.previous_steps = self.steps
        self.episode += 1
        self.score = 0

    def print_scores(self):
        print("Episode: {} Rewards: {:.{prec}f} Max: {:.{prec}f} Min: {:.{prec}f}"
              .format(self.episode, sum(self.scores) / len(self.scores),
                      max(self.scores), min(self.scores), prec=self.precision))
        self.scores = []

    def save_stat(self, game, timestamp):
        time_str = timestamp.strftime("%y%m%d-%H%M")
        pickle.dump(self, open('runs/{}_{}_{}/stat.st'.format(game, self.algorithm.value, time_str), 'wb'))

    @staticmethod
    def load_stat(game, algorithm, timestamp, start_time):
        stats = pickle.load(open('runs/{}_{}_{}/stat.st'.format(game, algorithm, timestamp), 'rb'))

        stats.score = 0
        stats.previous_steps = stats.steps
        stats.previous_time = start_time

        return stats


def initialize_logging(game, algorithm, log_interval, start_time, load_timestamp):
    tq = tqdm()

    if load_timestamp is not None:
        tensorboard_writer = TensorboardWrapper(game, algorithm.value, load_timestamp)
        stats = Stats.load_stat(game, algorithm.value, load_timestamp, start_time.timestamp())
    else:
        now_str = start_time.strftime("%y%m%d-%H%M")
        tensorboard_writer = TensorboardWrapper(game, algorithm.value, now_str)
        stats = Stats(algorithm=algorithm, log_interval=log_interval, precision=1, start=start_time.timestamp())

    return tensorboard_writer, tq, stats


def data_to_torch(data, torch_type, device):
    data = torch.tensor(data, device=device, dtype=torch_type)
    return data
