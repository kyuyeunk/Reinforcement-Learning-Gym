import gym
import torch
from shared.utils import data_to_torch


class Environment:
    def __init__(self, make, device):
        self.env = gym.make(make)
        self.total_steps = 0
        self.device = device

    def render(self):
        self.env.render()

    # Reset the environment and return new observation
    def reset(self):
        observation = self.env.reset()
        return data_to_torch(observation, torch.float32, self.device).flatten()

    # Perform step and return torch friendly observation and reward
    def step(self, action):
        self.total_steps += 1
        observation, reward, done, info = self.env.step(action)
        observation = data_to_torch(observation, torch.float32, self.device).flatten()
        reward = data_to_torch([reward], torch.float32, self.device)
        done = data_to_torch(done, torch.bool, self.device)
        return observation, reward, done, info

    def close(self):
        self.env.close()

    def get_n_actions(self):
        return self.env.action_space.n

    def get_shape_observations(self):
        return self.env.observation_space.shape

    def get_n_total_steps(self):
        return self.total_steps
