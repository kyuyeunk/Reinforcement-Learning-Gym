import gym
import torch
import numpy
from shared.utils import data_to_torch
from shared.image_processor import ImageProcessor, ImageProcessDimension
from enum import Enum


class Game:
    def __init__(self, name, use_image_input, image_dimension=None):
        self.name = name
        self.use_image_input = use_image_input
        self.image_dimension = image_dimension
        if self.use_image_input:
            assert self.image_dimension


class GameList(Enum):
    def __new__(cls, name, image_input, image_dimension=None):
        return Game(name, image_input, image_dimension=image_dimension)
    CartPole = ('CartPole-v1', False, ImageProcessDimension(24, 36, 0, 600, 180, 325))
    LunarLander = ('LunarLander-v2', False)


class Environment:
    def __init__(self, game, device, force_image_input=False, n_frames=4):
        self.game = game
        self.env = gym.make(game.name)
        self.total_steps = 0
        self.device = device

        self.n_frames = n_frames
        self.force_image_input = force_image_input
        if self.use_image_input():
            assert self.game.image_dimension
            self.image_processor = ImageProcessor(n_frames, game.image_dimension, device)

    def render(self):
        return self.env.render('rgb_array')

    # Reset the environment and return new observation
    def reset(self):
        observation = self.env.reset()
        if self.use_image_input():
            if self.force_image_input:
                observation = self.render()
            self.image_processor.reset_screen()
            return self.image_processor.process_screen(observation)
        else:
            return data_to_torch([observation], torch.float32, self.device)

    # Perform step and return torch friendly observation and reward
    def step(self, action):
        self.total_steps += 1
        observation, reward, done, info = self.env.step(action)
        if self.use_image_input():
            if self.force_image_input:
                observation = self.render()
            observation = self.image_processor.process_screen(observation)
        else:
            observation = data_to_torch([observation], torch.float32, self.device)
        reward = data_to_torch([[reward]], torch.float32, self.device)
        done = data_to_torch([[done]], torch.bool, self.device)
        return observation, reward, done, info

    def close(self):
        self.env.close()

    def get_n_actions(self):
        return self.env.action_space.n

    def get_shape_obs(self):
        return self.env.observation_space.shape

    def get_n_obs(self):
        return numpy.prod(self.get_shape_obs())

    def get_n_total_steps(self):
        return self.total_steps

    def get_game_name(self):
        if self.force_image_input:
            return self.game.name + "-image"
        else:
            return self.game.name

    def use_image_input(self):
        return self.force_image_input or self.game.use_image_input
