import gym
import numpy as np
from collections import deque
import random


class AtariWrapper():
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.last_observations = deque(maxlen=4)

    def reset(self):
        self.last_observations.append(self.preprocess(self.env.reset()))

        # fill up deque
        for _ in range(3):
            observation, _, _, _ = self.env.step(random.randint(0, 3))  # execute fire action to make game start
            self.last_observations.append(self.preprocess(observation))

        return np.stack(self.last_observations, axis=2)

    def step(self, action):
        """
        executes action in environment
        :param action:
        :return: last_4_observations, reward, is_done, info
        """
        observation, reward, done, info = self.env.step(action)
        if reward > 1.0:
            reward = 1.0
        elif reward < -1.0:
            reward = -1.0
        self.last_observations.append(self.preprocess(observation))
        state = np.stack(self.last_observations, axis=2)
        return state, reward, done, info

    def render(self):
        self.env.render()

    @staticmethod
    def to_grayscale(img):
        return np.mean(img, axis=2).astype(np.uint8)

    @staticmethod
    def downsample(img):
        return img[::2, ::2]

    @staticmethod
    def normalize(img):
        return np.divide(img, 255)

    @staticmethod
    def preprocess(img):
        downsampled = AtariWrapper.downsample(img)
        grayscale = AtariWrapper.to_grayscale(downsampled)
        cropped = grayscale[16:96, :]
        normalized = AtariWrapper.normalize(cropped)
        return normalized