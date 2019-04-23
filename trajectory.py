import numpy as np


class Trajectory:

    def __init__(self, gamma):
        self.rewards = []
        self.observations = []
        self.actions = []
        self.value_predictions = []
        self.gamma = gamma
        self.bootstrap_value = None

    def add_step(self, observation, action, value_prediction, reward):
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.value_predictions.append(value_prediction)

    def set_bootstrap_value(self, bootstrap_value):
        self.bootstrap_value = bootstrap_value

    def get_training_batch(self):
        assert self.bootstrap_value is not None, "bootstrap value has to be set before calling get_training_batch()"
        reward_return = self.bootstrap_value

        advantages = []
        returns = []
        for i in reversed(range(len(self.rewards))):
            reward_return = self.rewards[i] + self.gamma * reward_return
            advantages.append(reward_return - self.value_predictions[i])
            returns.append(reward_return)

        advantages.reverse()
        returns.reverse()
        return self.observations, self.actions, advantages, returns
