import numpy as np


class Trajectory:

    def __init__(self, gamma):
        self.rewards = []
        self.observations = []
        self.actions = []
        self.value_predictions = []
        self.gamma = gamma

    def add_step(self, observation, action, value_prediction, reward):
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.value_predictions.append(value_prediction)

    def get_training_batch(self, bootstrap_value):
        reward_return = bootstrap_value
        advantages = []
        returns = []
        for i in reversed(range(len(self.rewards))):
            reward_return = self.rewards[i] + self.gamma * reward_return
            advantages.append(reward_return - self.value_predictions[i])
            returns.append(reward_return)

        advantages.reverse()
        returns.reverse()
        return self.observations, self.actions, advantages, returns

    def __eq__(self, other):
        '''useful for testing'''
        if len(self.rewards) != len(other.rewards):
            return False

        for i in range(len(self.rewards)):
            if not np.array_equal(self.observations[i], other.observations[i]):
                return False

            if (self.rewards[i] != other.rewards[i] or
                    self.actions[i] != other.actions[i] or 
                    self.value_predicitons != other.value_predicitons):
                return False

        return True
