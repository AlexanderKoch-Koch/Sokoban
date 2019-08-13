import tensorflow as tf
import numpy as np
from trajectory import Trajectory
from observation_preprocessing import preprocess_atari
import matplotlib.pyplot as plt
import time
import  math

class Actor:

    def __init__(self, env, global_model, local_model, gamma, summary_writer):
        self.global_network = global_model
        self.gamma = gamma
        self.env = env
        # self.env = gym.make("CartPole-v0")
        self.model = local_model
        self.summary_writer = summary_writer

    def start(self):
        t_max = 5
        train_step = 0
        with self.summary_writer.as_default():
            for episode in range(10000000000):
                observation = self.env.reset()#astype(dtype=np.float32)
                done = False
                reward_sum = 0
                while not done:
                    time.sleep(0.01)
                    # memory = []
                    trajectory = Trajectory(self.gamma)
                    self.model.update_weights(self.global_network)
                    for t in range(t_max):
                        pi, v = self.model.predict(observation)
                        # pi /= np.sum(pi)
                        action = np.random.choice(range(4), p=pi)
                        observation_next, reward, done, _ = self.env.step(action)

                        # self.env.render()
                        reward_sum += reward
                        trajectory.add_step(observation, action, v, reward)
                        # observation = observation_next.astype(np.float32)
                        if done:
                            break

                    bootstrap_state_value = 0
                    if not done:
                        # predict state value in last observed state
                        _, v = self.model.predict(observation)
                        bootstrap_state_value = v
                    trajectory.set_bootstrap_value(bootstrap_state_value)

                    # log every 10 train steps
                    if train_step % 10 == 0:
                        self.model.train(trajectory, self.global_network, train_step)
                        '''if self.summary_writer is not None:
                            summary = tf.Summary(
                                value=[tf.Summary.Value(tag="sum_train_step_reward", simple_value=sum(trajectory.rewards))])
                            self.summary_writer.add_summary(summary, train_step)'''
                    else:
                        self.model.train(trajectory, self.global_network, train_step)
                        
                    train_step += 1

                tf.summary.scalar("reward_per_episode", reward_sum, step=episode)
                '''if self.summary_writer is not None:
                    summary = tf.Summary(value=[tf.Summary.Value(tag="reward_per_episode", simple_value=reward_sum)])
                    self.summary_writer.add_summary(summary, episode)'''

