import tensorflow as tf
import numpy as np
from datetime import datetime
from trajectory import Trajectory


class Actor:

    def __init__(self, name, tf_session, env, global_model, local_model, gamma):
        self.sess = tf_session
        self.global_network = global_model
        self.gamma = gamma
        self.env = env
        # self.env = gym.make("CartPole-v0")
        self.model = local_model

        self.summary_writer = None
        if name == "actor_0":
            # only one thread should create a summary
            log_dir = "./runs/" + datetime.now().strftime("%d.%m_%H:%M:%S")
            self.summary_writer = tf.summary.FileWriter(log_dir, tf_session.graph)
            self.summary_writer.flush()

    def start(self):
        t_max = 5
        train_step = 0
        with self.sess.as_default(), self.sess.graph.as_default():
            for episode in range(100000000):
                observation = self.env.reset()
                done = False
                reward_sum = 0
                while not done:
                    # memory = []
                    trajectory = Trajectory(self.gamma)
                    self.model.copy_global_weights(self.sess)
                    for t in range(t_max):
                        pi, v = self.model.predict(self.sess, observation)
                        action = np.random.choice(range(4), p=pi)
                        observation_next, reward, done, _ = self.env.step(action)
                        # self.env.render()

                        reward_sum += reward
                        trajectory.add_step(observation, action, v, reward)
                        observation = observation_next
                        if done:
                            break

                    bootstrap_state_value = 0
                    if not done:
                        # predict state value in last observed state
                        _, v = self.model.predict(self.sess, observation)
                        bootstrap_state_value = v
                    trajectory.set_bootstrap_value(bootstrap_state_value)
                    self.model.train(self.sess, trajectory, train_step, summary_writer=self.summary_writer)
                    train_step += 1

                if self.summary_writer is not None:
                    summary = tf.Summary(value=[tf.Summary.Value(tag="reward_per_episode", simple_value=reward_sum)])
                    self.summary_writer.add_summary(summary, episode)
