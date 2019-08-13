import tensorflow as tf
import numpy as np
from trajectory import Trajectory
from observation_preprocessing import preprocess_atari
import matplotlib.pyplot as plt
import time
import  math
from absl import flags
FLAGS = flags.FLAGS

class Actor:

    def __init__(self, env, global_model, local_model, summary_writer):
        self.global_model = global_model
        self.env = env
        self.model = local_model
        self.summary_writer = summary_writer

    def start(self):
        t_max = 5
        train_step = 0
        with self.summary_writer.as_default():
            for episode in range(10000000000):
                observation = self.env.reset()
                done = False
                reward_sum = 0
                while not done:
                    time.sleep(0.01)
                    trajectory = Trajectory(FLAGS.gamma)
                    self.model.set_weights(self.global_model.get_weights())
                    for t in range(t_max):
                        model_input = np.expand_dims(observation, axis=0)
                        pi, v = self.model.predict(model_input)
                        pi = pi[0]; v = v[0]
                        action = np.random.choice(range(4), p=pi)
                        observation_next, reward, done, _ = self.env.step(action)

                        # self.env.render()
                        reward_sum += reward
                        trajectory.add_step(observation, action, v, reward)
                        if done:
                            break
                    
                    if done:
                        bootstrap_state_value = 0
                    else:
                        bootstrap_state_value = v

                    self.train(trajectory, bootstrap_state_value)
                    train_step += 1

                tf.summary.scalar("reward_per_episode", reward_sum, step=episode)

    def train(self, trajectory, bootstrap_value):
        observations, actions, advantages, returns = trajectory.get_training_batch(bootstrap_value)

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            actor_outputs, critic_outputs = self.model(np.array(observations, dtype=np.float32))
            actor_log = tf.math.log(actor_outputs)
            actions_one_hot = tf.one_hot(actions, FLAGS.num_actions, dtype=tf.float32)
            actor_output_reduced = tf.reduce_sum(actions_one_hot * actor_outputs, [1])
            policy_loss = -tf.reduce_sum(tf.math.log(actor_output_reduced) * advantages)

            critic_loss = tf.reduce_mean(
                    tf.square(tf.subtract(critic_outputs, returns), name="critic_loss"))

            entropy = - tf.reduce_sum(actor_outputs * actor_log)
            loss = tf.subtract(0.5 * critic_loss + policy_loss, 0.01 * entropy, name="loss")

        # parameters from https://github.com/muupan/async-rl/wiki
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=7e-4, epsilon=0.1, decay=0.99)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, grad_norms = tf.clip_by_global_norm(gradients, 40.0)
        optimizer.apply_gradients(zip(gradients, self.global_model.variables))
