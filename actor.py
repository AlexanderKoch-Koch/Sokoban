import tensorflow as tf
import numpy as np
from trajectory import Trajectory
from observation_preprocessing import preprocess_atari
import matplotlib.pyplot as plt
import time
from absl import flags
FLAGS = flags.FLAGS


class Actor:
    def __init__(self, env, global_model, local_model, summary_writer, log=False):
        self.global_model = global_model
        self.env = env
        self.model = local_model
        self.summary_writer = summary_writer
        self.log = log

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
                    for t in range(FLAGS.act_steps_max):
                        model_input = np.expand_dims(observation, axis=0)
                        pi, v = self.model(model_input)
                        pi = np.array(pi[0]); v = v[0]
                        tf.summary.scalar('state value', v[0], step=train_step*5+t)
                        tf.summary.histogram('policy', pi, step=train_step*5+t)
                        action = np.random.choice(range(FLAGS.num_actions), p=pi)
                        observation_next, reward, done, _ = self.env.step(action)

                        # self.env.render()
                        reward_sum += reward
                        trajectory.add_step(observation, action, v, reward)
                        observation = np.copy(observation_next)
                        
                        if done:
                            break
                    
                    if done:
                        #breakpoint()
                        bootstrap_state_value = 0
                    else:
                        bootstrap_state_value = v

                    self.train(trajectory, bootstrap_state_value, train_step)
                    train_step += 1
                if self.log:
                    tf.summary.scalar("reward_per_episode", reward_sum, step=episode)

    def train(self, trajectory, bootstrap_value, training_step):
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
        #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, grad_norms = tf.clip_by_global_norm(gradients, 40.0)
        optimizer.apply_gradients(zip(gradients, self.global_model.variables))
        if self.log:
            tf.summary.scalar('policy_loss', policy_loss, training_step)
            tf.summary.scalar('critic_loss', critic_loss, training_step)
            tf.summary.scalar('entropy', entropy, training_step)
        
        return policy_loss, critic_loss, entropy, loss
