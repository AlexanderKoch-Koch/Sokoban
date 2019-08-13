import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import numpy as np
from .model import Model
from absl import flags
FLAGS = flags.FLAGS

class ConvNet(Model):
    def __init__(self, observation_shape=(10, 10), num_actions=4):
        self.model = ConvModel(observation_shape, num_actions)
        self.model.build(input_shape=(None,) + observation_shape)
        self.num_actions = num_actions

    def update_weights(self, global_model):
        self.model.set_weights(global_model.get_weights())

    def get_weights(self):
        return self.model.get_weights()

    def train(self, trajectory, global_network, steps):
        observations, actions, advantages, returns = trajectory.get_training_batch()
        with tf.GradientTape() as tape:
            actor_outputs, critic_outputs = self.predict_batch(observations)
            actor_log = tf.math.log(actor_outputs)
            actions_one_hot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
            actor_output_reduced = tf.reduce_sum(actions_one_hot * actor_outputs, [1])
            policy_loss = -tf.reduce_sum(tf.math.log(actor_output_reduced) * advantages)

            critic_loss = tf.reduce_mean(
                    tf.square(tf.subtract(critic_outputs, returns), name="critic_loss"))

            entropy = - tf.reduce_sum(actor_outputs * actor_log)
            loss = tf.subtract(0.5 * critic_loss + policy_loss, 0.01 * entropy, name="loss")

        # parameters from https://github.com/muupan/async-rl/wiki
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=7e-4, epsilon=0.1, decay=0.99)
        gradients = tape.gradient(loss, self.model.variables)
        gradients, grad_norms = tf.clip_by_global_norm(gradients, 40.0)
        optimizer.apply_gradients(zip(gradients, global_network.model.variables))
        

    def loss(y_true, y_pred):
        breakpoint()

    def predict(self, observation):
        """
        observation: 3d array with observation
        """
        observation = np.array(observation, dtype=np.float32)
        if observation.ndim == 3:
            # add dimension if ony single sample to create batch for keras model
            observation = np.expand_dims(observation, axis=0)
        
        assert observation.ndim == 4
        # return prediction of first sample in batch
        actor_output, critic_output = self.model(observation)
        return np.array(actor_output[0]), np.array(critic_output[0][0])

    def predict_batch(self, observations):
        """
        observation: list of 3d observations
        """
        observations = np.array(observations, dtype=np.float32)
        assert observations.ndim == 4
        actor_output, critic_output = self.model(observations)
        return actor_output, critic_output


class ConvModel(tf.keras.Model):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = layers.Conv2D(16, (8, 8), strides=(4, 4), input_shape=FLAGS.observation_shape, activation="relu")
        self.conv2 = layers.Conv2D(32, (4, 4), strides=(2, 2), activation="relu")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation="relu")

        self.actor_dense = layers.Dense(FLAGS.num_actions, activation="softmax")
        self.critic_dense = layers.Dense(1, activation="linear")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)

        actor_output = self.actor_dense(x)
        critic_output = self.critic_dense(x)

        return actor_output, critic_output
