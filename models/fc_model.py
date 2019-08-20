import tensorflow as tf
from tensorflow.keras import layers
from .model import Model

from absl import flags
FLAGS = flags.FLAGS

class FCModel(tf.keras.Model):
    def __init__(self):
        super(FCModel, self).__init__()
        self.dense1 = layers.Dense(32, input_shape=FLAGS.observation_shape,
                            activation="relu", name='dense_1')
        self.dense2 = layers.Dense(16, activation="relu", name='dense_2') 
        self.dense3 = layers.Dense(8, activation="relu", name='dense_3')
        self.actor_dense = layers.Dense(FLAGS.num_actions, activation="softmax", name='policy_dense_1')
        self.critic_dense = layers.Dense(1, activation="linear", name='critic_dense_1')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        actor_output = self.actor_dense(x)
        critic_output = self.critic_dense(x)

        return actor_output, critic_output

