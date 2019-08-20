import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import numpy as np
from .model import Model
from absl import flags
FLAGS = flags.FLAGS


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
