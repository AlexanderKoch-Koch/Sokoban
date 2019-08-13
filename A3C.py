import threading
import models.fc_model
from models.conv_model import ConvNet
import actor
import tensorflow as tf
from datetime import datetime
# from tensorflow.python import debug as tf_debug
# import sokoban_env
import gym
from atari_wrapper import AtariWrapper

num_workers = 1
gamma = 0.99

global_model = ConvNet(observation_shape=(80, 80, 4), num_actions=4)
log_dir = "./runs/" + datetime.now().strftime("%d.%m_%H:%M:%S")
summary_writer = tf.summary.create_file_writer(log_dir)

# start asynchronous actors
actors = []
for i in range(num_workers):
    name = "actor_" + str(i)
    actor_model = ConvNet(observation_shape=(80, 80, 4), num_actions=4)
    # env = sokoban_env.SokobanEnv("./boxoban-levels/unfiltered/train/")
    env = AtariWrapper("PongDeterministic-v4")

    actors.append(actor.Actor(env, global_model, actor_model, gamma=gamma, summary_writer=summary_writer))

# start threads
for i in range(num_workers):
    thread = threading.Thread(target=actors[i].start)
    thread.start()
