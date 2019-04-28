import threading
import fc_model
from conv_model import ConvModel
import actor
import tensorflow as tf
# from tensorflow.python import debug as tf_debug
# import sokoban_env
import gym
from atari_wrapper import AtariWrapper
num_workers = 1
gamma = 0.99

global_model = ConvModel("global", observation_shape=(80, 80, 4), num_actions=4)


sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(tf.global_variables_initializer())
# start asynchronous actors
actors = []
for i in range(num_workers):
    name = "actor_" + str(i)
    actor_model = ConvModel(name, observation_shape=(80, 80, 4), num_actions=4)
    # env = sokoban_env.SokobanEnv("./boxoban-levels/unfiltered/train/")
    env = AtariWrapper("PongDeterministic-v4")
    actors.append(actor.Actor(name, sess, env, global_model, actor_model, gamma=gamma))

sess.run(tf.global_variables_initializer())

# start threads
for i in range(num_workers):
    thread = threading.Thread(target=actors[i].start)
    thread.start()
