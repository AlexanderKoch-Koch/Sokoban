import threading
import model
import actor
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import sokoban_env


num_workers = 4
gamma = 0.95

global_model = model.Model("global")


sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(tf.global_variables_initializer())
# start asynchronous actors
actors = []
for i in range(num_workers):
    name = "actor_" + str(i)
    actor_model = model.Model(name, observation_shape=(10, 10), num_actions=4)
    env = sokoban_env.SokobanEnv("./boxoban-levels/unfiltered/train/")
    actors.append(actor.Actor(name, sess, env, global_model, actor_model, gamma=gamma))

sess.run(tf.global_variables_initializer())

# start threads
for i in range(num_workers):
    thread = threading.Thread(target=actors[i].start)
    thread.start()
