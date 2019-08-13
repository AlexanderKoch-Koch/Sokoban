import threading
import models.fc_model
from models.conv_model import ConvModel
import actor
import tensorflow as tf
from datetime import datetime
# from tensorflow.python import debug as tf_debug
# import sokoban_env
import gym
from atari_wrapper import AtariWrapper
from absl import flags, app
FLAGS = flags.FLAGS

flags.DEFINE_float('gamma', 0.99, 'discount value')
flags.DEFINE_list('observation_shape', [80, 80, 4], 'tuple')
flags.DEFINE_integer('num_actions', 4, 'number of actions')
flags.DEFINE_integer('num_workers', 1, 'number of parallel actors')


def main(argv):
    global_model = ConvModel()
    global_model.build(input_shape=(None,) + tuple(FLAGS.observation_shape))
    log_dir = "./runs/" + datetime.now().strftime("%d.%m_%H:%M:%S")
    summary_writer = tf.summary.create_file_writer(log_dir)

    # start asynchronous actors
    actors = []
    for i in range(FLAGS.num_workers):
        name = "actor_" + str(i)
        actor_model = ConvModel()
        actor_model.build(input_shape=(None,) + tuple(FLAGS.observation_shape))
        # env = sokoban_env.SokobanEnv("./boxoban-levels/unfiltered/train/")
        env = AtariWrapper("PongDeterministic-v4")

        actors.append(actor.Actor(env, global_model, actor_model, summary_writer=summary_writer))

    # start threads
    for i in range(FLAGS.num_workers):
        thread = threading.Thread(target=actors[i].start)
        thread.start()


if __name__ == '__main__':
    app.run(main)
