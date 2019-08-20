import threading
import actor
import tensorflow as tf
from datetime import datetime
# import sokoban_env
import gym
from models.fc_model import FCModel
from models.conv_model import ConvModel
from atari_wrapper import AtariWrapper
from absl import flags, app
FLAGS = flags.FLAGS

flags.DEFINE_float('gamma', 0.90, 'discount value')
flags.DEFINE_list('observation_shape', [4], 'tuple')
flags.DEFINE_integer('num_actions', 2, 'number of actions')
flags.DEFINE_integer('num_workers', 1, 'number of parallel actors')
flags.DEFINE_integer('act_steps_max', 5, 'max steps before bootstrapping')

def main(argv):
    global_model = FCModel()
    global_model.build(input_shape=(None,) + tuple(FLAGS.observation_shape))
    print(global_model.summary())
    log_dir = "./runs/" + datetime.now().strftime("%d.%m_%H:%M:%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    
    # create actor objects
    actors = []
    for i in range(FLAGS.num_workers):
        # env = sokoban_env.SokobanEnv("./boxoban-levels/unfiltered/train/")
        # env = AtariWrapper("PongDeterministic-v4")
        env = gym.make('CartPole-v1')

        actor_model = FCModel()
        actor_model.build(input_shape=(None,) + tuple(FLAGS.observation_shape))
        log = True if i==0 else False
        actors.append(actor.Actor(env, global_model, actor_model, summary_writer=summary_writer, log=log))

    # start threads
    for i in range(FLAGS.num_workers):
        thread = threading.Thread(target=actors[i].start)
        thread.start()


if __name__ == '__main__':
    app.run(main)
