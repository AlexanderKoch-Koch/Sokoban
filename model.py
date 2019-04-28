import tensorflow as tf


class Model:

    def __init__(self, scope, observation_shape=(10, 10)):
        self.tf_input = tf.placeholder(dtype=tf.float32, shape=(None,) + observation_shape, name="observation")
        self.tf_advantage = tf.placeholder(dtype=tf.float32, shape=(None,), name="advantage")
        self.tf_action = tf.placeholder(dtype=tf.int32, shape=(None,), name="action")
        self.tf_critic_y = tf.placeholder(dtype=tf.float32, shape=(None,), name="true_state_value")

        self.tf_actor_output = None
        self.tf_critic_output = None
        self.tf_apply_grads = None
        self.tf_summary = None

    def set_model_outputs(self, tf_actor_output, tf_critic_output, tf_apply_grads, tf_summary):
        self.tf_actor_output = tf_actor_output
        self.tf_critic_output = tf_critic_output
        self.tf_apply_grads = tf_apply_grads
        self.tf_summary = tf_summary

    def predict(self, tf_session, observation):
        observation = [observation]
        policy_output, value_output = tf_session.run([self.tf_actor_output, self.tf_critic_output],
                                                     feed_dict={self.tf_input: observation})
        return policy_output[0], value_output

    def train(self, tf_session, trajectory, step, summary_writer=None):
        observations, actions, advantages, returns = trajectory.get_training_batch()
        feed_dict = {
            self.tf_input: observations,
            self.tf_action: actions,
            self.tf_advantage: advantages,
            self.tf_critic_y: returns
        }
        _, summaries = tf_session.run([self.tf_apply_grads, self.tf_summary], feed_dict=feed_dict)

        if summary_writer is not None:
            summary_writer.add_summary(summaries, step)

    def copy_global_weights(self, tf_session):
        tf_session.run(self.copy_ops)
