import tensorflow as tf
from .model import Model


class ConvModel(Model):
    def __init__(self, scope, observation_shape=(10, 10), num_actions=4, debug=True):
        print("constructing model with scope: " + scope)
        with tf.variable_scope(scope):
            super().__init__(scope, observation_shape=observation_shape)

            with tf.name_scope("feature_extraction"):
                tf_l1 = tf.keras.layers.Conv2D(16, (8, 8), strides=(4, 4), activation="relu")(self.tf_input)
                tf_l2 = tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), activation="relu")(tf_l1)
                tf_flattened = tf.keras.layers.Flatten()(tf_l2)
                self.tf_features = tf.keras.layers.Dense(256, activation="relu")(tf_flattened)

            # Actor head
            with tf.name_scope("policy_head"):
                tf_actor_w3 = tf.Variable(tf.truncated_normal(shape=(256, num_actions), stddev=0.1))
                tf_actor_b3 = tf.Variable(tf.truncated_normal(shape=(num_actions,), stddev=0.1))
                self.tf_actor_output = tf.nn.softmax(tf.add(tf.matmul(self.tf_features, tf_actor_w3), tf_actor_b3))
                tf_actor_log = tf.log(self.tf_actor_output)
                tf_action_one_hot = tf.one_hot(self.tf_action, num_actions, dtype=tf.float32)
                tf_actor_output_reduced = tf.reduce_sum(tf_action_one_hot * self.tf_actor_output, [1])
                self.tf_policy_loss = -tf.reduce_sum(tf.log(tf_actor_output_reduced) * self.tf_advantage)

            # Critic head
            with tf.name_scope("value_head"):
                tf_critic_w3 = tf.Variable(tf.truncated_normal(shape=(256, 1), stddev=0.1))
                tf_critic_b3 = tf.Variable(tf.truncated_normal(shape=(1,), stddev=0.1))
                self.tf_critic_output = tf.squeeze(tf.add(tf.matmul(self.tf_features, tf_critic_w3), tf_critic_b3),
                                                   name="state_value_prediction")

                self.tf_critic_loss = tf.reduce_mean(
                    tf.square(tf.subtract(self.tf_critic_output, self.tf_critic_y), name="critic_loss"))

            self.entropy = - tf.reduce_sum(self.tf_actor_output * tf_actor_log)
            self.loss = tf.subtract(0.5 * self.tf_critic_loss + self.tf_policy_loss, 0.01 * self.entropy, name="loss")

            # parameters from https://github.com/muupan/async-rl/wiki
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.local_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.global_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global")

            self.copy_ops = list()
            for source_var, target_var in zip(self.global_variables, self.local_variables):
                self.copy_ops.append(target_var.assign(source_var))

            self.gradients, self.grad_norms = tf.clip_by_global_norm(tf.gradients(self.loss, self.local_variables),
                                                                     40.0)
            self.tf_apply_grads = self.optimizer.apply_gradients(zip(self.gradients, self.global_variables))

            if debug:
                # tf.summary.histogram("actor_output", self.tf_actor_output)
                tf.summary.scalar("actor loss", self.tf_policy_loss)
                # tf.summary.histogram("critic_b3", tf_critic_b3)
                tf.summary.histogram("critic_output", self.tf_critic_output)
                tf.summary.histogram("critic_y", self.tf_critic_y)
                tf.summary.scalar("critic_loss", self.tf_critic_loss)
                tf.summary.scalar("policy_entropy", self.entropy)

            self.tf_merged_summaries = tf.summary.merge_all(scope=scope)

            super().set_model_outputs(self.tf_actor_output, self.tf_critic_output, self.tf_apply_grads, self.tf_merged_summaries)
