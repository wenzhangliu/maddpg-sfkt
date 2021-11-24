import numpy as np
import tensorflow as tf
import random


class agent_model():
    def __init__(self, args, mas_label):
        self.label = mas_label
        self.which_task = args.id_task
        self.n_tasks = args.n_tasks
        self.n_agents = args.n_agents
        self.n_others = self.n_agents - 1
        self.d_w = args.dim_phi
        self.d_o = args.dim_o
        self.d_a = args.dim_a
        self.hidden_units_a = args.actor_net_h_unit
        self.n_hidden_layer_a = self.hidden_units_a.__len__()
        self.hidden_units_o = [256, 256]
        self.hidden_units_w = [128]  # 256
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lr_r = args.lr_r
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.tau = args.tau
        self.mu = []
        self.sigma = args.explore_sigma * np.eye(self.d_a, self.d_a)
        self.dim_sfs = args.dim_phi  # * self.n_agents
        self.n_related_tasks = args.n_related_task
        self.n_extend_task = self.n_related_tasks + 1

        self.epsilon = 0.9  # 0.0, 0.9
        self.alpha = 0.9  # 0.3, 0.6
        self.delta_alpha = float(0.7 / 250000)
        self.delta_epsilon = float(0.95 / 250000)

        # input variables
        self.obs_t = tf.placeholder(tf.float32, [None, self.d_o], "obs_t")
        self.act_t = tf.placeholder(tf.float32, [None, self.d_a], "act_t")
        self.target_task = tf.placeholder(tf.float32, [None, self.d_w], "target_task")
        self.obs_tt = tf.placeholder(tf.float32, [None, self.d_o], "obs_next")

        self.tasks_k = tf.placeholder(tf.float32, [None, self.d_w], "related_tasks_C2")
        self.tasks_k_tt = tf.placeholder(tf.float32, [None, self.d_w], "related_tasks_C2_next")

        # task embedding
        self.act_tt, self.tasks_z, self.tasks_z_tt = [], [], []
        self.mausfs, self.mausfs_target, self.mausfs_a, self.mausfs_target_a = [], [], [], []
        self.Q, self.Q_next = [], []
        for i_task in range(self.n_related_tasks+1):
            self.act_tt.append(tf.placeholder(tf.float32, [None, self.d_a], "act_next_"+str(i_task)))
            self.tasks_z.append(tf.placeholder(tf.float32, [None, self.d_w], "related_tasks_C1_"+str(i_task)))
            self.tasks_z_tt.append(tf.placeholder(tf.float32, [None, self.d_w], "related_tasks_C1_next_"+str(i_task)))
            self.mausfs.append(self.usfs(input_s=self.obs_t, input_w=self.tasks_z[i_task],
                                         reuse=tf.AUTO_REUSE,
                                         scope=self.label + "Current_SF_Net"))
            self.mausfs_target.append(self.usfs(input_s=self.obs_tt, input_w=self.tasks_z_tt[i_task],
                                                reuse=tf.AUTO_REUSE,
                                                scope=self.label + "Target_SF_Net"))

            self.mausfs_a.append(tf.einsum("ijk,ik->ij", self.mausfs[i_task], self.act_t))
            self.mausfs_target_a.append(tf.einsum("ijk,ik->ij", self.mausfs_target[i_task], self.act_tt[i_task]))

            self.Q.append(tf.einsum("ijk,ij->ik", self.mausfs[i_task], self.tasks_k))
            self.Q_next.append(tf.einsum("ijk,ij->ik", self.mausfs_target[i_task], self.tasks_k_tt))

        # get parameter lists
        self.sfs_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label + "Current_SF_Net")
        self.sfs_tar_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label + "Target_SF_Net")

    def usfs(self, input_s, input_w, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            x_in = input_s
            w_in = input_w
            # observation embedding: MLP with 2 hidden FC.
            for idx_layer in range(len(self.hidden_units_o)):
                layer_x = tf.layers.dense(
                    inputs=x_in,
                    units=self.hidden_units_o[idx_layer],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='layer_observation_' + str(idx_layer)
                )
                x_in = layer_x
            # task embedding layer: MLP with 1 hidden FC.
            for idx_layer in range(len(self.hidden_units_w)):
                layer_w = tf.layers.dense(
                    inputs=w_in,
                    units=self.hidden_units_w[idx_layer],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='layer_weight_' + str(idx_layer)
                )
                w_in = layer_w
            # concat layer: MLP with 1 hidden FC.
            input_joint = tf.concat([x_in, w_in], axis=-1)
            layer_out = tf.layers.dense(
                inputs=input_joint,
                units=128,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
                bias_initializer=tf.constant_initializer(0.1),
                name="layer_joint"
            )
            # output layer
            output_layer = tf.layers.dense(
                inputs=layer_out,
                units=self.d_a * self.d_w,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
                bias_initializer=tf.constant_initializer(0.1),
                name="layer_output"
            )
            output = tf.reshape(output_layer, [-1, self.d_w, self.d_a])
            # return x_in, w_in, output
            return output

    def get_actions(self, obs, target_task, related_tasks, sess, noise=False):
        """
        get actions by UneVEn policy.
        inputs:
            obs: the observations of the agent
            target_task: the weights of the target task, size: d_phi * 1
            related_tasks: the weights of the related tasks set v of the target task, size: |v| * d_phi
            sess: tf.session()
            noise: greedy policy or not
        """
        if noise:
            epsilon = self.epsilon
            alpha = self.alpha
        else:
            epsilon = 2.0
            alpha = 2.0

        C_1 = np.concatenate([related_tasks, target_task.reshape(1, -1)], axis=0)
        p_select = np.random.random()
        task_select = np.random.random()
        if p_select >= epsilon:  # Uniform(Action space)
            act = random.randint(0, self.d_a - 1)
        else:  # select actions based on UneVEn policy.
            if task_select >= alpha:  # select actions based on target task, with probability of alpha, C_2 = {w}
                C_2 = [target_task]
            else:  # C_2 = {w} + v, with probability of 1-alpha
                C_2 = C_1

            q_gpi = np.zeros([len(C_1), len(C_2), self.d_a])
            feed_task = {self.obs_t: [obs]}
            feed_task.update({self.tasks_z[i_task]: [C_1[i_task]] for i_task in range(self.n_related_tasks+1)})
            for i_task_k in range(len(C_2)):
                feed_task.update({self.tasks_k: [C_2[i_task_k]]})
                q = sess.run(self.Q, feed_dict=feed_task)
                q_gpi[:, i_task_k, :] = q
            act = np.where(q_gpi == np.max(q_gpi))[-1][0]

        return act

    def get_actions_next(self, obs_next, related_tasks_all, sess):
        actions_next = []
        for i_task in range(self.n_related_tasks+1):
            q_next = self.Q_next[-1].eval(feed_dict={self.obs_tt: obs_next,
                                                     self.tasks_z_tt[-1]: related_tasks_all[:, i_task, :],
                                                     self.tasks_k_tt: related_tasks_all[:, i_task, :]},  session=sess)
            actions_next.append(np.argmax(q_next, axis=-1))

        return actions_next
