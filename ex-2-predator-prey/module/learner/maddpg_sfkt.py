import numpy as np
import tensorflow as tf
import random


class maddpg():
    def __init__(self, args, mas_label):
        self.label = mas_label
        self.which_task = args.id_task
        self.n_tasks = args.n_tasks
        self.n_prey = 1
        self.n_agents = args.n_agents - self.n_prey
        self.n_others = self.n_agents - 1
        self.d_o = args.dim_o
        self.d_a = args.dim_a
        self.hidden_units_a = args.actor_net_h_unit
        self.n_hidden_layer_a = self.hidden_units_a.__len__()
        self.hidden_units_sfs = args.critic_net_h_unit
        self.n_hidden_layer_sfs = self.hidden_units_sfs.__len__()
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lr_r = args.lr_r
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.lr_a_2 = args.lr_a_2
        self.lr_c_2 = args.lr_c_2
        self.tau = args.tau
        self.mu = []
        self.sigma = args.explore_sigma * np.eye(self.d_a, self.d_a)
        self.dim_sfs = args.dim_phi  # * self.n_agents

        # input variables
        self.obs_t, self.act_t, self.obs_tt, self.act_tt = [], [], [], []
        for idx_agt in range(self.n_agents):
            with tf.name_scope(self.label + "agt_" + str(idx_agt)):
                self.obs_t.append(tf.placeholder(tf.float32, [None, self.d_o], "obs_t"))
                self.act_t.append(tf.placeholder(tf.float32, [None, self.d_a], "act_t"))
                self.obs_tt.append(tf.placeholder(tf.float32, [None, self.d_o], "obs_next"))
                self.act_t[idx_agt] = self.actor_net(self.obs_t[idx_agt],
                                                     self.label + "agt_" + str(idx_agt) + "/actor_net")
                self.act_tt.append(self.actor_net(self.obs_tt[idx_agt],
                                                  self.label + "agt_" + str(idx_agt) + "/actor_net_target"))

        self.reward_t = tf.placeholder(tf.float32, [None, 1], self.label + "reward_t")
        self.R_features = tf.placeholder(tf.float32, [None, self.dim_sfs], self.label + "features_t")
        # with tf.variable_scope(self.label + "reward_weights"):
        #     self.R_weight = tf.Variable(
        #         tf.random_uniform([self.dim_sfs, 1], minval=-0.1, maxval=0.1, name=self.label + "sfs_weights"))
        self.R_weight = tf.placeholder(tf.float32, [self.dim_sfs, 1], name=self.label + "reward_weights")
        self.reward = tf.einsum('ij,jk->ik', self.R_features, self.R_weight)
        # create successor networks
        self.obs_all = tf.concat(self.obs_t, axis=1)
        self.act_all = tf.concat(self.act_t, axis=1)
        self.SFs = self.sfs_net(self.obs_all, self.act_all, scope=self.label + "Current_SFs_Net")
        self.obs_all_t = tf.concat(self.obs_tt, axis=1)
        self.act_all_t = tf.concat(self.act_tt, axis=1)
        self.SFs_t = self.sfs_net(self.obs_all_t, self.act_all_t, scope=self.label + "Target_SFs_Net")
        # get Q values
        self.Q = tf.einsum('ij,jk->ik', self.SFs, self.R_weight)
        self.Q_t = tf.einsum('ij,jk->ik', self.SFs_t, self.R_weight)

        # get parameter lists
        self.a_var, self.a_tar_var = [], []
        for idx_agt in range(self.n_agents):
            self.a_var.append(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self.label + "agt_" + str(idx_agt) + "/actor_net/"))
            self.a_tar_var.append(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=self.label + "agt_" + str(idx_agt) + "/actor_net_target/"))
        self.sfs_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label + "Current_SFs_Net/")
        self.sfs_tar_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label + "Target_SFs_Net/")
        self.weight_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label + "reward_weights")

        # Loss functions and Optimizers
        self.sfs_target = tf.placeholder(tf.float32, [None, self.dim_sfs], name=self.label + "target_sfs")
        self.TD_error = self.sfs_target - self.SFs
        self.loss_a = tf.reduce_mean(-self.Q, name=self.label + "loss_a")
        self.loss_r = tf.reduce_mean(tf.square(self.reward - self.reward_t), name=self.label + "loss_r")
        self.loss_c = tf.reduce_mean(tf.reduce_sum(tf.square(self.TD_error), axis=1), name=self.label + "loss_c")
        # self.trainer_r = tf.train.AdamOptimizer(self.lr_r).minimize(self.loss_r, var_list=self.weight_var)
        self.trainer_a = [tf.train.AdamOptimizer(self.lr_a).minimize(self.loss_a, var_list=self.a_var[i]) for i in
                          range(self.n_agents)]
        self.trainer_c = tf.train.AdamOptimizer(self.lr_c).minimize(self.loss_c, var_list=self.sfs_var)
        # Phase II (mimic policy)
        self.act_t_target = [
            tf.placeholder(tf.float32, [None, self.d_a], name=self.label + "target_actions_agt_" + str(i))
            for i in range(self.n_agents)]
        self.act_error = [self.act_t[i] - self.act_t_target[i] for i in range(self.n_agents)]
        self.loss_a_2 = [
            tf.reduce_mean(tf.reduce_sum(tf.square(self.act_error[i]), axis=1), name=self.label + "loss_agt_" + str(i))
            for i in range(self.n_agents)]
        self.trainer_a_2 = [[tf.train.AdamOptimizer(self.lr_a_2).minimize(self.loss_a_2[i], var_list=self.a_var[i])
                             for i in range(self.n_agents)]]
        # Phase II (critic of mimic policy)
        self.Q_target = tf.placeholder(tf.float32, [None, 1], name=self.label + "target_q_value")
        self.TD_error_2 = self.Q_target - self.Q
        self.loss_c_2 = tf.reduce_mean(tf.square(self.TD_error_2), name=self.label + "loss_c_phase_2")
        self.trainer_c_2 = tf.train.AdamOptimizer(self.lr_c_2).minimize(self.loss_c_2, var_list=self.sfs_var)

        # tensorboard
        with tf.name_scope("losses"):
            self.loss_r_sum = tf.summary.scalar(self.label + "loss_r_sum", self.loss_r)
            # self.loss_c_sum = [tf.summary.scalar(self.label + "task_" + str(idx_task) + "_loss_c_sum",
            #                                      self.loss_c[idx_task]) for idx_task in range(self.dim_sfs)]
            self.loss_c_sum = tf.summary.scalar(self.label + "loss_c_sum", self.loss_c)
            self.loss_a_sum = tf.summary.scalar(self.label + "loss_a_sum", self.loss_a)
            self.loss_c_2_sum = tf.summary.scalar(self.label + "loss_c_phase_2_sum", self.loss_c_2)
            self.loss_a_2_sum = [tf.summary.scalar(self.label + "agt_"+str(idx_agt)+"_loss_a_phase_2_sum",
                                                   self.loss_a_2[idx_agt]) for idx_agt in range(self.n_agents)]
            self.merge_c_2 = tf.summary.merge([self.loss_c_2_sum])
            self.merge_a_2 = tf.summary.merge(self.loss_a_2_sum)
            # self.merge_r = tf.summary.merge([self.loss_r_sum])
            self.merge_c = tf.summary.merge([self.loss_c_sum])
            self.merge_a = tf.summary.merge([self.loss_a_sum])

        # soft update for target network
        self.A_update = []
        for idx_agt in range(self.n_agents):
            update = [self.a_tar_var[idx_agt][i].assign(
                tf.multiply(self.a_var[idx_agt][i], self.tau) + tf.multiply(self.a_tar_var[idx_agt][i], 1 - self.tau))
                for i in range(len(self.a_tar_var[idx_agt]))]
            self.A_update.append(update)
            for i in range(len(self.a_tar_var[idx_agt])):
                self.A_update[idx_agt][i] = tf.assign(self.a_tar_var[idx_agt][i],
                                                      tf.multiply(self.a_var[idx_agt][i], self.tau) + tf.multiply(
                                                          self.a_tar_var[idx_agt][i], 1 - self.tau))

        self.SF_update = [self.sfs_tar_var[i].assign(
            tf.multiply(self.sfs_var[i], self.tau) + tf.multiply(self.sfs_tar_var[i], 1 - self.tau)) for i in
            range(len(self.sfs_tar_var))]
        for i in range(len(self.sfs_tar_var)):
            self.SF_update[i] = tf.assign(self.sfs_tar_var[i],
                                          tf.multiply(self.sfs_var[i], self.tau) + tf.multiply(self.sfs_tar_var[i],
                                                                                               1 - self.tau))

    def actor_net(self, input_s, scope):
        with tf.variable_scope(scope):
            # hidden layers
            x_in = input_s
            for idx_layer in range(self.n_hidden_layer_a):
                layer = tf.layers.dense(
                    inputs=x_in,
                    units=self.hidden_units_a[idx_layer],
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
                    bias_initializer=tf.constant_initializer(0.0),
                    name='layer_' + str(idx_layer)
                )
                x_in = layer

            # output layer
            output_act = tf.layers.dense(
                inputs=x_in,
                units=self.d_a,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                bias_initializer=tf.constant_initializer(0.0),
                name='layer_output'
            )
        return output_act

    def sfs_net(self, input_s, input_a, scope):
        with tf.variable_scope(scope):
            x_in = tf.concat([input_s, input_a], axis=1)
            # hidden layers
            for idx_layer in range(self.n_hidden_layer_sfs):
                layer = tf.layers.dense(
                    inputs=x_in,
                    units=self.hidden_units_sfs[idx_layer],
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='layer_' + str(idx_layer)
                )
                x_in = layer

            # output layer
            output_features = tf.layers.dense(
                inputs=x_in,
                units=self.dim_sfs,
                activation=None,  # sigmoid
                kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='layer_output'
            )
            output = output_features

            ########################sublayers for subtasks#######################
            # hidden layers
            # for idx_layer in range(self.n_hidden_layer_sfs - 1):
            #     layer = tf.layers.dense(
            #         inputs=x_in,
            #         units=self.hidden_units_sfs[idx_layer],
            #         activation=tf.nn.leaky_relu,
            #         kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
            #         bias_initializer=tf.constant_initializer(0.1),
            #         name='layer_' + str(idx_layer)
            #     )
            #     x_in = layer
            # sub_layers = []
            # for idx_task in range(self.dim_sfs):
            #     sub_layers.append(tf.layers.dense(
            #         inputs=x_in,
            #         units=self.hidden_units_sfs[-1],
            #         activation=tf.nn.leaky_relu,
            #         kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
            #         bias_initializer=tf.constant_initializer(0.1),
            #         name='task_' + str(idx_task) + '_layer_' + str(-1)
            #     ))
            # # output layer
            # output_layer = []
            # for idx_task in range(self.dim_sfs):
            #     output_layer.append(tf.layers.dense(
            #         inputs=sub_layers[idx_task],
            #         units=1,
            #         activation=None,
            #         kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
            #         bias_initializer=tf.constant_initializer(0.1),
            #         name='layer_output_task_' + str(idx_task)
            #     ))
            # output = tf.concat(output_layer, axis=1)

            # sub critic networks
            # sub_networks = []
            # for idx_task in range(self.dim_sfs):
            #     # hidden layers
            #     x_in = tf.concat([input_s, input_a], axis=1)
            #     for idx_layer in range(self.n_hidden_layer_sfs):
            #         layer = tf.layers.dense(
            #             inputs=x_in,
            #             units=self.hidden_units_sfs[idx_layer],
            #             activation=tf.nn.leaky_relu,
            #             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
            #             bias_initializer=tf.constant_initializer(0.1),
            #             name='task_'+ str(idx_task) +'_layer_' + str(idx_layer)
            #         )
            #         x_in = layer
            #
            #     # output layer
            #     output_features = tf.layers.dense(
            #         inputs=x_in,
            #         units=1,
            #         activation=None,
            #         kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
            #         bias_initializer=tf.constant_initializer(0.1),
            #         name='task_'+ str(idx_task) + '_layer_output'
            #     )
            #     sub_networks.append(output_features)
            # output = tf.concat(sub_networks, axis=1)

        return output

    # def reward_features(self, positions):
    #     x_in = positions  # n_agt * 2
    #     distances = np.zeros([self.n_agents, self.n_centers])
    #     for idx_agt in range(self.n_agents):
    #         for i in range(self.n_centers):
    #             distances[idx_agt][i] = np.sqrt(np.sum(np.square(self.center[i] - x_in[idx_agt])))
    #     # phi_s = distances.reshape([-1])
    #     phi_s = np.min(distances, axis=0, keepdims=False)
    #     return phi_s

    def update_target_net(self, sess, init=False):
        for idx_agt in range(self.n_agents):
            sess.run(self.A_update[idx_agt])
        sess.run(self.SF_update)

        if init:
            for i in range(len(self.sfs_tar_var)):
                sess.run(tf.assign(self.sfs_tar_var[i], self.sfs_var[i]))
            for idx_agt in range(self.n_agents):
                for i in range(len(self.a_tar_var[idx_agt])):
                    sess.run(tf.assign(self.a_tar_var[idx_agt][i], self.a_var[idx_agt][i]))

    def get_actions(self, obs, sess, noise=False):
        action_t = []
        for idx_agt in range(self.n_agents):
            action_t.append(self.act_t[idx_agt].eval(feed_dict={self.obs_t[idx_agt]: [obs[idx_agt]]}, session=sess)[0])
            if noise:
                self.mu = action_t[idx_agt]
                for i in range(self.d_a):
                    action_t[idx_agt][i] = action_t[idx_agt][i] + np.random.normal(0, self.sigma[i][i])

        return action_t

    def get_target_sfs(self, features, obs_next, sess):
        feed_obs_next = {self.obs_tt[idx_agt]: obs_next[:, idx_agt, :] for idx_agt in range(self.n_agents)}
        sfs_next = self.SFs_t.eval(feed_dict=feed_obs_next,
                                   session=sess)
        sfs_predict = features + self.gamma * sfs_next
        return sfs_predict

    def get_target_q(self, features, obs_next, task_weight, sess):
        feed_obs_next = {self.obs_tt[idx_agt]: obs_next[:, idx_agt, :] for idx_agt in range(self.n_agents)}
        feed_q = {}
        feed_q.update(feed_obs_next)
        feed_q.update({self.R_weight: task_weight.reshape([-1, 1])})
        q_next = self.Q_t.eval(feed_dict=feed_q, session=sess)
        r = np.dot(features, task_weight).reshape([-1, 1])
        q_predict = r + self.gamma * q_next
        return q_predict
