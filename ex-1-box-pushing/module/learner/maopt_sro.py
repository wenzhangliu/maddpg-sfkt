import numpy as np
import tensorflow as tf
import random


class agent_model():
    def __init__(self, args, mas_label):
        self.label = mas_label
        self.d_o = args.dim_o
        self.d_a = args.dim_a
        self.n_agents = args.n_agents
        self.n_others = self.n_agents - 1
        self.d_a_others = self.n_others * self.d_a
        self.gamma = args.gamma
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.tau = args.tau
        self.dim_units_a = args.actor_net_h_unit
        self.dim_units_c = args.critic_net_h_unit

        # exploration noise
        self.mu = []
        self.sigma = args.explore_sigma * np.eye(self.d_a, self.d_a)

        # Input and Output For Actor Network
        self.obs_t = tf.placeholder(tf.float32, [None, self.d_o], name=self.label + "obs_t")
        self.obs_tt = tf.placeholder(tf.float32, [None, self.d_o], name=self.label + "obs_next")
        self.act_t = tf.placeholder(tf.float32, [None, self.d_a], name=self.label + "act_t")
        self.act_t = self.actor_net(obs_in=self.obs_t, scope=self.label + "actor_net")
        self.act_tt = self.actor_net(obs_in=self.obs_tt, scope=self.label + "actor_target_net")
        self.obs_others = tf.placeholder(tf.float32, [None, self.d_o * self.n_others],
                                         name=self.label + "obs_t_others")
        self.obs_next_others = tf.placeholder(tf.float32, [None, self.d_o * self.n_others],
                                              name=self.label + "obs_next_others")
        self.act_others = tf.placeholder(tf.float32, [None, self.d_a * self.n_others],
                                         name=self.label + "act_t_others")
        self.act_next_others = tf.placeholder(tf.float32, [None, self.d_a * self.n_others],
                                              name=self.label + "act_next_others")
        # Input And Output for Critic Network
        self.Q = self.critic_net(obs_in=self.obs_t, act_in=self.act_t,
                                 obs_others_in=self.obs_others, act_others_in=self.act_others,
                                 scope=self.label + "critic_net")
        self.Q_t = self.critic_net(obs_in=self.obs_tt, act_in=self.act_tt,
                                   obs_others_in=self.obs_next_others, act_others_in=self.act_next_others,
                                   scope=self.label + "critic_target_net")
        self.Q_target = tf.placeholder(tf.float32, [None, 1], name=self.label + "q_predict")

        # Parameter Collection
        self.a_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label + "actor_net")
        self.c_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label + "critic_net")
        self.a_tar_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label + "actor_target_net")
        self.c_tar_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label + "critic_target_net")

        # loss function and Optimizer
        self.TD_error = self.Q - self.Q_target
        self.loss_c = tf.reduce_mean(tf.square(self.TD_error))
        self.option_adv = tf.placeholder(tf.float32, [None, self.d_a], name=self.label + "option_advisor")
        self.f_of_t = tf.placeholder(tf.float32, [], name=self.label + "transfer_weights")
        self.loss_transfer = self.f_of_t * tf.reduce_mean(tf.square(self.option_adv - self.act_t))
        self.loss_a = tf.reduce_mean(-self.Q) + self.loss_transfer
        self.trainer_c = tf.train.AdamOptimizer(self.lr_c).minimize(self.loss_c, var_list=self.c_var)
        self.trainer_a = tf.train.AdamOptimizer(self.lr_a).minimize(self.loss_a, var_list=self.a_var)

        # soft update for target network
        self.soft_update_a = [self.a_tar_var[i].assign(
            tf.multiply(self.a_var[i], self.tau) + tf.multiply(self.a_tar_var[i], 1 - self.tau)) for i in
            range(len(self.a_tar_var))]
        for i in range(len(self.a_tar_var)):
            self.soft_update_a[i] = tf.assign(self.a_tar_var[i],
                                              tf.multiply(self.a_var[i], self.tau) + tf.multiply(self.a_tar_var[i],
                                                                                                   1 - self.tau))
        self.soft_update_c = [self.c_tar_var[i].assign(
            tf.multiply(self.c_var[i], self.tau) + tf.multiply(self.c_tar_var[i], 1 - self.tau)) for i in
            range(len(self.c_tar_var))]
        for i in range(len(self.c_tar_var)):
            self.soft_update_c[i] = tf.assign(self.c_tar_var[i],
                                              tf.multiply(self.c_var[i], self.tau) + tf.multiply(self.c_tar_var[i],
                                                                                                   1 - self.tau))

    def actor_net(self, obs_in, scope):
        with tf.variable_scope(scope):
            # hidden layers
            x_in = obs_in
            for idx_layer in range(self.dim_units_a.__len__()):
                layer = tf.layers.dense(
                    inputs=x_in,
                    units=self.dim_units_a[idx_layer],
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

    def critic_net(self, obs_in, act_in, obs_others_in, act_others_in, scope):
        with tf.variable_scope(scope):
            # hidden layers
            x_in = tf.concat([obs_in, act_in, obs_others_in, act_others_in], axis=1)
            for idx_layer in range(self.dim_units_c.__len__()):
                layer = tf.layers.dense(
                    inputs=x_in,
                    units=self.dim_units_c[idx_layer],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='layer_' + str(idx_layer)
                )
                x_in = layer

            # output layer
            output_q = tf.layers.dense(
                inputs=x_in,
                units=1,
                activation=None,
                kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='layer_output'
            )

            return output_q

    def update_target_net(self, sess, init=False):
        sess.run(self.soft_update_a)
        sess.run(self.soft_update_c)

        if init:
            for i in range(len(self.c_tar_var)):
                sess.run(tf.assign(self.c_tar_var[i], self.c_var[i]))
            for i in range(len(self.a_tar_var)):
                sess.run(tf.assign(self.a_tar_var[i], self.a_var[i]))

    def get_actions(self, obs, sess, noise=False):
        action_t = self.act_t.eval(feed_dict={self.obs_t: [obs]}, session=sess)
        if noise:
            self.mu = action_t
            for i in range(self.d_a):
                action_t[:, i] = action_t[:, i] + np.random.normal(0, self.sigma[i][i])

        return action_t

    def get_q_values(self, obs, obs_others, act, act_others, sess):
        return self.Q.eval(feed_dict={self.obs_t: obs,
                                      self.obs_others: obs_others,
                                      self.act_t: act,
                                      self.act_others: act_others},
                           session=sess)

    def get_q_predict(self, r, obs_next, obs_next_others, act_next_others, sess):
        q_next = self.Q_t.eval(feed_dict={self.obs_tt: obs_next,
                                          self.obs_next_others: obs_next_others,
                                          self.act_next_others: act_next_others},
                               session=sess)
        q_predict = r + self.gamma * q_next

        return q_predict


class option_model_SRO():
    def __init__(self, args, sro_label):
        self.label = sro_label
        self.d_o = args.dim_o
        self.d_a = args.dim_a
        self.n_agents = args.n_agents
        self.n_options = self.n_agents
        self.gamma = args.gamma
        self.lr_obs = 1e-5  # phi
        self.lr_w = 1e-5
        self.lr_sr = 1e-5
        self.lr_beta = 1e-5
        self.tau = args.tau
        self.dim_weights = self.n_options
        self.xi = 0.01  # 0.01
        self.epsilon_start = 0.0
        self.epsilon = self.epsilon_start  # 0.9
        self.epsilon_finish = 0.95
        self.epsilon_annel_time = 5e4  # 5e4 (833episodes)
        self.batch_size = 32

        self.obs = tf.placeholder(tf.float32, [None, self.d_o], name=self.label + "obs_t")
        self.obs_next = tf.placeholder(tf.float32, [None, self.d_o], name=self.label + "obs_next")
        self.reward = tf.placeholder(tf.float32, [None, 1], name=self.label + "reward")
        self.option = tf.placeholder(tf.float32, [None, self.n_options], name=self.label + "option")

        self.phi, self.param_phi = self.build_nets(input=self.obs, layer_units=[64, 32, self.dim_weights],
                                                   scope=self.label + "State_Feature")
        self.obs_decode, self.param_obs = self.build_nets(input=self.phi, layer_units=[64, self.d_o],
                                                          scope=self.label + "State_Reconstruction")
        self.r_weight = tf.Variable(tf.random_uniform([self.dim_weights, 1]), name="Reward_Weights")
        self.sr_features, self.param_sr = self.build_nets(input=self.phi, layer_units=[64, 32 * self.dim_weights,
                                                                                       self.dim_weights ** 2],
                                                          scope=self.label + "SR_Network")
        self.sr_next, self.param_sr_next = self.build_nets(input=self.phi,
                                                           layer_units=[64, 32 * self.dim_weights,
                                                                        self.dim_weights ** 2],
                                                           scope=self.label + "SR_Target_Network")
        self.beta, self.param_beta = self.build_nets(input=self.phi,
                                                     layer_units=[64, self.dim_weights, self.dim_weights],
                                                     scope=self.label + "Termination_Network")
        self.beta = tf.nn.sigmoid(self.beta)
        # get the * corresponding to the option
        self.sr_features = tf.reshape(self.sr_features, [-1, self.dim_weights, self.dim_weights])
        self.sr_features_opt = tf.einsum("ijk,ij->ik", self.sr_features, self.option)
        self.sr_next = tf.reshape(self.sr_next, [-1, self.dim_weights, self.dim_weights])
        self.sr_next_opt = tf.einsum("ijk,ij->ik", self.sr_next, self.option)
        self.beta_opt = tf.reshape(tf.einsum("ij,ij->i", self.beta, self.option), [-1, 1])
        # self.Q_option = tf.reshape(tf.matmul(self.sr_features, self.r_weight), [-1, self.n_options], name="Q_Options")
        # self.Q_next = tf.reshape(tf.matmul(self.sr_next, self.r_weight), [-1, self.n_options], name="Q_Next_Options")
        self.Q_option = tf.reshape(tf.einsum("ijk,kl->ijl", self.sr_features, self.r_weight), [-1, self.n_options], name="Q_Options")
        self.Q_next = tf.reshape(tf.einsum("ijk,kl->ijl", self.sr_next, self.r_weight), [-1, self.n_options], name="Q_Next_Options")

        # obs feature, obs reconstruction, weight losses.
        self.loss_obs = tf.reduce_mean(tf.square(self.obs_decode - self.obs))

        self.reward_eval = tf.matmul(self.phi, self.r_weight)
        self.loss_weight = tf.reduce_mean(tf.square(self.reward - self.reward_eval))

        # option loss.

        self.Q_next_argmax = tf.cast(tf.argmax(self.Q_next, axis=1), dtype=tf.int32)
        opt_indices = tf.stack([tf.range(tf.shape(self.Q_next_argmax)[0], dtype=tf.int32), self.Q_next_argmax], axis=1)
        self.sr_next_selected = tf.gather_nd(params=self.sr_next, indices=opt_indices)
        self.U_next = (1 - self.beta_opt) * self.sr_next_opt + self.beta_opt * self.sr_next_selected
        self.U_target = tf.placeholder(tf.float32, [None, self.dim_weights], name=self.label + "target_U")
        self.loss_option = tf.reduce_mean(tf.square(self.sr_features_opt - self.U_target))

        # termination loss.
        self.Q_option_next = tf.reshape(tf.einsum("ij,ij->i", self.Q_next, self.option), [-1, 1])
        self.Q_option_next_max = tf.reduce_max(self.Q_next, axis=1, keepdims=True)
        self.advantage = tf.placeholder(tf.float32, [None, 1], name=self.label + "Advantage")
        self.loss_beta = tf.reduce_sum(self.advantage * self.beta_opt)

        # trainers.
        self.trainer_obs = tf.train.AdamOptimizer(self.lr_obs).minimize(self.loss_obs,
                                                                        var_list=self.param_obs + self.param_phi)
        self.trainer_weight = tf.train.AdamOptimizer(self.lr_w).minimize(self.loss_weight,
                                                                         var_list=[self.r_weight] + self.param_phi)
        self.trainer_option = tf.train.AdamOptimizer(self.lr_sr).minimize(self.loss_option,
                                                                          var_list=self.param_sr)
        self.trainer_beta = tf.train.AdamOptimizer(self.lr_beta).minimize(self.loss_beta, var_list=self.param_beta)

        # target network copy
        self.soft_update_sr = [self.param_sr_next[i].assign(self.param_sr[i]) for i in range(len(self.param_sr_next))]
        for i in range(len(self.param_sr_next)):
            self.soft_update_sr[i] = tf.assign(self.param_sr_next[i], self.param_sr[i])

        return

    def build_nets(self, input, layer_units, scope):
        with tf.variable_scope(scope):
            x_in = input
            for idx_hidden in range(len(layer_units) - 1):
                h_layer = tf.layers.dense(inputs=x_in, units=layer_units[idx_hidden], activation=tf.nn.relu)
                x_in = h_layer
            output = tf.layers.dense(inputs=x_in, units=layer_units[-1], activation=None, name="output_layer")
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return output, params

    def update_target_sr(self, sess):
        sess.run(self.soft_update_sr)

    def get_target_SR(self, obs_t, obs_next, option_t, sess):
        phi_t = self.phi.eval(feed_dict={self.obs: obs_t}, session=sess)


        u_next = self.U_next.eval(feed_dict={self.obs: obs_next,
                                             self.option: option_t}, session=sess)

        target_sr = phi_t + self.gamma * u_next

        return target_sr

    def explore(self, input_q, epsilon):
        p_select = np.random.random()
        if p_select >= epsilon:
            option = random.randint(0, self.n_options - 1)
        else:
            option = np.argmax(input_q, axis=1)
        return option

    def get_options(self, obs, sess, noise=False):
        action_probs = self.Q_option.eval(feed_dict={self.obs: [obs]}, session=sess)
        if noise:
            option = self.explore(action_probs, epsilon=self.epsilon)
        else:
            option = self.explore(action_probs, epsilon=2.0)  # don't explore

        return option

    def get_advantage(self, obs, option, sess):
        q_next, q_max = sess.run([self.Q_option_next, self.Q_option_next_max],
                                 feed_dict={self.obs: obs,
                                            self.option: option})
        advantage = q_next - q_max + self.xi
        return advantage
