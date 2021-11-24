import tensorflow as tf
import numpy as np


class mix_value():
    def __init__(self, agts, n_agents, n_related_task, d_task, d_act, d_obs, d_state, batch_size, method):
        self.agents = agts
        self.n_agents = n_agents
        self.dim_task = d_task
        self.dim_act = d_act
        self.dim_obs = d_obs
        self.dim_state = d_state
        self.d_mix_w1 = 32
        self.d_mix_b2 = 32
        self.batch_size = batch_size
        self.lr = self.agents[0].lr_c
        self.tau = self.agents[0].tau
        self.gamma = self.agents[0].gamma
        self.n_related_tasks = n_related_task

        gather_SFs_a, gather_SFs_tar_a = [], []
        self.mix_input_t, self.mix_input_next = [], []
        self.SFs_holistic, self.SFs_holistic_tar = [], []
        self.sfs_target_tot, self.TD_error_tot, self.loss_tot = [], [], []
        for i_task in range(self.n_related_tasks+1):
            gather_SFs_a.append([tf.expand_dims(self.agents[idx_agt].mausfs_a[i_task], 2) for idx_agt in range(self.n_agents)])
            gather_SFs_tar_a.append([tf.expand_dims(self.agents[idx_agt].mausfs_target_a[i_task], 2) for idx_agt in range(self.n_agents)])
            self.mix_input_t.append(tf.concat(gather_SFs_a[i_task], axis=-1, name="mix_network_sfs_input"))
            self.mix_input_next.append(tf.concat(gather_SFs_tar_a[i_task], axis=-1, name="mix_network_sfs_input_next"))

            self.SFs_holistic.append(self.mixer(agent_qs=self.mix_input_t[i_task], states=None, mixer=method,
                                                reuse=None, scope="Mixing_Network"))
            self.SFs_holistic_tar.append(self.mixer(agent_qs=self.mix_input_next[i_task], states=None, mixer=method,
                                                    reuse=None, scope="Target_Mixing_Network"))
            self.sfs_target_tot.append(tf.placeholder(dtype=tf.float32, shape=[None, self.dim_task],
                                                      name="MAUSFs_tot_target"))
            self.TD_error_tot.append(self.sfs_target_tot[i_task] - self.SFs_holistic[i_task])
            self.loss_tot.append(tf.reduce_mean(tf.reduce_sum(tf.square(self.TD_error_tot[i_task]), axis=1),
                                                name="loss_tot"))

        self.mix_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Mixing_Network")
        self.mix_tar_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Target_Mixing_Network")
        self.var_eval, self.var_eval_tar = [], []
        for idx_agt, agent in enumerate(self.agents):
            self.var_eval.extend(agent.sfs_var)
            self.var_eval_tar.extend(agent.sfs_tar_var)
        self.var_eval.extend(self.mix_var)
        self.var_eval_tar.extend(self.mix_tar_var)

        self.Par_update = [
            self.var_eval_tar[i].assign(
                tf.multiply(self.var_eval[i], self.tau) + tf.multiply(self.var_eval_tar[i], (1 - self.tau)))
            for i in range(len(self.var_eval_tar))]


        self.loss_vdn = sum(self.loss_tot)
        self.loss_vdn_sum = tf.summary.scalar("loss_vdn_sum", self.loss_vdn)
        self.merge_c = tf.summary.merge([self.loss_vdn_sum])

        self.trainer_mix = tf.train.AdamOptimizer(self.lr).minimize(self.loss_vdn, var_list=self.var_eval)

    def mixer(self, agent_qs, states, mixer, reuse, scope):
        if mixer == "UneVEn":
            # q_values = tf.reshape(agent_qs, [self.batch_size, self.dim_task, self.n_agents])
            # output = tf.reduce_sum(q_values, axis=-1, keepdims=True)
            output = tf.reduce_sum(agent_qs, axis=-1)

        elif mixer == "QMIX":
            q_values = tf.reshape(agent_qs, [self.batch_size, 1, self.n_agents])
            with tf.variable_scope(scope, reuse):
                init_range = [0.0, 0.1]
                # first layer: Weight => [n_batch, n_agent, n_h_unit]
                w_hyper_shape_W = [self.dim_state, self.n_agents * self.d_mix_w1]
                b_hyper_shape_W = [self.n_agents * self.d_mix_w1]
                w_hyper_1_1 = tf.Variable(
                    tf.random_uniform(w_hyper_shape_W, minval=init_range[0], maxval=init_range[1]),
                    name="w_hyper_Weight_layer_1")
                b_hyper_1_1 = tf.Variable(
                    tf.random_uniform(b_hyper_shape_W, minval=init_range[0], maxval=init_range[1]),
                    name="b_hyper_Weight_layer_1")
                Weight_mix_1 = tf.reshape(tf.abs(tf.matmul(states, w_hyper_1_1) + b_hyper_1_1),
                                          [-1, self.n_agents, self.d_mix_w1])
                # first layer: Bias => [n_batch, 1, n_h_unit]
                w_hyper_shape_B = [self.dim_state, self.d_mix_w1]
                b_hyper_shape_B = [self.d_mix_w1]
                w_hyper_1_2 = tf.Variable(
                    tf.random_uniform(w_hyper_shape_B, minval=init_range[0], maxval=init_range[1]),
                    name="w_hyper_Bias_layer_1")
                b_hyper_1_2 = tf.Variable(
                    tf.random_uniform(b_hyper_shape_B, minval=init_range[0], maxval=init_range[1]),
                    name="b_hyper_Bias_layer_1")
                Bias_mix_1 = tf.reshape(tf.matmul(states, w_hyper_1_2) + b_hyper_1_2, [-1, 1, self.d_mix_w1])

                # first layer: hidden output => [n_batch, 1, n_h_unit]
                h_output_1 = tf.nn.elu(tf.matmul(q_values, Weight_mix_1) + Bias_mix_1)

                # final layer: Weight => [n_batch, n_h_unit, 1]
                w_hyper_shape_W = [self.dim_state, self.d_mix_w1]
                b_hyper_shape_W = [self.d_mix_w1]
                w_hyper_2_1 = tf.Variable(
                    tf.random_uniform(w_hyper_shape_W, minval=init_range[0], maxval=init_range[1]),
                    name="w_hyper_Weight_layer_2")
                b_hyper_2_1 = tf.Variable(
                    tf.random_uniform(b_hyper_shape_W, minval=init_range[0], maxval=init_range[1]),
                    name="b_hyper_Weight_layer_2")
                Weight_mix_2 = tf.reshape(tf.abs(tf.matmul(states, w_hyper_2_1) + b_hyper_2_1), [-1, self.d_mix_w1, 1])

                # final layer Bias: layer-1 => [n_batch, 16]
                w_hyper_shape_B = [self.dim_state, self.d_mix_b2]
                b_hyper_shape_B = [self.d_mix_b2]
                w_hyper_2_2 = tf.Variable(
                    tf.random_uniform(w_hyper_shape_B, minval=init_range[0], maxval=init_range[1]),
                    name="w_hyper_Bias_layer_2_1")
                b_hyper_2_2 = tf.Variable(
                    tf.random_uniform(b_hyper_shape_B, minval=init_range[0], maxval=init_range[1]),
                    name="b_hyper_Bias_layer_2_1")
                Bias_mix_2_h1 = tf.nn.relu(tf.matmul(states, w_hyper_2_2) + b_hyper_2_2)
                # final layer Bias: layer-2 => [n_batch, 1, 1]
                w_hyper_shape_B = [self.d_mix_b2, 1]
                b_hyper_shape_B = [1]
                w_hyper_2_3 = tf.Variable(
                    tf.random_uniform(w_hyper_shape_B, minval=init_range[0], maxval=init_range[1]),
                    name="w_hyper_Bias_layer_2_2")
                b_hyper_2_3 = tf.Variable(
                    tf.random_uniform(b_hyper_shape_B, minval=init_range[0], maxval=init_range[1]),
                    name="b_hyper_Bias_layer_2_2")
                Bias_mix_2_output = tf.reshape(tf.matmul(Bias_mix_2_h1, w_hyper_2_3) + b_hyper_2_3, [-1, 1, 1])
                h_output_2 = tf.matmul(h_output_1, Weight_mix_2) + Bias_mix_2_output

                output = tf.reshape(h_output_2, [-1, 1])

        else:
            output = None

        return output

    def update_target_net(self, sess, init=False):

        if init:
            for i in range(len(self.var_eval_tar)):
                sess.run(tf.assign(self.var_eval_tar[i], self.var_eval[i]))
        else:
            sess.run(self.Par_update)

    def get_target_mausfs(self, sess, obs_next, act_next, p_embedding, reward_features):
        feed_inputs = {self.agents[idx_agt].obs_tt: obs_next[:, idx_agt, :] for idx_agt in range(self.n_agents)}
        for i_task in range(self.n_related_tasks+1):
            feed_inputs.update({self.agents[idx_agt].act_tt[i_task]: act_next[idx_agt, i_task, :, :]
                                for idx_agt in range(self.n_agents)})
            feed_inputs.update({self.agents[idx_agt].tasks_z_tt[i_task]: p_embedding[:, i_task, :]
                                for idx_agt in range(self.n_agents)})
        mausfs_tot_next = sess.run(self.SFs_holistic_tar, feed_dict=feed_inputs)
        mausfs_target = [reward_features + self.gamma * mausfs_tot_next[i_task] for i_task in range(self.n_related_tasks+1)]
        return mausfs_target
