import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
import pandas as pd


class Train_engine():
    def __init__(self, MAS, SRO, Memorys, args):
        self.agents = MAS
        self.sr_option = SRO
        self.replay_buffer = Memorys
        self.args = args
        self.batch_size_agt = args.batch_size
        self.batch_size_sro = self.sr_option.batch_size
        self.n_agents = args.n_agents
        self.n_prey = 1
        self.n_predators = self.n_agents - self.n_prey
        self.predators = MAS[0: self.n_predators]
        self.n_others = self.n_predators - 1
        self.n_mdp = args.n_tasks
        self.d_o = self.args.dim_o
        self.d_a = self.args.dim_a
        self.n_options = self.n_predators
        self.random_policy_std = args.random_policy_std

        self.max_action = 1.0

        self.mu = 8e-5
        self.epsilon_annel_time = self.sr_option.epsilon_annel_time
        self.epsilon_finish = self.sr_option.epsilon_finish
        self.delta_epsilon = self.epsilon_finish / (self.epsilon_annel_time - 1)

    def run(self, env, len_episodes, num_episode, test_period, model_path, log_path, is_Train=False):
        sess_prey = tf.Session(graph=self.args.graph_prey)
        with self.args.graph_prey.as_default():
            init = tf.global_variables_initializer()
            sess_prey.run(init)
            saver_prey = tf.train.Saver()
            model_path_prey = "saved_model_DDPG/checkpoint_pre"
            ckpt_pre = tf.train.get_checkpoint_state(model_path_prey)
            if ckpt_pre and ckpt_pre.model_checkpoint_path:
                saver_prey.restore(sess_prey, ckpt_pre.model_checkpoint_path)

        sess = tf.Session(graph=self.args.graph_predators)
        with self.args.graph_predators.as_default():
            init = tf.global_variables_initializer()
            sess.run(init)

            if is_Train:
                self.Total_reward = tf.placeholder(tf.float32, [self.n_agents], "total_reward")
                self.Total_reward_sum = []
                self.Loss_Critic = tf.placeholder(tf.float32, [self.n_predators], "loss_c")
                self.Loss_Actor = tf.placeholder(tf.float32, [self.n_predators], "loss_a")
                self.Loss_SRO = tf.placeholder(tf.float32, [4], "loss_sro")
                with tf.name_scope("Episode_Reward"):
                    name = "/Score_agt_"
                    self.Total_reward_sum = [tf.summary.scalar(name + str(idx_agt + 1), self.Total_reward[idx_agt])
                                             for idx_agt in range(self.n_agents)]
                with tf.name_scope("Losses_Critic"):
                    name = "/Loss_Critic_agt_"
                    self.loss_c_sum = [tf.summary.scalar(name + str(idx_agt + 1), self.Loss_Critic[idx_agt])
                                       for idx_agt in range(self.n_predators)]
                with tf.name_scope("Losses_Actor"):
                    name = "/Loss_Actor_agt_"
                    self.loss_a_sum = [tf.summary.scalar(name + str(idx_agt + 1), self.Loss_Actor[idx_agt])
                                       for idx_agt in range(self.n_predators)]
                with tf.name_scope("Losses_SRO"):
                    self.loss_sro_sum = [tf.summary.scalar("/Loss_phi", self.Loss_SRO[0]),
                                         tf.summary.scalar("/Loss_w", self.Loss_SRO[1]),
                                         tf.summary.scalar("/Loss_sr", self.Loss_SRO[2]),
                                         tf.summary.scalar("/Loss_beta", self.Loss_SRO[3])]
                # merge_all = tf.summary.merge_all()
                merge_reward = tf.summary.merge([self.Total_reward_sum])
                merge_loss_train = tf.summary.merge([self.loss_c_sum, self.loss_a_sum])
                merge_loss_sro = tf.summary.merge([self.loss_sro_sum])
                writer = tf.summary.FileWriter(log_path, sess.graph)

            self.saver = tf.train.Saver()
            model_step = 0
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                model_step = int(ckpt.model_checkpoint_path[-1])

            penalty_1 = self.args.penalty1
            penalty_2 = self.args.penalty2
            if self.n_predators == 4:
                weight_predator = [penalty_1, penalty_2, 1.0, 2.0]  # * 10
                weight_prey = [-1.0, -2.0, -3.0, -4.0]  # * 10
            else:
                weight_predator = [penalty_1, penalty_2, 1.0]  # * 10
                weight_prey = [-1.0, -2.0, -3.0]  # * 10
            # task_weight = np.array([weight_predator, weight_predator, weight_predator, weight_prey])
            task_weight = [weight_predator for idx_agt in range(self.n_predators)]
            task_weight.append(weight_prey)

            options = np.zeros([self.n_predators, 1], dtype=np.int)
            opt_one_hot = np.zeros([self.n_predators, self.n_options])

            history_data = np.zeros([int(num_episode / test_period), self.n_agents])
            saved_to_csv_column = ["Value_Agt_" + str(idx_agt) for idx_agt in range(self.n_agents)]
            print("Start training")
            for idx_epo in range(num_episode):
                obs_t = env.reset()

                if not is_Train:
                    for idx_step in range(len_episodes):
                        # get actions for each agent
                        act_t, _ = self.get_actions(obs=obs_t,
                                                    sess_predator=sess, sess_prey=sess_prey,
                                                    noise=False)
                        act_step_t = self.joint_action(act_t)
                        obs_next, reward, done, info = env.step(act_step_t)
                        r_t = [np.dot(reward[idx_agt], task_weight[idx_agt]) for idx_agt in range(self.n_agents)]
                        print(self.args.method, ". Step: ", idx_step, r_t)
                        obs_t = deepcopy(obs_next)
                        env.render()
                        time.sleep(0.08)
                    continue

                # Training stage

                # choose an option for each agent
                for idx_agt in range(self.n_predators):
                    options[idx_agt] = self.sr_option.get_options(obs=obs_t[idx_agt], sess=sess, noise=True)
                    opt_one_hot[idx_agt] = self.get_one_hot_opt(options[idx_agt])

                # run an episode
                self.f_of_t = 0.5 + np.tanh(3 - self.mu * (idx_epo * len_episodes)) / 2
                # self.f_of_t = 0.0
                for idx_step in range(len_episodes):
                    # get actions and advised actions for each agent
                    act_t, act_adv = self.get_actions(obs=obs_t,
                                                      sess_predator=sess,
                                                      sess_prey=sess_prey,
                                                      options=options,
                                                      noise=True)
                    act_step_t = self.joint_action(act_t)
                    obs_next, reward, done, info = env.step(act_step_t)

                    r_t = [np.dot(reward[idx_agt], task_weight[idx_agt]) for idx_agt in range(self.n_agents)]
                    self.replay_buffer.append(obs0=obs_t, action=act_t, reward=r_t, obs1=obs_next,
                                              options=opt_one_hot.copy(), a_adv=act_adv,
                                              terminal1=False, training=is_Train)
                    obs_t = deepcopy(obs_next)

                    # if option terminates, choose another option
                    for idx_agt in range(self.n_predators):
                        beta = self.sr_option.beta_opt.eval(feed_dict={self.sr_option.obs: [obs_t[idx_agt]],
                                                                       self.sr_option.option: [opt_one_hot[idx_agt]]},
                                                            session=sess)[0, 0]
                        if beta >= 0.8:
                            options[idx_agt] = self.sr_option.get_options(obs=obs_t[idx_agt], sess=sess, noise=True)
                            opt_one_hot[idx_agt] = self.get_one_hot_opt(options[idx_agt])

                    # training sro networks
                    if self.replay_buffer.nb_entries < self.batch_size_sro:
                        continue

                    samples = self.replay_buffer.sample(batch_size=self.batch_size_sro)
                    losses_sro = self.sro_training(batch=samples, session=sess)
                    summary_sro = sess.run(merge_loss_sro, feed_dict={self.Loss_SRO: losses_sro})
                    writer.add_summary(summary_sro, idx_epo * len_episodes + idx_step)

                    if (idx_epo * len_episodes + idx_step) % 1000 == 0:
                        self.sr_option.update_target_sr(sess=sess)
                    if (idx_epo * len_episodes + idx_step) <= 5e4:
                        self.sr_option.epsilon = self.sr_option.epsilon + 1.9e-5

                    if self.replay_buffer.nb_entries < self.batch_size_agt:
                        continue

                    # training agent networks
                    samples_agt = self.replay_buffer.sample(batch_size=self.batch_size_agt)
                    loss_c, loss_a = self.agents_training(batch=samples_agt, session=sess)
                    summary_train = sess.run(merge_loss_train, feed_dict={self.Loss_Critic: loss_c,
                                                                          self.Loss_Actor: loss_a})
                    writer.add_summary(summary_train, idx_epo)

                # Test the Model
                if idx_epo % test_period == 0:
                    total_reward = np.zeros(self.n_agents)
                    num_test = 5
                    for idx_test in range(num_test):
                        obs_t = env.reset()
                        for t in range(len_episodes):
                            # get actions for each agent
                            act_t, _ = self.get_actions(obs=obs_t,
                                                        sess_predator=sess, sess_prey=sess_prey,
                                                        noise=False)
                            act_step_t = self.joint_action(act_t)
                            obs_next, reward, done, info = env.step(act_step_t)
                            r_t = [np.dot(reward[idx_agt], task_weight[idx_agt]) for idx_agt in range(self.n_agents)]
                            total_reward = total_reward + np.reshape(r_t, [self.n_agents])
                            obs_t = deepcopy(obs_next)

                    ave_total_reward = np.divide(total_reward, num_test)
                    summary = sess.run(merge_reward, feed_dict={self.Total_reward: ave_total_reward})
                    writer.add_summary(summary, idx_epo)

                    # save as .csv directly
                    for idx_agt in range(self.n_agents):
                        history_data[model_step] = deepcopy(ave_total_reward)

                    print('Method: {0}, E: {1:5d}, Score: {2}. F_t: {3:2f}, epsilon: {4:2f}'.format(self.args.method,
                                                                                                    idx_epo,
                                                                                                    ave_total_reward,
                                                                                                    self.f_of_t,
                                                                                                    self.sr_option.epsilon))

                    # save model
                    model_step += 1
                    self.saver.save(sess, model_path, global_step=model_step)

            saved_to_csv = pd.DataFrame(columns=saved_to_csv_column, data=history_data)
            saved_to_csv.to_csv(self.args.data_dir + "/run_.-tag-Total_reward_" + self.args.method + ".csv")

    def sro_training(self, batch, session):
        samples_obs = np.concatenate(batch['obs0'][:, 0:self.n_predators, :])
        samples_r = np.concatenate(batch['rewards'][:, 0:self.n_predators]).reshape([-1, 1])
        _, loss_phi = session.run([self.sr_option.trainer_obs, self.sr_option.loss_obs],
                                  feed_dict={self.sr_option.obs: samples_obs,
                                             self.sr_option.reward: samples_r})
        _, loss_w = session.run([self.sr_option.trainer_weight, self.sr_option.loss_weight],
                                feed_dict={self.sr_option.obs: samples_obs,
                                           self.sr_option.reward: samples_r})
        u_target, advantage_next = [], []
        for idx_agt, agent in enumerate(self.predators):
            # calculate U_target
            u_target.append(self.sr_option.get_target_SR(obs_t=batch['obs0'][:, idx_agt, :],
                                                         obs_next=batch['obs1'][:, idx_agt, :],
                                                         option_t=batch['options'][:, idx_agt, :],
                                                         sess=session))
            advantage_next.append(self.sr_option.get_advantage(obs=batch['obs1'][:, idx_agt, :],
                                                               option=batch['options'][:, idx_agt, :],
                                                               sess=session))
        samples_u_target = np.concatenate(u_target)
        samples_advantage_next = np.concatenate(advantage_next)
        samples_options = np.concatenate(batch['options'][:, 0:self.n_predators, :])
        samples_obs_next = np.concatenate(batch['obs1'][:, 0:self.n_predators, :])

        _, loss_sr = session.run([self.sr_option.trainer_option, self.sr_option.loss_option],
                                 feed_dict={self.sr_option.obs: samples_obs,
                                            self.sr_option.option: samples_options,
                                            self.sr_option.U_target: samples_u_target})
        _, loss_beta = session.run([self.sr_option.trainer_beta, self.sr_option.loss_beta],
                                   feed_dict={self.sr_option.advantage: samples_advantage_next,
                                              self.sr_option.obs: samples_obs_next,
                                              self.sr_option.option: samples_options})

        return [loss_phi, loss_w, loss_sr, loss_beta]

    def agents_training(self, batch, session):
        sess = session
        act_next = np.zeros([self.batch_size_agt, self.n_agents, self.d_a])
        l_a = np.zeros([self.n_predators])
        l_c = np.zeros([self.n_predators])
        for idx_agt, agent in enumerate(self.predators):
            feed_obs_next = {agent.obs_tt: batch['obs1'][:, idx_agt, :]}
            act_next[:, idx_agt, :] = agent.act_tt.eval(feed_dict=feed_obs_next,
                                                        session=sess)

        for idx_agt, agent in enumerate(self.predators):
            o_others = np.delete(batch['obs0'], idx_agt, axis=1)
            a_others = np.delete(batch['actions'], idx_agt, axis=1)
            o_next_others = np.delete(batch['obs1'], idx_agt, axis=1)
            a_next_others = np.delete(act_next, idx_agt, axis=1)
            q_predict = agent.get_q_predict(r=batch['rewards'][:, idx_agt:idx_agt + 1],
                                            obs_next=batch['obs1'][:, idx_agt, :],
                                            obs_next_others=o_next_others.reshape(self.batch_size_agt, -1),
                                            act_next_others=a_next_others.reshape(self.batch_size_agt, -1),
                                            sess=sess)
            # update critic network
            _, l_c[idx_agt] = sess.run([agent.trainer_c, agent.loss_c],
                                          feed_dict={agent.obs_t: batch['obs0'][:, idx_agt, :],
                                                     agent.act_t: batch['actions'][:, idx_agt, :],
                                                     agent.obs_others: o_others.reshape(self.batch_size_agt, -1),
                                                     agent.act_others: a_others.reshape(self.batch_size_agt, -1),
                                                     agent.Q_target: q_predict
                                                     })
            # update actor network
            _, l_a[idx_agt] = sess.run([agent.trainer_a, agent.loss_a],
                                          feed_dict={agent.obs_t: batch['obs0'][:, idx_agt, :],
                                                     agent.obs_others: o_others.reshape(self.batch_size_agt, -1),
                                                     agent.act_others: a_others.reshape(self.batch_size_agt, -1),
                                                     agent.option_adv: batch['a_advice'][:, idx_agt, :],
                                                     agent.f_of_t: self.f_of_t
                                                     })

            agent.update_target_net(sess=sess, init=False)

        return l_c, l_a

    def get_actions(self, obs, sess_predator, options=None, sess_prey=None, noise=False):
        actions = np.zeros([self.n_agents, self.d_a])
        actions_adv = np.zeros([self.n_predators, self.d_a])

        for idx_agt in range(self.n_agents):
            if idx_agt < self.n_predators:
                actions[idx_agt] = self.agents[idx_agt].get_actions(obs=[obs[idx_agt]],
                                                                    sess=sess_predator,
                                                                    noise=noise)
                # get advised actions
                if options is not None:
                    advisor = int(options[idx_agt])
                    feed_obs = {self.agents[advisor].obs_t: [obs[idx_agt]]}
                    actions_adv[idx_agt] = self.agents[advisor].act_t.eval(feed_dict=feed_obs, session=sess_predator)

            else:
                if self.args.prey_policy == "random":
                    actions[idx_agt] = np.random.normal(0, self.random_policy_std, size=2)
                else:
                    actions[idx_agt] = self.agents[idx_agt].get_actions(obs=[obs[idx_agt]],
                                                                        sess=sess_prey,
                                                                        noise=noise)
        return actions.copy(), actions_adv.copy()

    def joint_action(self, acts):
        act_joint = np.zeros([self.n_agents, 5])
        act_joint[:, [1, 3]] = acts
        return act_joint

    def get_one_hot_opt(self, opt):
        one_hot_opt = np.zeros([self.n_options])
        one_hot_opt[opt] = 1.0
        return one_hot_opt
