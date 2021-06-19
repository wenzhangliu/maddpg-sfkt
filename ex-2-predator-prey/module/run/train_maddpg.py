import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
import pandas as pd


class Train_engine():
    def __init__(self, MAS, Memorys, args):
        self.agents = MAS
        self.replay_buffer = Memorys
        self.args = args
        self.batch_size = args.batch_size
        self.n_agents = args.n_agents
        self.n_prey = 1
        self.n_predators = self.n_agents - self.n_prey
        self.predators = MAS[0: self.n_predators]
        self.n_others = self.n_predators - 1
        self.n_mdp = args.n_tasks
        self.d_o = self.args.dim_o
        self.d_a = self.args.dim_a
        self.random_policy_std = args.random_policy_std

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
            for idx_agt in range(self.n_predators):
                self.agents[idx_agt].update_target_net(sess=sess, init=True)

            if is_Train:
                self.Total_reward = tf.placeholder(tf.float32, [self.n_agents], "total_reward")
                self.Total_reward_sum = []
                with tf.name_scope(self.args.method):
                    name = "/Score_agt_"
                    self.Total_reward_sum = [tf.summary.scalar(name + str(idx_agt + 1), self.Total_reward[idx_agt])
                                             for idx_agt in range(self.n_agents)]
                # merge_all = tf.summary.merge_all()
                merge_reward = tf.summary.merge([self.Total_reward_sum])
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

            act_t = np.zeros([self.n_agents, self.d_a])
            act_next = np.zeros([self.batch_size, self.n_agents, self.d_a])
            loss_a = np.zeros([self.n_predators])
            loss_c = np.zeros([self.n_predators])

            history_data = np.zeros([int(num_episode / test_period), self.n_agents])
            saved_to_csv_column = ["Value_Agt_" + str(idx_agt) for idx_agt in range(self.n_agents)]
            print("Start training")
            for idx_epo in range(num_episode):
                obs_t = env.reset()

                if not is_Train:
                    for idx_step in range(len_episodes):
                        # get actions for each agent
                        act_t = self.get_actions(obs=obs_t,
                                                 sess_predator=sess, sess_prey=sess_prey,
                                                 noise=False)
                        act_step_t = self.joint_action(act_t)
                        obs_next, reward, done, info = env.step(act_step_t)
                        r_t = [np.dot(reward[idx_agt], task_weight[idx_agt]) for idx_agt in range(self.n_agents)]
                        print(self.args.method, ". Step: ", idx_step, r_t)
                        obs_t = deepcopy(obs_next)
                        env.render()
                        time.sleep(0.03)
                    continue

                for idx_step in range(len_episodes):
                    # get actions for each agent
                    act_t = self.get_actions(obs=obs_t,
                                             sess_predator=sess, sess_prey=sess_prey,
                                             noise=True)
                    act_step_t = self.joint_action(act_t)
                    obs_next, reward, done, info = env.step(act_step_t)
                    r_t = [np.dot(reward[idx_agt], task_weight[idx_agt]) for idx_agt in range(self.n_agents)]
                    self.replay_buffer.append(obs0=obs_t, action=act_t, reward=r_t, obs1=obs_next,
                                              terminal1=False, training=is_Train)
                    obs_t = deepcopy(obs_next)

                    if self.replay_buffer.nb_entries < self.batch_size:
                        continue

                    samples = self.replay_buffer.sample(batch_size=self.batch_size)

                    for idx_agt, agent in enumerate(self.predators):
                        feed_obs_next = {agent.obs_tt: samples['obs1'][:, idx_agt, :]}
                        act_next[:, idx_agt, :] = agent.act_tt.eval(feed_dict=feed_obs_next,
                                                                    session=sess)

                    for idx_agt, agent in enumerate(self.predators):
                        o_others = np.delete(samples['obs0'], idx_agt, axis=1)
                        a_others = np.delete(samples['actions'], idx_agt, axis=1)
                        o_next_others = np.delete(samples['obs1'], idx_agt, axis=1)
                        a_next_others = np.delete(act_next, idx_agt, axis=1)
                        q_predict = agent.get_q_predict(r=samples['rewards'][:, idx_agt:idx_agt + 1],
                                                        obs_next=samples['obs1'][:, idx_agt, :],
                                                        obs_next_others=o_next_others.reshape(self.batch_size, -1),
                                                        act_next_others=a_next_others.reshape(self.batch_size, -1),
                                                        sess=sess)
                        # update critic network
                        _, loss_c[idx_agt] = sess.run([agent.trainer_c, agent.loss_c],
                                                      feed_dict={agent.obs_t: samples['obs0'][:, idx_agt, :],
                                                                 agent.act_t: samples['actions'][:, idx_agt, :],
                                                                 agent.obs_others: o_others.reshape(self.batch_size,
                                                                                                    -1),
                                                                 agent.act_others: a_others.reshape(self.batch_size,
                                                                                                    -1),
                                                                 agent.Q_target: q_predict
                                                                 })
                        # update actor network
                        _, loss_a[idx_agt] = sess.run([agent.trainer_a, agent.loss_a],
                                                      feed_dict={agent.obs_t: samples['obs0'][:, idx_agt, :],
                                                                 agent.obs_others: o_others.reshape(self.batch_size,
                                                                                                    -1),
                                                                 agent.act_others: a_others.reshape(self.batch_size, -1)
                                                                 })

                        agent.update_target_net(sess=sess, init=False)

                if idx_epo % test_period == 0:
                    total_reward = np.zeros(self.n_agents)
                    num_test = 5
                    for idx_test in range(num_test):
                        obs_t = env.reset()
                        for t in range(len_episodes):
                            # get actions for each agent
                            act_t = self.get_actions(obs=obs_t,
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

                    print('Method: {0}, E: {1:5d}, Score: {2}'.format(self.args.method, idx_epo, ave_total_reward))

                    # save model
                    model_step += 1
                    self.saver.save(sess, model_path, global_step=model_step)

            saved_to_csv = pd.DataFrame(columns=saved_to_csv_column, data=history_data)
            saved_to_csv.to_csv(self.args.data_dir + "/run_.-tag-Total_reward_" + self.args.method + ".csv")

    def get_actions(self, obs, sess_predator, sess_prey, noise=False):
        actions = np.zeros([self.n_agents, self.d_a])
        for idx_agt in range(self.n_agents):
            if idx_agt < self.n_predators:
                actions[idx_agt] = self.agents[idx_agt].get_actions(obs=[obs[idx_agt]],
                                                                    sess=sess_predator,
                                                                    noise=noise)
            else:
                if self.args.prey_policy == "random":
                    actions[idx_agt] = np.random.normal(0, self.random_policy_std, size=2)
                else:
                    actions[idx_agt] = self.agents[idx_agt].get_actions(obs=[obs[idx_agt]],
                                                                        sess=sess_prey,
                                                                        noise=noise)
        return actions

    def joint_action(self, acts):
        act_joint = np.zeros([self.n_agents, 5])
        act_joint[:, [1, 3]] = acts
        return act_joint
