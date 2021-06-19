import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
import pandas as pd


class Train_engine():
    def __init__(self, MAS, Memorys, args):
        self.agents = MAS
        self.args = args
        self.task_id = args.id_task
        self.replay_buffer = Memorys
        self.batch_size = args.batch_size
        self.n_agents = args.n_agents
        self.n_prey = 1
        self.n_predators = self.n_agents - self.n_prey
        self.predators = MAS[0]
        self.n_others = self.n_agents - 1
        self.n_mdp = args.n_tasks
        self.d_o = args.dim_o
        self.d_a = args.dim_a
        self.d_phi = args.dim_phi
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
            self.predators.update_target_net(sess=sess, init=True)

            if is_Train:
                self.Total_reward = tf.placeholder(tf.float32, [self.n_agents], "total_reward")
                self.Total_reward_sum = []
                with tf.name_scope(self.args.method + "_MDP_" + str(self.task_id)):
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

            if self.n_predators == 4:
                task_base = [[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.5, 1.0, 0.0],
                             [0.3, 0.5, 0.8, 1.0]]
                weight_predator = task_base[self.task_id]
                weight_prey = [-1.0, -2.0, -3.0, -4.0]  # * 10
            else:
                task_base = [[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             # [0.3, 0.6, 1.0],
                             [0.0, 0.5, 1.0],
                             [0.0, 0.0, 1.0]]
                self.args.n_tasks = len(task_base)
                weight_predator = task_base[self.task_id]
                weight_prey = [-1.0, -2.0, -3.0]  # * 10
            # task_weight = np.array([weight_predator, weight_predator, weight_predator, weight_prey])
            task_weight = [weight_predator for idx_agt in range(self.n_predators)]
            task_weight.append(weight_prey)

            # history_data = np.zeros([int(num_episode / test_period), self.n_agents])
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
                    act_t = self.get_actions(obs=obs_t,
                                             sess_predator=sess, sess_prey=sess_prey,
                                             noise=True)
                    act_step_t = self.joint_action(act_t)
                    obs_next, reward, done, info = env.step(act_step_t)
                    r_feature = np.sum(reward[0:self.n_predators], axis=0)
                    r_t = np.dot(r_feature, task_weight[0])
                    self.replay_buffer.append(obs0=obs_t, action=act_t,
                                              reward=r_t, feature=r_feature,
                                              obs1=obs_next,
                                              terminal1=False, training=is_Train)
                    obs_t = deepcopy(obs_next)

                    if self.replay_buffer.nb_entries < self.batch_size:
                        continue

                    samples = self.replay_buffer.sample(batch_size=self.batch_size)

                    sfs_predict = self.predators.get_target_sfs(features=samples['features'],
                                                                obs_next=samples['obs1'],
                                                                sess=sess)
                    feed_obs = {self.predators.obs_t[i]: samples['obs0'][:, i, :] for i in range(self.n_predators)}
                    feed_act = {self.predators.act_t[i]: samples['actions'][:, i, :] for i in range(self.n_predators)}
                    feed_critic = {}
                    feed_critic.update(feed_obs)
                    feed_critic.update(feed_act)
                    feed_critic.update({self.predators.sfs_target: sfs_predict})
                    _, _, summary_c = sess.run([self.predators.trainer_c,
                                                self.predators.loss_c, self.predators.merge_c],
                                               feed_dict=feed_critic)
                    writer.add_summary(summary_c, idx_epo * len_episodes + idx_step)
                    # update actor network
                    agt_list = range(self.n_predators)
                    for idx_agt in agt_list:
                        # if idx_agt == (self.n_agents - 1): continue
                        others = np.delete(agt_list, idx_agt)
                        feed_act = {}
                        feed_act_others = {self.predators.act_t[others[i]]: samples['actions'][:, others[i], :] for i in
                                           range(self.n_predators - 1)}
                        feed_act.update(feed_obs)
                        feed_act.update(feed_act_others)
                        feed_act.update({self.predators.R_weight: np.reshape(task_weight[idx_agt], [self.d_phi, 1])})
                        _, _, summary_a = sess.run([self.predators.trainer_a[idx_agt],
                                                    self.predators.loss_a, self.predators.merge_a],
                                                   feed_dict=feed_act)
                        writer.add_summary(summary_a, idx_epo * len_episodes + idx_step)

                    self.predators.update_target_net(sess=sess, init=False)

                if idx_epo % test_period == 0:
                    total_reward = np.zeros([self.n_agents])
                    num_test = 5
                    for idx_test in range(num_test):
                        obs_t = env.reset()
                        for t in range(len_episodes):
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

                    print('MDP: {0}, E: {1:5d}, Score: {2}'.format(self.task_id, idx_epo, ave_total_reward))

                    # save model
                    model_step += 1
                    self.saver.save(sess, model_path, global_step=model_step)

            saved_to_csv = pd.DataFrame(columns=saved_to_csv_column, data=history_data)
            saved_to_csv.to_csv(
                "./Phase_task_" + str(self.task_id) + "/run_.-tag-Total_reward_mdp_" + str(self.task_id) + ".csv")

    def get_actions(self, obs, sess_predator, sess_prey, noise=False):
        actions = np.zeros([self.n_agents, self.d_a])
        actions[0: self.n_predators, :] = self.predators.get_actions(obs=obs[0: self.n_predators],
                                                                     sess=sess_predator,
                                                                     noise=noise)
        if self.args.prey_policy == "random":
            actions[-1, :] = np.random.normal(0, self.random_policy_std, size=2)
        else:
            actions[-1, :] = self.agents[-1].get_actions(obs=[obs[-1]],
                                                         sess=sess_prey,
                                                         noise=noise)
        return actions

    def joint_action(self, acts):
        act_joint = np.zeros([self.n_agents, 5])
        act_joint[:, [1, 3]] = acts
        return act_joint
