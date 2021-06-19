import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
import pandas as pd


class Train_engine():
    def __init__(self, MAS, Memorys, args):
        self.MAS = MAS
        self.args = args
        self.task_id = args.id_task
        self.replay_buffer = Memorys
        self.batch_size = args.batch_size
        self.n_agents = args.n_agents
        self.n_others = self.n_agents - 1
        self.n_mdp = args.n_tasks
        self.d_o = args.dim_o
        self.d_a = args.dim_a
        self.d_phi = args.dim_phi
        self.move_range = args.move_bound

    def run(self, env, len_episodes, num_episode, test_period, model_path, log_path, is_Train=False):
        # prepare the models and parameters
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.MAS.update_target_net(sess=sess, init=True)

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

            task_weight = np.zeros([self.n_mdp + 1], dtype=np.float)
            task_weight[self.task_id] = 1.0
            task_weight[-1] = 1.0

            # history_data = np.zeros([int(num_episode / test_period), self.n_agents])
            history_data = np.zeros([int(num_episode / test_period), self.n_agents])
            saved_to_csv_column = ["Value_Agt_" + str(idx_agt) for idx_agt in range(self.n_agents)]
            print("Start training")
            for idx_epo in range(num_episode):
                obs_t = env.reset()

                if not is_Train:
                    for idx_step in range(len_episodes):
                        # get actions for each agent
                        act_t = self.MAS.get_actions(obs=obs_t,
                                                     sess=sess,
                                                     noise=False)
                        print(self.args.method + ", MDP: ", self.task_id, ". Step: ", idx_step, act_t[0], act_t[1])
                        act_step_t = self.joint_action(act_t)
                        obs_next, reward, done, info = env.step(act_step_t)
                        obs_t = deepcopy(obs_next)
                        env.render()
                        time.sleep(0.03)
                    continue

                for idx_step in range(len_episodes):
                    act_t = self.MAS.get_actions(obs=obs_t,
                                                 sess=sess,
                                                 noise=True)
                    act_step_t = self.joint_action(act_t)
                    obs_next, reward, done, info = env.step(act_step_t)
                    r_feature = np.sum(reward, axis=0)
                    r_t = np.dot(r_feature, task_weight)
                    self.replay_buffer.append(obs0=obs_t, action=act_t,
                                              reward=r_t, feature=r_feature,
                                              obs1=obs_next,
                                              terminal1=False, training=is_Train)
                    obs_t = deepcopy(obs_next)

                    if self.replay_buffer.nb_entries < self.batch_size:
                        continue

                    samples = self.replay_buffer.sample(batch_size=self.batch_size)

                    sfs_predict = self.MAS.get_target_sfs(features=samples['features'],
                                                          obs_next=samples['obs1'],
                                                          sess=sess)
                    feed_obs = {self.MAS.obs_t[i]: samples['obs0'][:, i, :] for i in range(self.n_agents)}
                    feed_act = {self.MAS.act_t[i]: samples['actions'][:, i, :] for i in range(self.n_agents)}
                    feed_critic = {}
                    feed_critic.update(feed_obs)
                    feed_critic.update(feed_act)
                    feed_critic.update({self.MAS.sfs_target: sfs_predict})
                    _, _, summary_c = sess.run([self.MAS.trainer_c, self.MAS.loss_c, self.MAS.merge_c],
                                               feed_dict=feed_critic)
                    writer.add_summary(summary_c, idx_epo * len_episodes + idx_step)
                    # update actor network
                    agt_list = range(self.n_agents)
                    for idx_agt in agt_list:
                        # if idx_agt == (self.n_agents - 1): continue
                        others = np.delete(agt_list, idx_agt)
                        feed_act = {}
                        feed_act_others = {self.MAS.act_t[others[i]]: samples['actions'][:, others[i], :] for i in
                                           range(self.n_agents - 1)}
                        feed_act.update(feed_obs)
                        feed_act.update(feed_act_others)
                        feed_act.update({self.MAS.R_weight: np.reshape(task_weight, [self.d_phi, 1])})
                        _, _, summary_a = sess.run([self.MAS.trainer_a[idx_agt], self.MAS.loss_a, self.MAS.merge_a],
                                                   feed_dict=feed_act)
                        writer.add_summary(summary_a, idx_epo * len_episodes + idx_step)

                    self.MAS.update_target_net(sess=sess, init=False)

                if idx_epo % test_period == 0:
                    total_reward = np.zeros([self.n_agents])
                    num_test = 5
                    for idx_test in range(num_test):
                        obs_t = env.reset()
                        for t in range(len_episodes):
                            act_t = self.MAS.get_actions(obs=obs_t,
                                                         sess=sess,
                                                         noise=False)
                            act_step_t = self.joint_action(act_t)
                            obs_next, reward, done, info = env.step(act_step_t)
                            r_t = np.dot(reward, task_weight)
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

    def joint_action(self, acts):
        act_joint = np.zeros([self.n_agents, 5])
        act_joint[:, [1, 3]] = acts
        return act_joint

