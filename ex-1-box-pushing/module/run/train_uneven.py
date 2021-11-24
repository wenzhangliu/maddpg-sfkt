import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
import pandas as pd
from module.learner.mixer_uneven import mix_value


class Train_engine():
    def __init__(self, MAS, Memorys, args):
        self.args = args
        self.MAS = MAS
        self.task_id = args.id_task
        self.replay_buffer = Memorys
        self.batch_size = args.batch_size
        self.n_agents = args.n_agents
        self.n_others = self.n_agents - 1
        self.n_mdp = args.n_tasks
        self.d_o = args.dim_o
        self.d_a = args.dim_a
        self.d_s = args.dim_s
        self.d_phi = args.dim_phi
        self.move_range = args.move_bound
        self.max_action = 2.0
        self.sigma_task = 0.01  # or 0.1
        self.n_related_task = args.n_related_task
        self.mixer = mix_value(agts=self.MAS, n_agents=self.n_agents, n_related_task=args.n_related_task,
                               d_task=self.d_phi, d_act=self.d_a, d_obs=self.d_o, d_state=self.d_s,
                               batch_size=self.batch_size, method=args.method)

    def run(self, env, len_episodes, num_episode, test_period, model_path, log_path, is_Train=False):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.mixer.update_target_net(sess=sess, init=True)

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
            # load models for current task
            model_step = 0
            model_path = model_path
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                model_step = int(ckpt.model_checkpoint_path[-1])

            penalty = self.args.penalty
            task_weight = np.array([penalty, penalty, 1.0, 1.0], dtype=np.float32)
            # task_weight = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

            actions = np.zeros([self.n_agents, 1], dtype=np.int)
            action_one_hot = np.zeros([self.n_agents, self.d_a], dtype=np.float)
            actions_one_hot_next = np.zeros([self.n_agents, self.n_related_task+1, self.batch_size, self.d_a], dtype=np.float)
            act_step_t = np.zeros([self.n_agents, 5], dtype=np.float)
            history_data = np.zeros([int(num_episode / test_period), self.n_agents])
            saved_to_csv_column = ["Value_Agt_" + str(idx_agt) for idx_agt in range(self.n_agents)]

            # related task Gaussian distribution
            related_mean = task_weight
            related_cov = self.sigma_task * np.eye(self.d_phi)
            n_related_task = self.n_related_task

            print("Start training")
            for idx_epo in range(num_episode):
                related_task_set = np.random.multivariate_normal(mean=related_mean, cov=related_cov,
                                                                 size=n_related_task)
                related_task_all = np.concatenate([related_task_set, related_mean.reshape([1, -1])], axis=0)

                obs_t = env.reset()

                if not is_Train:
                    for idx_step in range(len_episodes):
                        # get actions for each agent
                        for idx_agt, agent in enumerate(self.MAS):
                            actions[idx_agt] = agent.get_actions(obs=obs_t[idx_agt],
                                                                 target_task=task_weight,
                                                                 related_tasks=related_task_set,
                                                                 sess=sess,
                                                                 noise=False)
                            act_step_t[idx_agt] = self.joint_action(actions[idx_agt])
                        print(self.args.method, ". Step: ", idx_step, actions[0], actions[1])

                        obs_next, reward, done, info = env.step(act_step_t)
                        obs_t = deepcopy(obs_next)
                        env.render()
                        time.sleep(0.03)
                    continue

                for idx_step in range(len_episodes):
                    index_steps = idx_epo * len_episodes + idx_step
                    for idx_agt, agent in enumerate(self.MAS):
                        actions[idx_agt] = agent.get_actions(obs=obs_t[idx_agt],
                                                             target_task=task_weight,
                                                             related_tasks=related_task_set,
                                                             sess=sess,
                                                             noise=True)
                        act_step_t[idx_agt] = self.joint_action(actions[idx_agt])
                        action_one_hot[idx_agt] = self.get_one_hot(actions[idx_agt])

                    obs_next, reward, done, info = env.step(act_step_t)
                    r_feature = np.sum(reward, axis=0)
                    r_t = np.dot(r_feature, task_weight)

                    # append replay buffer
                    self.replay_buffer.append(obs0=obs_t,
                                              action=action_one_hot,
                                              obs1=obs_next,
                                              reward=r_t,
                                              feature=r_feature,
                                              p_embedding=related_task_all,
                                              terminal1=False, training=is_Train)
                    obs_t = deepcopy(obs_next)

                    if self.replay_buffer.nb_entries < self.batch_size:
                        continue

                    samples = self.replay_buffer.sample(batch_size=self.batch_size)

                    # get next action
                    for idx_agt, agent in enumerate(self.MAS):
                        actions_next = agent.get_actions_next(obs_next=samples['obs1'][:, idx_agt, :],
                                                              related_tasks_all=samples['p_embedding'],
                                                              sess=sess)
                        actions_one_hot_next[idx_agt] = self.get_one_hot_set(actions_next)

                    sfs_predict = self.mixer.get_target_mausfs(sess=sess,
                                                               obs_next=samples['obs1'],
                                                               act_next=actions_one_hot_next,
                                                               p_embedding=samples['p_embedding'],
                                                               reward_features=samples['features'])

                    feed_mixer = {self.mixer.agents[ia].obs_t: samples['obs0'][:, ia, :] for ia in range(self.n_agents)}
                    feed_mixer.update({self.mixer.agents[ia].act_t: samples['actions'][:, ia, :] for ia in range(self.n_agents)})
                    for i_task in range(self.n_related_task+1):
                        feed_mixer.update({self.mixer.agents[ia].tasks_z[i_task]: samples['p_embedding'][:, i_task, :] for ia in range(self.n_agents)})
                        feed_mixer.update({self.mixer.sfs_target_tot[i_task]: sfs_predict[i_task]})
                    _, _, summary_c = sess.run([self.mixer.trainer_mix, self.mixer.loss_tot, self.mixer.merge_c],
                                               feed_dict=feed_mixer)

                    writer.add_summary(summary_c, idx_epo * len_episodes + idx_step)

                    self.mixer.update_target_net(sess=sess, init=False)

                if idx_epo % test_period == 0:
                    total_reward = np.zeros([self.n_agents])
                    num_test = 5
                    for idx_test in range(num_test):
                        related_task_set = np.random.multivariate_normal(mean=related_mean, cov=related_cov,
                                                                         size=n_related_task)
                        obs_t = env.reset()
                        for t in range(len_episodes):
                            for idx_agt, agent in enumerate(self.MAS):
                                # obs_repeat = np.tile(obs_t[idx_agt], (n_related_task + 1, 1))
                                actions[idx_agt] = agent.get_actions(obs=obs_t[idx_agt],
                                                                     target_task=task_weight,
                                                                     related_tasks=related_task_set,
                                                                     sess=sess,
                                                                     noise=False)
                                act_step_t[idx_agt] = self.joint_action(actions[idx_agt])
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

                    print('Method: {0}, E: {1:5d}, Score: {2}, alpha: {3:.4f}, epsilon: {4:.4f}, sigma:{5:.4f}'.format(
                        self.args.method, idx_epo, ave_total_reward, self.MAS[0].alpha, self.MAS[0].epsilon, self.sigma_task))

                    # save model
                    model_step += 1
                    self.saver.save(sess, model_path, global_step=model_step)

            saved_to_csv = pd.DataFrame(columns=saved_to_csv_column, data=history_data)
            saved_to_csv.to_csv(self.args.data_dir + "/run_.-tag-Total_reward_" + self.args.method + ".csv")

    def joint_action(self, act):
        act_return = np.zeros([5])
        if act == 1:
            act_return[1] = -self.max_action
        elif act == 2:
            act_return[1] = self.max_action
        elif act == 3:
            act_return[3] = -self.max_action
        elif act == 4:
            act_return[3] = self.max_action
        else:
            act_return[1] = 0.0
            act_return[3] = 0.0

        return act_return

    def get_one_hot(self, act):
        one_hot_act = np.zeros([self.d_a])
        one_hot_act[act] = 1.0
        return one_hot_act

    def get_one_hot_set(self, actions):
        one_hot_act_set = []
        for i_task in range(self.n_related_task + 1):
            one_hot_act_set.append(np.zeros([len(actions[i_task]), self.d_a]))
            for i_a in range(len(actions[i_task])):
                one_hot_act_set[i_task][i_a, actions[i_task][i_a]] = 1.0
        return one_hot_act_set
