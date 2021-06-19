import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
import pandas as pd
from module.learner import maddpg_sfkt


class Train_engine():
    def __init__(self, MAS, Memorys, Memorys_2, args):
        self.args = args
        self.MAS = MAS
        self.task_id = args.id_task
        self.replay_buffer = Memorys
        self.replay_buffer_2 = Memorys_2
        self.batch_size = args.batch_size
        self.n_agents = args.n_agents
        self.n_others = self.n_agents - 1
        self.n_mdp = args.n_tasks
        self.d_o = args.dim_o
        self.d_a = args.dim_a
        self.d_phi = args.dim_phi
        self.move_range = args.move_bound

    def run(self, env, len_episodes, num_episode, test_period, model_path, log_path, is_Train=False):
        # prepare the previous models
        task_weight_base = []
        graph_base, sess_base, model_base, saver_base = [], [], [], []
        for idx_task in range(self.d_phi):
            if idx_task == self.d_phi - 1: idx_task = -1
            weight = np.zeros([self.d_phi], dtype=np.float)
            weight[idx_task] = 1.0
            weight[-1] = 1.0
            task_weight_base.append(weight)
            graph_base.append(tf.Graph())
            sess_base.append(tf.Session(graph=graph_base[idx_task]))
            with graph_base[idx_task].as_default():
                model_base.append(maddpg_sfkt.maddpg(args=self.args, mas_label="mdp_" + str(idx_task) + "_"))
                saver_base.append(tf.train.Saver())
                model_path_pre = "Phase_task_" + str(idx_task) + "/checkpoint_pre/"
                ckpt_pre = tf.train.get_checkpoint_state(model_path_pre)
                if ckpt_pre and ckpt_pre.model_checkpoint_path:
                    saver_base[idx_task].restore(sess_base[idx_task], ckpt_pre.model_checkpoint_path)

        # New task learning
        sess = tf.Session(graph=self.MAS.graph)
        with self.MAS.graph.as_default():
            init = tf.global_variables_initializer()
            sess.run(init)
            self.MAS.update_target_net(sess=sess, init=True)

            if is_Train:
                self.Total_reward = tf.placeholder(tf.float32, [self.n_agents], "total_reward")
                self.Total_reward_sum = []
                with tf.name_scope("MDP_new"):
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

            # Copy the parameters for task 2 directly
            # id_copied = 2
            # for idx_agt in range(self.n_agents):
            #     for idx_var, param in enumerate(model_base[id_copied].a_var[idx_agt]):
            #         sess.run(tf.assign(self.MAS.a_var[idx_agt][idx_var], param.eval(session=sess_base[id_copied])))
            #     for idx_var, param in enumerate(model_base[id_copied].a_tar_var[idx_agt]):
            #         sess.run(tf.assign(self.MAS.a_tar_var[idx_agt][idx_var],
            #                            param.eval(session=sess_base[id_copied])))

        task_weight = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32)
        Q_weight = np.array([-1.0, -1.0, 1.0, 2.0], dtype=np.float32)

        history_data = np.zeros([int(num_episode / test_period), self.n_agents])
        saved_to_csv_column = ["Value_Agt_" + str(idx_agt) for idx_agt in range(self.n_agents)]

        # mimic the GPI policy and Q function
        print("Start transferring...")
        is_transfer = self.args.istransfer
        if is_transfer:
            batch = 512
            n_epo_transfer = 2000
            for idx_epo in range(n_epo_transfer):
                obs_t = env.reset()
                acc_reward = 0.0
                for idx_step in range(len_episodes):
                    SFs_base_tasks = [model_base[i].SFs.eval(feed_dict={model_base[i].obs_t[0]: [obs_t[0]],
                                                                        model_base[i].obs_t[1]: [obs_t[1]]
                                                                        },
                                                             session=sess_base[i])
                                      for i in range(self.d_phi)]

                    Q_eval = np.matmul(np.reshape(SFs_base_tasks, [self.d_phi, self.d_phi]), Q_weight)
                    idx_selected = np.argmax(Q_eval[0: self.d_phi-1])  # except the policy for common task(-1)
                    act_t = model_base[idx_selected].get_actions(obs=obs_t,
                                                                 sess=sess_base[idx_selected],
                                                                 noise=False)
                    act_step_t = self.joint_action(act_t)
                    obs_next, reward, done, info = env.step(act_step_t)
                    # env.render()
                    r_t = np.dot(np.sum(reward, axis=0), task_weight)
                    acc_reward += r_t
                    self.replay_buffer_2.append(obs0=obs_t, action=act_t, value=np.max(Q_eval[0: self.d_phi-1]))
                    obs_t = deepcopy(obs_next)

                    if self.replay_buffer_2.nb_entries < batch:
                        continue

                    samples = self.replay_buffer_2.sample(batch_size=batch)

                    _, _, summary_a_2 = sess.run([self.MAS.trainer_a_2, self.MAS.loss_a_2, self.MAS.merge_a_2],
                                    feed_dict={self.MAS.obs_t[0]: samples['obs0'][:, 0, :],
                                               self.MAS.obs_t[1]: samples['obs0'][:, 1, :],
                                               self.MAS.act_t_target[0]: samples['actions'][:, 0, :],
                                               self.MAS.act_t_target[1]: samples['actions'][:, 1, :], })
                    writer.add_summary(summary_a_2, idx_epo * len_episodes + idx_step)
                    feed = {self.MAS.obs_t[0]: samples['obs0'][:, 0, :],
                                                               self.MAS.obs_t[1]: samples['obs0'][:, 1, :],
                                                               self.MAS.act_t[0]: samples['actions'][:, 0, :],
                                                               self.MAS.act_t[1]: samples['actions'][:, 1, :],
                                                               self.MAS.Q_target: samples['q_values']}
                    _, lc_2 = sess.run([self.MAS.trainer_c, self.MAS.loss_c], feed_dict=feed)
                    _, lc_2, summary_c_2 = sess.run([self.MAS.trainer_c, self.MAS.loss_c, self.MAS.merge_c], feed_dict=feed)
                    writer.add_summary(summary_c_2, idx_epo * len_episodes + idx_step)

                print("Step: ", idx_epo, "Mean Episode Reward: ", acc_reward)

                # save model
                model_step += 1
                if model_step >= n_epo_transfer - 10:
                    self.MAS.update_target_net(sess=sess, init=True)
                self.saver.save(sess, model_path, global_step=model_step)

            return

        print("Start training")
        for idx_epo in range(num_episode):
            obs_t = env.reset()

            if not is_Train:
                for idx_step in range(len_episodes):
                    # get actions for each agent
                    act_t = self.MAS.get_actions(obs=obs_t,
                                                 sess=sess,
                                                 noise=False)

                    print("MDP: ", 3, ". Step: ", idx_step, act_t[0], act_t[1])

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
                                          obs1=obs_next, reward=r_t,
                                          terminal1=False, training=is_Train)
                obs_t = deepcopy(obs_next)

                if self.replay_buffer.nb_entries < self.batch_size:
                    continue

                samples = self.replay_buffer.sample(batch_size=self.batch_size)

                # update successor feature networks
                q_predict = self.MAS.get_target_q(rewards=samples['rewards'],
                                                  obs_next=samples['obs1'],
                                                  sess=sess)
                _, _, summary_c = sess.run([self.MAS.trainer_c, self.MAS.loss_c, self.MAS.merge_c],
                                           feed_dict={self.MAS.obs_t[0]: samples['obs0'][:, 0, :],
                                                      self.MAS.obs_t[1]: samples['obs0'][:, 1, :],
                                                      self.MAS.act_t[0]: samples['actions'][:, 0, :],
                                                      self.MAS.act_t[1]: samples['actions'][:, 1, :],
                                                      self.MAS.Q_target: q_predict,
                                                      # self.MAS.R_weight: np.reshape(task_weight,
                                                      #                               [self.d_phi, 1])
                                                      })
                writer.add_summary(summary_c, idx_epo * len_episodes + idx_step)
                # update actor network
                _, _ = sess.run([self.MAS.trainer_a[0], self.MAS.loss_a],
                                feed_dict={self.MAS.obs_t[0]: samples['obs0'][:, 0, :],
                                           self.MAS.obs_t[1]: samples['obs0'][:, 1, :],
                                           self.MAS.act_t[1]: samples['actions'][:, 1, :],
                                           # self.MAS.R_weight: np.reshape(task_weight, [self.d_phi, 1])
                                           })
                _, _, summary_a = sess.run([self.MAS.trainer_a[1], self.MAS.loss_a, self.MAS.merge_a],
                                           feed_dict={self.MAS.obs_t[0]: samples['obs0'][:, 0, :],
                                                      self.MAS.obs_t[1]: samples['obs0'][:, 1, :],
                                                      self.MAS.act_t[0]: samples['actions'][:, 0, :],
                                                      # self.MAS.R_weight: np.reshape(task_weight, [self.d_phi, 1])
                                                      })
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

    def is_out_of_range(self):
        pos = [agent.state.p_pos for agent in self.agents]
        if np.max(pos) > self.move_range:
            return True
        else:
            return False
