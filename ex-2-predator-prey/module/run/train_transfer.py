import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
import pandas as pd
from module.learner import maddpg_sfkt


class Train_engine():
    def __init__(self, MAS, Memorys, Memorys_2, args):
        self.args = args
        self.agents = MAS
        self.task_id = args.id_task
        self.replay_buffer = Memorys
        self.replay_buffer_2 = Memorys_2
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
        # prepare the previous models
        graph_base, sess_base, self.model_base, saver_base = [], [], [], []
        for idx_task in range(self.args.n_tasks):
            graph_base.append(tf.Graph())
            sess_base.append(tf.Session(graph=graph_base[idx_task]))
            with graph_base[idx_task].as_default():
                self.model_base.append(maddpg_sfkt.maddpg(args=self.args, mas_label="mdp_" + str(idx_task) + "_"))
                saver_base.append(tf.train.Saver())
                model_path_pre = "Phase_task_" + str(idx_task) + "/checkpoint_pre/"
                ckpt_pre = tf.train.get_checkpoint_state(model_path_pre)
                if ckpt_pre and ckpt_pre.model_checkpoint_path:
                    saver_base[idx_task].restore(sess_base[idx_task], ckpt_pre.model_checkpoint_path)

        # New task learning
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

            if is_Train or self.args.istransfer:
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

        penalty1 = self.args.penalty1
        penalty2 = self.args.penalty2
        if self.n_predators == 4:
            weight_predator = np.array([penalty1, penalty2, 1.0, 2.0], dtype=np.float32)
            weight_prey = [-1.0, -2.0, -3.0, -4.0]  # * 10
        else:
            weight_predator = np.array([penalty1, penalty2, 1.0], dtype=np.float32)
            weight_prey = [-1.0, -2.0, -3.0]  # * 10
        task_weight = [weight_predator for idx_agt in range(self.n_predators)]
        task_weight.append(weight_prey)

        history_data = np.zeros([int(num_episode / test_period), self.n_agents])
        step_data = 0
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
                    SFs_base_tasks = []
                    for i in range(self.args.n_tasks):
                        feed_obs = {self.model_base[i].obs_t[idx_a]: [obs_t[idx_a]] for idx_a in
                                    range(self.n_predators)}
                        SFs_base_tasks.append(self.model_base[i].SFs.eval(feed_dict=feed_obs, session=sess_base[i]))

                    Q_eval = np.matmul(np.reshape(SFs_base_tasks, [self.args.n_tasks, self.d_phi]), task_weight[0])
                    idx_selected = np.argmax(Q_eval)  # except the policy for common task(-1)
                    act_t = self.get_actions_pre(id_base=idx_selected, obs=obs_t,
                                                 sess_predator=sess_base[int(idx_selected)], sess_prey=sess_prey,
                                                 noise=False)
                    act_step_t = self.joint_action(act_t)
                    obs_next, reward, done, info = env.step(act_step_t)
                    # env.render()
                    r_feature = np.sum(reward[0:self.n_predators], axis=0)
                    r_t = np.dot(r_feature, task_weight[0])
                    acc_reward += r_t
                    self.replay_buffer_2.append(obs0=obs_t, action=act_t,
                                                reward=r_t, feature=r_feature,
                                                obs1=obs_next,
                                                terminal1=False, training=is_Train)
                    obs_t = deepcopy(obs_next)

                    if self.replay_buffer_2.nb_entries < batch:
                        continue

                    samples = self.replay_buffer_2.sample(batch_size=batch)

                    feed_obs = {self.predators.obs_t[idx_a]: samples['obs0'][:, idx_a, :]
                                for idx_a in range(self.n_predators)}
                    feed_act = {self.predators.act_t_target[idx_a]: samples['actions'][:, idx_a, :]
                                for idx_a in range(self.n_predators)}
                    feed_actor = {}
                    feed_actor.update(feed_obs)
                    feed_actor.update(feed_act)
                    _, _, summary_a_2 = sess.run([self.predators.trainer_a_2, self.predators.loss_a_2, self.predators.merge_a_2], feed_dict=feed_actor)
                    writer.add_summary(summary_a_2, idx_epo * len_episodes + idx_step)

                    q_predict = self.predators.get_target_q(features=samples['features'],
                                                            obs_next=samples['obs1'],
                                                            task_weight=task_weight[0],
                                                            sess=sess)

                    feed_critic = {}
                    feed_critic.update(feed_obs)
                    feed_critic.update(feed_act)
                    feed_critic.update({self.predators.Q_target: q_predict})
                    feed_critic.update({self.predators.R_weight: task_weight[0].reshape([-1, 1])})
                    _, lc_2 = sess.run([self.predators.trainer_c_2, self.predators.loss_c_2], feed_dict=feed_critic)
                    _, lc_2, summary_c_2 = sess.run([self.predators.trainer_c_2, self.predators.loss_c_2, self.predators.merge_c_2], feed_dict=feed_critic)
                    writer.add_summary(summary_c_2, idx_epo * len_episodes + idx_step)

                if idx_epo % 50 == 0:
                    print("Step: ", idx_epo, "Mean Episode Reward: ", acc_reward)

                # save model
                model_step += 1
                if model_step >= n_epo_transfer - 10:
                    self.predators.update_target_net(sess=sess, init=True)
                self.saver.save(sess, model_path, global_step=model_step)

            return

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

                # update successor feature networks
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
                    others = np.delete(agt_list, idx_agt)
                    feed_act = {}
                    feed_act_others = {self.predators.act_t[others[idx_o]]: samples['actions'][:, others[idx_o], :]
                                       for idx_o in range(self.n_predators - 1)}
                    feed_act.update(feed_obs)
                    feed_act.update(feed_act_others)
                    feed_act.update({self.predators.R_weight: np.reshape(task_weight[idx_agt], [-1, 1])})
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
                history_data[step_data] = deepcopy(ave_total_reward)

                print('MDP: new, E: {0:5d}, Score: {1}'.format(idx_epo, ave_total_reward))

                # save model
                model_step += 1
                step_data += 1
                self.saver.save(sess, model_path, global_step=model_step)

        saved_to_csv = pd.DataFrame(columns=saved_to_csv_column, data=history_data)
        saved_to_csv.to_csv("Phase_task_new/run_.-tag-Total_reward_mdp_new.csv")

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

    def get_actions_pre(self, id_base, obs, sess_predator, sess_prey, noise=False):
        actions = np.zeros([self.n_agents, self.d_a])
        actions[0: self.n_predators, :] = self.model_base[id_base].get_actions(obs=obs[0: self.n_predators],
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
