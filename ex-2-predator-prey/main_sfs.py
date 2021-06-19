import config

from module import memory_sfs as memory
from module.run import train_sfs as train
from module.learner import maddpg_sfkt as RL_model
from module.learner import ddpg as single_agent_model

import tensorflow as tf


def run(arglist):
    # file path for previous models and data
    fold_name = "Phase_task_" + str(arglist.id_task)
    arglist.data_dir = fold_name
    arglist.checkpoint_dir = fold_name + "/checkpoint_pre/"
    arglist.log_dir = "./" + fold_name + "/log_pre"
    # Create environment
    env = config.make_env(arglist.scenario, num_agt=arglist.n_agts, target_id=arglist.id_task)

    arglist.n_agents = arglist.n_agts
    arglist.dim_o = env.observation_space[0].shape[0]
    arglist.dim_a = 2
    arglist.dim_phi = arglist.d_phi
    arglist.actor_net_h_unit = [256, 256]
    arglist.critic_net_h_unit = [256, 256, 256]
    obs_shape = (arglist.n_agents, arglist.dim_o)
    act_shape = (arglist.n_agents, arglist.dim_a)
    reward_shape = (1,)
    feature_shape = (arglist.dim_phi,)

    replay_buffers = memory.Memory(limit=arglist.buffer_size,
                                   observation_shape=obs_shape,
                                   action_shape=act_shape,
                                   reward_shape=reward_shape,
                                   feature_shape=feature_shape)

    # MADDPG-SFs policy for predators
    arglist.graph_predators = tf.Graph()
    with arglist.graph_predators.as_default():
        marl_model = RL_model.maddpg(args=arglist, mas_label="mdp_" + str(arglist.id_task) + "_")

    # DDPG policy for prey
    arglist.graph_prey = tf.Graph()
    with arglist.graph_prey.as_default():
        prey_model = single_agent_model.agent_model(args=arglist, mas_label="DDPG_")

    Trainer = train.Train_engine(MAS=[marl_model, prey_model], Memorys=replay_buffers, args=arglist)

    Trainer.run(env=env,
                len_episodes=arglist.len_episode,
                num_episode=arglist.num_episodes,
                test_period=arglist.test_period,
                model_path=arglist.checkpoint_dir,
                log_path=arglist.log_dir,
                is_Train=arglist.iftrain)


if __name__ == '__main__':
    arglist = config.parse_args()
    run(arglist)

