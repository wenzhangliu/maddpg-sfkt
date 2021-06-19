import argparse
import config

from module import memory as memory
from module.learner import pid_controller as PID_model
from module.learner import ddpg as RL_model
from module.run import train_escape as train
import tensorflow as tf


def run(arglist):
    # Create environment
    arglist.method = "DDPG"
    # file path for previous models and data
    fold_name = "saved_model_" + arglist.method
    arglist.data_dir = fold_name
    arglist.checkpoint_dir = fold_name + "/checkpoint_pre/"
    arglist.log_dir = "./" + fold_name + "/log_pre"
    env = config.make_env(arglist.scenario, num_agt=arglist.n_agts, target_id=arglist.id_task)

    arglist.n_agents = arglist.n_agts
    arglist.dim_o = env.observation_space[0].shape[0]
    arglist.dim_a = 2
    arglist.actor_net_h_unit = [256, 256]
    arglist.critic_net_h_unit = [256, 256, 256]
    obs_shape = (arglist.n_agents, arglist.dim_o)
    act_shape = (arglist.n_agents, arglist.dim_a)
    reward_shape = (arglist.n_agents,)

    replay_buffers = memory.Memory(limit=arglist.buffer_size,
                                   observation_shape=obs_shape,
                                   action_shape=act_shape,
                                   reward_shape=reward_shape
                                   )

    # PID controller for predators
    marl_model = [PID_model.controller(obs_shape=obs_shape, dim_act=arglist.dim_a, n_agents=arglist.n_agents)
                  for idx_agt in range(arglist.n_agents-1)]

    # DDPG controller for prey
    arglist.graph_prey = tf.Graph()
    with arglist.graph_prey.as_default():
        marl_model.append(RL_model.agent_model(args=arglist, mas_label="DDPG_"))

    Trainer = train.Train_engine(MAS=marl_model,
                                 Memorys=replay_buffers,
                                 args=arglist)

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
