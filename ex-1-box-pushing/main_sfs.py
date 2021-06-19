import argparse

from module import memory_sfs as memory
from module.run import train_sfs as train
from module.learner import maddpg_sfkt as RL_model


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    # parser.add_argument("--iftrain", type=int, default=1, help="train or not")
    parser.add_argument("--iftrain", type=int, default=0, help="train or not")

    parser.add_argument("--scenario", type=str, default="simple_push_box_multi", help="name of the scenario script")
    parser.add_argument("--method", type=str, default="MADDPG-SFs", help="name of the scenario script")

    parser.add_argument("--len-episode", type=int, default=60, help="maximum episode length")
    parser.add_argument("--test-period", type=int, default=20, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=6000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--buffer-size", type=int, default=200000, help="length of replay buffer")
    parser.add_argument("--batch-size", type=int, default=512, help="size of mini batch")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--lr-a", type=float, default=0.0001, help="learning rate for actor")
    parser.add_argument("--lr-c", type=float, default=0.0001, help="learning rate for critic")
    parser.add_argument("--lr-r", type=float, default=0.001, help="learning rate for reward")
    parser.add_argument("--explore-sigma", type=float, default=0.2, help="sigma: explore noise std")
    parser.add_argument("--tau", type=float, default=0.001, help="tau: soft update rate")
    parser.add_argument("--move-bound", type=float, default=8.0, help="agent move range")
    # Added for ddpg-single
    parser.add_argument("--n-tasks", type=int, default=3)
    parser.add_argument("--id-task", type=int, default=2)
    parser.add_argument("--n-agts", type=int, default=2)
    parser.add_argument("--penalty", type=float, default=-0.00)
    # file path for previous models and data
    fold_name = "Phase_task_" + str(parser.parse_args().id_task)
    parser.add_argument("--phase-source", type=str, default=fold_name + '_')
    parser.add_argument("--checkpoint-dir-pre", type=str, default=fold_name + "/checkpoint_pre/")
    parser.add_argument("--log-dir-pre", type=str, default="./" + fold_name + "/log_pre")
    # file path for target models and data
    parser.add_argument("--phase-target", type=str, default="Phase_target_")
    parser.add_argument("--checkpoint-dir", type=str, default="Phase_target/checkpoint/")
    parser.add_argument("--log-dir", type=str, default="./Phase_target/log")

    return parser.parse_args()


def make_env(scenario_name, num_agt=2, target_id=None, benchmark=False):
    from multiagent_local.environment import MultiAgentEnv
    import multiagent_local.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num_agt=num_agt, target_idx=target_id)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            scenario.done)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done)
    return env


def run(arglist):
    # Create environment
    env = make_env(arglist.scenario, num_agt=arglist.n_agts, target_id=arglist.id_task)

    arglist.n_agents = arglist.n_agts
    arglist.dim_o = env.observation_space[0].shape[0]
    arglist.dim_a = 2
    arglist.dim_phi = arglist.n_tasks + 1
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

    marl_model = RL_model.maddpg(args=arglist, mas_label="mdp_" + str(arglist.id_task) + "_")

    Trainer = train.Train_engine(MAS=marl_model, Memorys=replay_buffers, args=arglist)

    Trainer.run(env=env,
                len_episodes=arglist.len_episode,
                num_episode=arglist.num_episodes,
                test_period=arglist.test_period,
                model_path=arglist.checkpoint_dir_pre,
                log_path=arglist.log_dir_pre,
                is_Train=arglist.iftrain)


if __name__ == '__main__':
    arglist = parse_args()
    run(arglist)

