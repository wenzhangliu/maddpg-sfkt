import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    # parser.add_argument("--iftrain", type=int, default=1, help="train or not")
    parser.add_argument("--iftrain", type=int, default=0, help="train or not")

    parser.add_argument("--scenario", type=str, default="simple_push_box_multi", help="name of the scenario script")
    parser.add_argument("--method", type=str, default="MAOPT", help="name of the scenario script")

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
    fold_name = "saved_model_" + parser.parse_args().method  # + datetime.datetime.now().strftime("%Y-%m%d-%H%M%S")
    parser.add_argument("--data-dir", type=str, default=fold_name)
    parser.add_argument("--checkpoint-dir", type=str, default=fold_name + "/checkpoint_pre/")
    parser.add_argument("--log-dir", type=str, default="./" + fold_name + "/log_pre")

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
    arglist.dim_a = 2 if ((arglist.method == "MADDPG") or (arglist.method == "ATT_MADDPG")) else 5
    if arglist.method == "MAOPT":
        arglist.dim_a = 2
    arglist.dim_s = env.get_global_state_size()
    arglist.dim_phi = arglist.n_tasks + 1
    arglist.actor_net_h_unit = [256, 256]
    arglist.critic_net_h_unit = [256, 256, 256]
    obs_shape = (arglist.n_agents, arglist.dim_o)
    act_shape = (arglist.n_agents, arglist.dim_a)
    state_shape = None
    option_shape = None

    if arglist.method == "IQL":
        from module.learner import independent_q as RL_model
        from module.run import train_iql as train
        reward_shape = (arglist.n_agents,)
    elif (arglist.method == "VDN") or (arglist.method == "QMIX"):
        from module.learner import vdn as RL_model
        from module.run import train_mix as train
        reward_shape = (1, )
        state_shape = (arglist.dim_s, )
    elif (arglist.method == "ATT_MADDPG"):
        from module.learner import att_maddpg as RL_model
        from module.run import train_att_maddpg as train
        reward_shape = (arglist.n_agents,)
    elif (arglist.method == "MAOPT"):
        from module.learner import maopt_sro as RL_model
        from module.run import train_maopt as train
        act_shape = (arglist.n_agents, arglist.dim_a)
        reward_shape = (arglist.n_agents,)
        option_shape = (arglist.n_agents, arglist.n_agents)
    else:
        from module.learner import maddpg as RL_model
        from module.run import train_maddpg as train
        reward_shape = (arglist.n_agents, )

    if arglist.method == "MAOPT":
        from module import memory_maopt as memory
        replay_buffers = memory.Memory(limit=arglist.buffer_size,
                                       observation_shape=obs_shape,
                                       action_shape=act_shape,
                                       reward_shape=reward_shape,
                                       option_shape=option_shape,
                                       state_shape=state_shape)
    else:
        from module import memory as memory
        replay_buffers = memory.Memory(limit=arglist.buffer_size,
                                       observation_shape=obs_shape,
                                       action_shape=act_shape,
                                       reward_shape=reward_shape,
                                       state_shape=state_shape
                                       )

    marl_model = [RL_model.agent_model(args=arglist, mas_label=arglist.method+"_agt_"+str(i)+"_")
                  for i in range(arglist.n_agents)]

    # successor representation option model
    if (arglist.method == "MAOPT"):
        sro_model = RL_model.option_model_SRO(args=arglist, sro_label="SRO_")

    if (arglist.method == "MAOPT"):
        Trainer = train.Train_engine(MAS=marl_model,
                                     SRO=sro_model,
                                     Memorys=replay_buffers,
                                     args=arglist)
    else:
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
    arglist = parse_args()
    run(arglist)
