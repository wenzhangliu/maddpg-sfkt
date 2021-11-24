from module.learner import ddpg as single_agent_model
import config
import tensorflow as tf


def run(arglist):
    # file path for previous models and data
    fold_name = "saved_model_" + arglist.method
    arglist.data_dir = fold_name
    arglist.checkpoint_dir = fold_name + "/checkpoint_pre/"
    arglist.log_dir = "./" + fold_name + "/log_pre"
    # Create environment
    env = config.make_env(arglist.scenario, num_agt=arglist.n_agts, target_id=arglist.id_task)

    arglist.n_agents = arglist.n_agts
    arglist.dim_o = env.observation_space[0].shape[0]
    arglist.dim_a = 2 if ((arglist.method == "MADDPG") or (arglist.method == "MADDPG")) else 5
    if arglist.method == "MAOPT":
        arglist.dim_a = 2
    arglist.dim_s = env.get_global_state_size()
    arglist.dim_phi = arglist.d_phi
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
        reward_shape = (1,)
        state_shape = (arglist.dim_s, )
    elif (arglist.method == "ATT_MADDPG"):
        from module.learner import att_maddpg as RL_model
        from module.run import train_att_maddpg as train
        reward_shape = (arglist.n_agents,)
    elif (arglist.method == "MAOPT"):
        from module.learner import maopt_sro as RL_model
        from module.run import train_maopt as train
        obs_shape = (arglist.n_agents, arglist.dim_o)
        act_shape = (arglist.n_agents, arglist.dim_a)
        act_adv_shape = (arglist.n_agents - 1, arglist.dim_a)
        reward_shape = (arglist.n_agents,)
        option_shape = (arglist.n_agents - 1, arglist.n_agents - 1)
    elif (arglist.method == "UneVEn"):
        from module.learner import uneven as RL_model
        from module.run import train_uneven as train
        arglist.n_related_task = 3
        obs_shape = (arglist.n_agents, arglist.dim_o)
        act_shape = (arglist.n_agents, arglist.dim_a)
        reward_shape = (1,)
        feature_shape = (arglist.dim_phi,)
        policy_embedding_shape = (arglist.n_related_task+1, arglist.dim_phi,)
    else:
        from module.learner import maddpg as RL_model
        from module.run import train_maddpg as train
        reward_shape = (arglist.n_agents, )

    if (arglist.method == "MAOPT"):
        from module import memory_maopt as memory
        replay_buffers = memory.Memory(limit=arglist.buffer_size,
                                       observation_shape=obs_shape,
                                       action_shape=act_shape,
                                       act_adv_shape=act_adv_shape,
                                       reward_shape=reward_shape,
                                       option_shape=option_shape,
                                       state_shape=state_shape)
    elif (arglist.method == "UneVEn"):
        from module import memory_uneven as memory
        replay_buffers = memory.Memory(limit=arglist.buffer_size,
                                       observation_shape=obs_shape,
                                       action_shape=act_shape,
                                       reward_shape=reward_shape,
                                       feature_shape=feature_shape,
                                       policy_embedding_shape=policy_embedding_shape
                                       )
    else:
        from module import memory as memory
        replay_buffers = memory.Memory(limit=arglist.buffer_size,
                                       observation_shape=obs_shape,
                                       action_shape=act_shape,
                                       reward_shape=reward_shape,
                                       state_shape=state_shape
                                       )

    # MARL policy for predators
    arglist.graph_predators = tf.Graph()
    with arglist.graph_predators.as_default():
        marl_model = [RL_model.agent_model(args=arglist, mas_label=arglist.method+"_agt_"+str(i)+"_")
                      for i in range(arglist.n_agents-1)]
        # successor representation option model
        if (arglist.method == "MAOPT"):
            sro_model = RL_model.option_model_SRO(args=arglist, sro_label="SRO_")

    # DDPG policy for prey
    arglist.graph_prey = tf.Graph()
    with arglist.graph_prey.as_default():
        marl_model.append(single_agent_model.agent_model(args=arglist, mas_label="DDPG_"))

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
    arglist = config.parse_args()
    run(arglist)
