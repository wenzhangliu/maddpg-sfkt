import argparse


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    # parser.add_argument("--iftrain", type=int, default=1, help="train or not")
    parser.add_argument("--iftrain", type=int, default=0, help="train or not")
    parser.add_argument("--istransfer", type=int, default=0, help="transfer step")

    parser.add_argument("--scenario", type=str, default="predator_prey", help="name of the scenario script")
    parser.add_argument("--method", type=str, default="MAOPT", help="name of the method")

    parser.add_argument("--len-episode", type=int, default=30, help="maximum episode length")
    parser.add_argument("--test-period", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--buffer-size", type=int, default=200000, help="length of replay buffer")
    parser.add_argument("--batch-size", type=int, default=512, help="size of mini batch")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--lr-a", type=float, default=0.001, help="learning rate for actor")
    parser.add_argument("--lr-c", type=float, default=0.001, help="learning rate for critic")
    parser.add_argument("--lr-a-2", type=float, default=0.001, help="learning rate for actor during transfer")
    parser.add_argument("--lr-c-2", type=float, default=0.001, help="learning rate for critic during transfer")
    parser.add_argument("--lr-r", type=float, default=0.001, help="learning rate for reward")
    parser.add_argument("--explore-sigma", type=float, default=0.2, help="sigma: explore noise std")
    parser.add_argument("--tau", type=float, default=0.001, help="tau: soft update rate")
    parser.add_argument("--move-bound", type=float, default=8.0, help="agent move range")
    # Added for ddpg-single
    parser.add_argument("--n-tasks", type=int, default=4)
    parser.add_argument("--d-phi", type=int, default=3)
    parser.add_argument("--id-task", type=int, default=2)
    parser.add_argument("--n-agts", type=int, default=4)
    parser.add_argument("--penalty1", type=float, default=-0.0)
    parser.add_argument("--penalty2", type=float, default=-0.0)
    parser.add_argument("--prey-policy", type=str, default="DDPG")  # random, DDPG
    parser.add_argument("--random-policy-std", type=str, default=1.5, help="std for random forces of random policy")

    return parser.parse_args()


def make_env(scenario_name, num_agt=None, target_id=None, benchmark=False):
    from multiagent_local.environment import MultiAgentEnv
    import multiagent_local.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num_agt=num_agt, target_idx=target_id)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world=world,
                            reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            info_callback=scenario.benchmark_data,
                            done_callback=scenario.done)
    else:
        env = MultiAgentEnv(world=world,
                            reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            done_callback=scenario.done)
    return env
