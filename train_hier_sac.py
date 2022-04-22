import numpy as np
import gym
from arguments.arguments_hier_sac import get_args_ant
from algos.hier_sac import hier_sac_agent

# import goal_env
from goal_env.mujoco import *

import random
import torch


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    test_env = gym.make(args.test)
    test_env1 = test_env2 = None
    print("test_env", test_env1, test_env2)

    # set random seeds for reproduce
    env.seed(args.seed)
    if args.env_name != "NChain-v1" and args.env_name[:5] != "Fetch":
        env.env.env.wrapped_env.seed(args.seed)
        test_env.env.env.wrapped_env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device is not 'cpu':
        torch.cuda.manual_seed(args.seed)
    gym.spaces.prng.seed(args.seed)

    # get the environment parameters
    if args.env_name[:3] in ["Ant", "Poi", "Swi"]:
        env.env.env.visualize_goal = args.animate
        test_env.env.env.visualize_goal = args.animate
    env_params = get_env_params(env)
    env_params['max_test_timesteps'] = test_env._max_episode_steps

    # create the ddpg agent to interact with the environment
    sac_trainer = hier_sac_agent(args, env, env_params, test_env, test_env1, test_env2)
    if args.eval:
        if not args.resume:
            print("random policy !!!")

        # # different options for evaluation below
        # sac_trainer.plot_chain()
        # sac_trainer._eval_hier_agent(test_env)
        # sac_trainer.vis_hier_policy()
        # sac_trainer.cal_slow()
        # sac_trainer.visualize_representation(100)
        # sac_trainer.vis_learning_process('img_maze_scale4_1.pkl')
        # sac_trainer.multi_eval()
        # sac_trainer.plot_exploration()
        # sac_trainer.cal_fall_over()
        # sac_trainer.edge_representation()
        # sac_trainer.picvideo()
        # sac_trainer.plot_density()
        # sac_trainer.cal_phi_loss()
        sac_trainer.eval_intrinsic_rewards()

    else:
        sac_trainer.learn()


# get the params
args = get_args_ant()

if __name__ == '__main__':
    launch(args)
