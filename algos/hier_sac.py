import os
import sys

sys.path.append('../')
from datetime import datetime
from tensorboardX import SummaryWriter
from models.networks import *
from algos.replay_buffer import replay_buffer, replay_buffer_energy
from algos.her import her_sampler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from algos.sac.sac import SAC
from algos.sac.replay_memory import ReplayMemory, Array_ReplayMemory
import gym
import pickle
from planner.simhash import HashingBonusEvaluator
from planner.grid_hash import GridHashing
from planner.direct_grid import DirectGrid
import imageio
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_color_codes()
import matplotlib
import time
import torch.nn.utils.prune as prune
import copy


SUBGOAL_RANGE = 1000.0


class hier_sac_agent:
    def __init__(self, args, env, env_params, test_env, test_env1=None, test_env2=None):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.device = args.device
        self.resume = args.resume
        self.resume_epoch = args.resume_epoch
        self.not_train_low = False
        self.learn_hi = True
        self.test_env1 = test_env1
        self.test_env2 = test_env2
        self.old_sample = args.old_sample
        self.marker_set = ["v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*"]
        self.color_set = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        T = self.env_params['max_timesteps']
        size = args.buffer_size // T
        self.candidate_idxs = np.array([[i, j] for i in range(size) for j in range(T - args.c + 1)])
        idxs_for_low = np.array([[i, j] for i in range(size) for j in range(T)])

        # add phi(s) to low obs
        self.add_phi = True

        # get maze id
        try:
            self.maze_id = self.env.env.env._maze_id
        except:
            self.maze_id = None
        print("Maze_id", self.maze_id, "!!!")

        if self.add_phi:
            self.low_dim = env_params['obs'] + 2
        else:
            self.low_dim = env_params['obs']
        self.env_params['low_dim'] = self.low_dim
        self.hi_dim = env_params['obs']
        print("hi_dim", self.hi_dim)

        self.learn_goal_space = True
        self.not_update_phi = False
        self.whole_obs = False  # use whole observation space as subgoal space
        self.abs_range = abs_range = args.abs_range  # absolute goal range
        self.feature_reg = 0.0  # feature l2 regularization
        print("abs_range", abs_range)

        if args.env_name[:5] == "Fetch":
            maze_low = self.env.env.initial_gripper_xpos[:2] - self.env.env.target_range
            maze_high = self.env.env.initial_gripper_xpos[:2] + self.env.env.target_range
            self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)
        else:
            if args.env_name != "NChain-v1":
                self.hi_act_space = self.env.env.maze_space
            else:
                self.hi_act_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]))
        if self.learn_goal_space:
            if args.env_name == "NChain-v1":
                self.hi_act_space = gym.spaces.Box(low=np.array([-abs_range]), high=np.array([abs_range]))
            else:
                self.hi_act_space = gym.spaces.Box(low=np.array([-abs_range, -abs_range]), high=np.array([abs_range, abs_range]))
        if self.whole_obs:
            vel_low = [-10.] * 4
            vel_high = [10.] * 4
            maze_low = np.concatenate((self.env.env.maze_low, np.array(vel_low)))
            maze_high = np.concatenate((self.env.env.maze_high, np.array(vel_high)))
            self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)


        if args.replay_strategy == "future":
            self.low_forward = True
            dense_low = False
        else:
            self.low_forward = False
            dense_low = True
        self.low_use_clip = not dense_low  # only sparse reward use clip
        self.hi_sparse = (self.env.env.reward_type == "sparse")

        # params of learning phi
        resume_phi = args.resume
        phi_path = args.resume_path

        self.save_fig = False
        self.save_model = True
        self.start_update_phi = args.start_update_phi
        self.phi_interval = 100
        self.early_stop = args.early_stop  # after success rate converge, don't update low policy and feature
        if args.early_stop:
            if args.env_name in ['AntPush-v1', 'AntFall-v1']:
                self.early_stop_thres = 3500
            elif args.env_name in ["PointMaze1-v1"]:
                self.early_stop_thres = 2000
            elif args.env_name == "AntMaze1-v1":
                self.early_stop_thres = args.n_epochs
            else:
                self.early_stop_thres = 10000
        else:
            self.early_stop_thres = args.n_epochs
        print("early_stop_threshold", self.early_stop_thres)
        self.success_log = []

        try:
            self.scaling = scaling = self.env.env.env.MAZE_SIZE_SCALING
            print("scaling", scaling)
            print("reward type:", self.env.env.reward_type)
        except:
            pass

        # use count
        self.count_latent = True  # False is not using the count
        self.usual_update_hash = False
        self.grid_scale = grid_scale = 3.0
        self.subgoal_grid_scale = 3.0
        if self.count_latent:
            self.hash = GridHashing(grid_scale, obs_processed_flat_dim=2)
            self.xy_hash = GridHashing(grid_scale * 0.2, obs_processed_flat_dim=2)
        self.hi_horizon = int(self.env_params['max_timesteps'] / args.c)
        self.count_xy_record = [[] for _ in range(self.hi_horizon)]
        self.subgoal_record = [[] for _ in range(self.hi_horizon)]  # record subgoals selected in recent 50 episodes
        self.distance_record = [[] for _ in range(self.hi_horizon)]
        self.valid_times = 0
        self.start_explore = self.phi_interval

        # add some noise to the selected subgoal
        self.delta_r = 5.0
        self.success_hash = GridHashing(self.subgoal_grid_scale, 2)
        self.success_coeff = 10.0
        self.start_count_success = 400
        self.dist_to_goal = 0.

        # use future count
        self.future_count_coeff = 1.0  # 0.0 is not using the future count
        self.future_hash = GridHashing(self.subgoal_grid_scale, 2)
        self.inc_number_hash = GridHashing(self.subgoal_grid_scale, 2)
        self.direct_grid_xy = DirectGrid(env, grid_scale * 0.2)

        # ablation study of intrinsic rewards
        self.intrinsic_coeff = 0.0  # set to zero, when action shaping
        self.intrinsic_reward_log = []  # record intrinsic rewards

        # consider the min count for a trajectory
        self.min_coeff = 0.0

        # fixed configs
        self.distance_coeff = 0
        self.history_subgoal_coeff = 0.0
        self.min_dist = 0.0
        self.usual_update_history = False
        self.p_phi_old = True  # prioritize feature learning
        self.add_reg = True  # add stable loss
        self.stable_coeff = 0.001
        self.hi_ratio = 0.5
        self.low_ratio = 0.3
        self.low_p = []

        # record all history subgoal
        if self.history_subgoal_coeff != 0:
            self.subgoal_xy_hash = GridHashing(self.subgoal_grid_scale * 0.2, 2)
            self.subgoal_hash = GridHashing(self.subgoal_grid_scale, 2)
            self.all_history_xy = []
            self.all_history_subgoal = []
        if self.usual_update_history:
            self.all_history_obs = []

        self.count_obs = False
        if self.count_obs:
            self.hash = HashingBonusEvaluator(512, env_params['obs'])

        self.high_correct = False
        self.k = args.c
        self.delta_k = 0
        self.prediction_coeff = 0.0
        tanh_output = False
        self.use_prob = False
        print("prediction_coeff", self.prediction_coeff)

        if args.save:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = 'runs/hier/' + str(args.env_name) + '/Intrinsic_Reach_' + current_time + \
                            "_C_" + str(args.c) + "_Image_" + str(args.image) + "_Early_" + str(self.early_stop_thres)  + \
                            "_Seed_" + str(args.seed) + "_Abs_" + str(args.abs_range) + "_NoPhi_" + str(self.not_update_phi) +\
                            "_Grid_" + str(self.subgoal_grid_scale) + "_Start_" + str(self.start_count_success) + "_Success_" + str(self.success_coeff) + \
                            '_Count_' + str(self.count_latent) + '_Usual_' + str(grid_scale) + str(self.usual_update_hash) + \
                            "_Intrinsic_" + str(self.intrinsic_coeff) + "_DeltaR_" + str(self.delta_r)

            self.writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name + "_" + current_time)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            self.fig_path = os.path.join(self.model_path, "figs")
            if not os.path.exists(self.fig_path):
                os.mkdir(self.fig_path)
                print("fig path", self.fig_path)

        # init low-level network
        self.real_goal_dim = self.hi_act_space.shape[0]  # low-level goal space and high-level action space
        self.init_network()

        # init high-level agent
        self.hi_agent = SAC(self.hi_dim + env_params['goal'], self.hi_act_space, args, False, env_params['goal'],
                            args.gradient_flow_value, args.abs_range, tanh_output)
        self.env_params['real_goal_dim'] = self.real_goal_dim
        self.hi_buffer = ReplayMemory(args.buffer_size)

        # her sampler
        self.c = self.args.c  # interval of high level action
        self.low_her_module = her_sampler(args.replay_strategy, args.replay_k, args.distance, args.future_step,
                                          dense_reward=dense_low, direction_reward=False, low_reward_coeff=args.low_reward_coeff,
                                          low_idxs=idxs_for_low)
        if args.env_name[:5] == "Fetch":
            self.low_buffer = replay_buffer_energy(self.env_params, self.args.buffer_size,
                                               self.low_her_module.sample_her_energy, args.env_name)
        else:
            self.low_buffer = replay_buffer(self.env_params, self.args.buffer_size, self.low_her_module.sample_her_transitions, k=self.k)
            # self.low_buffer = replay_buffer(self.env_params, self.args.buffer_size, self.low_her_module.sample_her_prioritized, k=self.k)

        not_load_buffer, not_load_high = False, False
        if self.resume is True:
            self.start_epoch = self.resume_epoch
            if not not_load_high:
                print("load high !!!")
                self.hi_agent.policy.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/hi_actor_5000.pt', map_location='cuda:1')[0])
                self.hi_agent.critic.load_state_dict(torch.load(self.args.resume_path + \
                                                               '/hi_critic_model.pt', map_location='cuda:1')[0])

            # print("not load low !!!")
            print("load low !!!")
            self.low_actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                             '/low_actor_5000.pt', map_location='cuda:1')[0])
            self.low_critic_network.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/low_critic_model.pt', map_location='cuda:1')[0])

            if not not_load_buffer:
                # self.hi_buffer = torch.load(self.args.resume_path + '/hi_buffer.pt', map_location='cuda:1')
                self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')

        # sync target network of low-level
        self.sync_target()

        if hasattr(self.env.env, 'env'):
            self.animate = self.env.env.env.visualize_goal
        else:
            self.animate = self.args.animate
        self.distance_threshold = self.args.distance

        if not (args.gradient_flow or args.use_prediction or args.gradient_flow_value):
            self.representation = RepresentationNetwork(env_params, 3, self.abs_range, self.real_goal_dim).to(args.device)
            self.pruned_phi = None
            if args.use_target:
                self.target_phi = RepresentationNetwork(env_params, 3, self.abs_range, 2).to(args.device)
                # load the weights into the target networks
                self.target_phi.load_state_dict(self.representation.state_dict())
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=0.0001)
            if resume_phi is True:
                print("load phi from: ", phi_path)
                self.representation.load_state_dict(torch.load(phi_path + \
                                                               '/phi_model_5000.pt', map_location='cuda:1')[0])
        elif args.use_prediction:
            self.representation = DynamicsNetwork(env_params, self.abs_range, 2, tanh_output=tanh_output, use_prob=self.use_prob, device=args.device).to(args.device)
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=0.0001)
            if resume_phi is True:
                print("load phi from: ", phi_path)
                self.representation.load_state_dict(torch.load(phi_path + \
                                                               '/phi_model_4000.pt', map_location='cuda:1')[0])

        print("learn goal space", self.learn_goal_space, " update phi", not self.not_update_phi)
        self.train_success = 0.
        self.count_prob = 1.
        self.furthest_task = 0.

    def adjust_lr_actor(self, epoch):
        lr_actor = self.args.lr_actor * (0.5 ** (epoch // self.args.lr_decay_actor))
        for param_group in self.low_actor_optim.param_groups:
            param_group['lr'] = lr_actor

    def adjust_lr_critic(self, epoch):
        lr_critic = self.args.lr_critic * (0.5 ** (epoch // self.args.lr_decay_critic))
        for param_group in self.low_critic_optim.param_groups:
            param_group['lr'] = lr_critic

    # train the agent
    def learn(self):
        for epoch in range(self.start_epoch, self.args.n_epochs):
            if epoch > 0 and epoch % self.args.lr_decay_actor == 0:
                self.adjust_lr_actor(epoch)
            if epoch > 0 and epoch % self.args.lr_decay_critic == 0:
                self.adjust_lr_critic(epoch)

            ep_obs, ep_ag, ep_g, ep_actions, ep_ag_record = [], [], [], [], []
            last_hi_obs = None
            success = 0
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal'][:self.real_goal_dim]
            ag_record = observation['achieved_goal']
            g = observation['desired_goal']

            # identify the furthest task
            if g[1] >= 8:
                self.furthest_task += 1
                is_furthest_task = True
            else:
                is_furthest_task = False
            if self.learn_goal_space:
                if self.args.gradient_flow:
                    if self.args.use_target:
                        ag = self.hi_agent.policy_target.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()
                    else:
                        ag = self.hi_agent.policy.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()
                elif self.args.gradient_flow_value:
                    ag = self.hi_agent.critic.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                elif self.args.use_prediction:
                    ag = self.representation.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                else:
                    if self.args.use_target:
                        ag = self.target_phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                    else:
                        ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
            if self.whole_obs:
                ag = obs.copy()

            valid_count = False

            for t in range(self.env_params['max_timesteps']):
                act_obs, act_g = self._preproc_inputs(obs, g)
                if t % self.c == 0:
                    # record final distance to subgoal
                    if t != 0 and valid_count and self.count_latent and epoch > self.start_explore:
                        # make imagined goal absolute position
                        last_hi_ag = self.representation(torch.Tensor(last_hi_obs[:self.hi_dim]).to(self.device)).detach().cpu().numpy()[0]
                        distance_to_goal = np.linalg.norm(last_hi_a + last_hi_ag - ag)
                        if self.intrinsic_coeff == 0.:
                            self.success_hash.inc_multi(hi_action_ini[None], distance_to_goal)
                            self.dist_to_goal += distance_to_goal
                            self.distance_record[t//self.c].append(distance_to_goal)
                        else:
                            last_hi_r += self.success_coeff / np.sqrt(distance_to_goal)
                    hi_act_obs = np.concatenate((obs[:self.hi_dim], g))

                    # append high-level rollouts
                    if last_hi_obs is not None:
                        mask = float(not done)
                        if self.high_correct:
                            last_hi_a = ag
                        self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, hi_act_obs, mask, epoch)
                    random_num = random.random()
                    valid_count = bool(random_num < self.count_prob)

                    # select subgoal by counts, the subgoal is absolute subgoal
                    if self.count_latent and epoch > self.start_explore and valid_count and (self.intrinsic_coeff == 0.):
                        hi_action_ini, _ = self.select_by_count(obs[:self.hi_dim], t, epoch)
                        if self.success_coeff != 0 and epoch > self.start_count_success:
                            # add some noise to the selected subgoal
                            direction = hi_action_ini - ag
                            norm_direction = direction / np.linalg.norm(direction)
                            hi_action = hi_action_ini + self.delta_r * norm_direction
                        else:
                            hi_action = hi_action_ini
                        hi_action_for_low = hi_action
                        # put delta position to the high-level buffer
                        hi_action_delta = hi_action - ag
                    else:
                        if epoch < self.args.start_epoch:
                            hi_action = self.hi_act_space.sample()
                        else:
                            hi_action = self.hi_agent.select_action(hi_act_obs)
                        if self.old_sample:
                            hi_action_for_low = hi_action
                        else:
                            # make hi_action a delta phi(s)
                            hi_action_for_low = ag.copy() + hi_action.copy()
                            hi_action_for_low = np.clip(hi_action_for_low, -SUBGOAL_RANGE, SUBGOAL_RANGE)
                            # record subgoal selected by intrinsic rewards
                            if self.intrinsic_coeff > 0.:
                                current_hi_step = int(t / self.c)
                                self.subgoal_record[current_hi_step].append(hi_action_for_low)
                        hi_action_delta = hi_action

                    last_hi_obs = hi_act_obs.copy()
                    last_hi_a = hi_action_delta.copy()
                    last_hi_r = 0.
                    done = False
                    hi_action_tensor = torch.tensor(hi_action_for_low, dtype=torch.float32).unsqueeze(0).to(self.device)

                    # update high-level policy
                    if len(self.hi_buffer) > self.args.batch_size and self.learn_hi:
                        self.update_hi(epoch, t)
                with torch.no_grad():
                    if self.not_train_low:
                        action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                    else:
                        action = self.explore_policy(act_obs[:, :self.low_dim], hi_action_tensor)

                # feed the actions into the environment
                observation_new, r, _, info = self.env.step(action)
                if info['is_success']:
                    done = True
                    # only record the first success
                    if success == 0 and is_furthest_task:
                        success = t
                        self.train_success += 1
                if self.animate:
                    self.env.render()

                # update observation
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal'][:self.real_goal_dim]
                ag_record_new = observation_new['achieved_goal']
                if self.learn_goal_space:
                    if self.args.gradient_flow:
                        if self.args.use_target:
                            ag_new = self.hi_agent.policy_target.phi(
                                torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()
                        else:
                            ag_new = self.hi_agent.policy.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()
                    elif self.args.gradient_flow_value:
                        ag_new = self.hi_agent.critic.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                    elif self.args.use_prediction:
                        ag_new = self.representation.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                    else:
                        if self.args.use_target:
                            ag_new = self.target_phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                        else:
                            ag_new = self.representation(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                if self.whole_obs:
                    ag_new = obs_new.copy()

                # counting after starting to update phi (updating phi every 100 episodes)
                if self.count_latent:
                    if not self.usual_update_hash and epoch > self.phi_interval:
                        count_feature = self.pruned_phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()
                        self.hash.inc_hash(count_feature)
                    # add intrinsic rewards
                    if self.intrinsic_coeff > 0. and t > 0 and (t % (self.c - 1) == 0) and epoch > self.phi_interval:
                        intrinsic_rewards = self.hash.predict_rewards(ag_new[None])[0] * self.intrinsic_coeff
                        self.intrinsic_reward_log.append(intrinsic_rewards)
                        r += intrinsic_rewards
                    self.xy_hash.inc_hash(ag_record[:self.real_goal_dim].copy()[None])
                if self.count_obs:
                    self.hash.inc_hash(obs[None])
                    r += self.hash.predict(obs_new[None])[0] * 0.1

                # add r to high-level reward
                last_hi_r += r

                # append rollouts
                if self.add_phi:
                    new_low_obs = np.concatenate((obs[:self.low_dim], ag))
                else:
                    new_low_obs = obs[:self.low_dim]
                ep_obs.append(new_low_obs.copy())
                ep_ag.append(ag.copy())
                ep_ag_record.append(ag_record.copy())
                ep_g.append(hi_action_for_low.copy())
                ep_actions.append(action.copy())

                # re-assign the observation
                obs = obs_new
                ag = ag_new
                ag_record = ag_record_new
            if self.add_phi:
                new_low_obs = np.concatenate((obs[:self.low_dim], ag))
            else:
                new_low_obs = obs[:self.low_dim]
            ep_obs.append(new_low_obs.copy())
            ep_ag.append(ag.copy())
            ep_ag_record.append(ag_record.copy())
            mask = float(not done)
            hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
            self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, hi_act_obs, mask, epoch)

            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_ag_record = np.array([ep_ag_record])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            self.low_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, success, False, mb_ag_record[:, :, :self.real_goal_dim]])
            if self.count_latent:
                if not self.usual_update_hash and epoch > self.phi_interval:
                    count_feature = self.pruned_phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()
                    self.hash.inc_hash(count_feature)
                self.xy_hash.inc_hash(ag_record[:self.real_goal_dim].copy()[None])

            if self.args.save and self.args.env_name == "NChain-v1":
                self.writer.add_scalar('Explore/coverage_' + self.args.env_name, self.env.env.coverage, epoch)

            # update low-level
            if not self.not_train_low:
                for n_batch in range(self.args.n_batches):
                    self._update_network(epoch, self.low_buffer, self.low_actor_target_network,
                                         self.low_critic_target_network,
                                         self.low_actor_network, self.low_critic_network, 'max_timesteps',
                                         self.low_actor_optim, self.low_critic_optim, use_forward_loss=self.low_forward,
                                         clip=self.low_use_clip, n_batch=n_batch)
                    if n_batch % self.args.period == 0:
                        self._soft_update_target_network(self.low_actor_target_network, self.low_actor_network)
                        self._soft_update_target_network(self.low_critic_target_network, self.low_critic_network)

            # piecewise stable-coefficient
            if self.args.image:
                if epoch > 1000:
                    self.stable_coeff = 0.1
            else:
                if epoch > 500:
                    self.stable_coeff = 0.1

            # start to do the evaluation
            if epoch % self.args.eval_interval == 0 and epoch != 0:
                # calculate train success rate
                train_success_rate = self.train_success / self.args.eval_interval * 10
                self.count_prob = np.exp(1 - train_success_rate) / (np.exp(1 - train_success_rate) + np.exp(train_success_rate))
                self.train_SR = train_success_rate / 10
                self.train_success = 0

                # calculate coverage
                transitions, _ = self.low_buffer.sample(1000)
                positions = transitions['ag_record']
                self.direct_grid_xy.update_occupied(positions)
                coverage_ratio = self.direct_grid_xy.coverage()
                print("coverage ratio", coverage_ratio)
                if self.test_env1 is not None:
                    eval_success1, _ = self._eval_hier_agent(env=self.test_env1)
                    eval_success2, _ = self._eval_hier_agent(env=self.test_env2)
                farthest_success_rate, _ = self._eval_hier_agent(env=self.test_env)
                random_success_rate, _ = self._eval_hier_agent(env=self.env)
                self.success_log.append(farthest_success_rate)
                mean_success = np.mean(self.success_log[-5:])

                # stop updating phi and low
                if self.early_stop and (mean_success >= 0.9 or epoch > self.early_stop_thres):
                    print("early stop !!!")
                    self.not_update_phi = True
                    self.not_train_low = True

                # identify whether low-level policy is good or not
                if epoch > self.start_explore and self.dist_to_goal < 100 and (self.intrinsic_coeff == 0.) and self.count_latent:
                    # self.not_train_low = True
                    self.not_update_phi = True
                    print("not update phi !!!")
                print('[{}] epoch is: {}, eval hier success rate is: {:.3f}'.format(datetime.now(), epoch, random_success_rate))
                if self.save_fig:
                    self.vis_hier_policy(epoch=epoch)
                    self.visualize_representation(epoch=epoch)

                if self.args.save:
                    print("log_dir: ", self.log_dir)
                    torch.save([self.hi_agent.critic.state_dict()], self.model_path + '/hi_critic_model.pt')
                    torch.save([self.low_critic_network.state_dict()], self.model_path + '/low_critic_model.pt')
                    torch.save(self.hi_buffer, self.model_path + '/hi_buffer.pt')
                    if not self.args.gradient_flow and not self.args.gradient_flow_value:
                        if self.save_model:
                            # self.cal_MIV(epoch)
                            torch.save([self.representation.state_dict()], self.model_path + '/phi_model_{}.pt'.format(epoch))
                            torch.save([self.hi_agent.policy.state_dict()], self.model_path + '/hi_actor_{}.pt'.format(epoch))
                            torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_{}.pt'.format(epoch))
                            torch.save(self.low_buffer, self.model_path + '/low_buffer.pt')
                        else:
                            torch.save([self.representation.state_dict()], self.model_path + '/phi_model.pt')
                            torch.save([self.hi_agent.policy.state_dict()], self.model_path + '/hi_actor_model.pt')
                            torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_model.pt')
                            torch.save(self.low_buffer, self.model_path + '/low_buffer.pt')

                    # write into logger
                    self.writer.add_scalar('Success_rate/hier_farthest_' + self.args.env_name, farthest_success_rate, epoch)
                    self.writer.add_scalar('Success_rate/hier_random_' + self.args.env_name, random_success_rate, epoch)
                    self.writer.add_scalar('Success_rate/train_' + self.args.env_name, self.train_SR, epoch)
                    self.writer.add_scalar("Success_rate/low_dist_to_goal", self.dist_to_goal, epoch)
                    self.writer.add_scalar("Success_rate/coverage", coverage_ratio, epoch)
                    self.writer.add_scalar('Explore/furthest_task_' + self.args.env_name, self.furthest_task, epoch)
                    self.writer.add_scalar('Explore/Valid_' + self.args.env_name, self.valid_times, epoch)

                    # record intrinsic rewards
                    if self.intrinsic_coeff > 0 and self.count_latent and len(self.intrinsic_reward_log) > 0:
                        intrinsic_array = np.array(self.intrinsic_reward_log)
                        mean_intrinsic = np.mean(intrinsic_array)
                        var_intrinsic = np.var(intrinsic_array)
                        self.writer.add_scalar('Explore/Intrinsic_mean_' + self.args.env_name, mean_intrinsic, epoch)
                        self.writer.add_scalar('Explore/Intrinsic_var_' + self.args.env_name, var_intrinsic, epoch)
                        self.intrinsic_reward_log = []
                    if self.test_env1 is not None:
                        self.writer.add_scalar('Success_rate/eval1_' + self.args.env_name,
                                               eval_success1, epoch)
                        self.writer.add_scalar('Success_rate/eval2_' + self.args.env_name, eval_success2,
                                               epoch)

                self.count_xy_record = [[] for _ in range(self.hi_horizon)]
                self.subgoal_record = [[] for _ in range(self.hi_horizon)]
                self.distance_record = [[] for _ in range(self.hi_horizon)]
                self.dist_to_goal = 0.

            # very very slow to learn phi, update after plotting
            if epoch > self.start_update_phi and not self.not_update_phi and epoch % self.phi_interval == 0:
                start_time1 = time.time()
                # keep a target phi for regularization loss
                self.target_phi_reg = copy.deepcopy(self.representation)
                self.p_lst = []
                self.idx_lst = []
                # replace random.choice with random.randint
                episode_num = self.low_buffer.current_size
                self.cur_candidate_idxs = self.candidate_idxs[:episode_num * (self.low_buffer.T - self.k + 1)]
                p = self.low_buffer.get_all_data()['p']
                p = p[:, :self.low_buffer.T - self.k + 1]
                p = p.reshape(-1)
                argsort_p = np.argsort(p)
                self.high_p = argsort_p[-int(len(argsort_p) * self.hi_ratio):]
                self.low_p = argsort_p[int(len(argsort_p) * self.low_ratio):]
                for update_i in range(50000):
                    self.slow_update_phi(epoch, update_i)

                p_array = np.array(self.p_lst)
                p_array = p_array.reshape(-1, 1)
                idx_array = np.array(self.idx_lst)
                idx_array = idx_array.reshape(-1, idx_array.shape[2])
                self.low_buffer.buffers['p'][idx_array[:, 0], idx_array[:, 1]] = p_array

                # prune phi
                self.pruned_phi = copy.deepcopy(self.representation)

                # update hash table for history subgoal after updating phi
                if self.usual_update_history and self.history_subgoal_coeff != 0 and len(self.all_history_obs) > 0:
                    self.subgoal_hash = GridHashing(self.subgoal_grid_scale, 2)
                    state = np.array(self.all_history_obs)
                    obs_tensor = torch.Tensor(state).to(self.device)
                    features = self.representation(obs_tensor).detach().cpu().numpy()
                    self.subgoal_hash.inc_hash(features)
                    self.all_history_subgoal = features
                print("update phi time", time.time() - start_time1)

            # update hash table after updating phi or every 100 episodes, as sometimes early stop phi
            if self.usual_update_hash and epoch % self.phi_interval == 0 and epoch > self.start_update_phi:
                self.hash = GridHashing(self.grid_scale, obs_processed_flat_dim=2)
                state = self.low_buffer.get_all_data()['obs']
                state = state.reshape(-1, state.shape[2])
                obs_tensor = torch.Tensor(state[:, :self.hi_dim]).to(self.device)
                features = self.representation(obs_tensor).detach().cpu().numpy()
                self.hash.inc_hash(features)

            # calculate future count
            if self.future_count_coeff > 0 and epoch % 10 == 0 and epoch > self.phi_interval and self.count_latent:
                # reinitialize future hash and inc_number_hash
                self.future_hash = GridHashing(self.subgoal_grid_scale, 2)
                self.inc_number_hash = GridHashing(self.subgoal_grid_scale, 2)
                # extract high-level samples from low-level replay buffer
                state = self.low_buffer.get_all_data()['obs']
                selected_indexs = (np.arange(self.hi_horizon) + 1) * self.c
                selected_state = state[:, selected_indexs]
                # query hash table of buffer
                state = selected_state.reshape(-1, selected_state.shape[2])
                obs_tensor = torch.Tensor(state[:, :self.hi_dim]).to(self.device)
                features = self.representation(obs_tensor).detach().cpu().numpy()
                n_hi_s = np.array(self.hash.predict(features))
                # calculate future count with one-step count
                n_hi_s = n_hi_s.reshape(-1, self.hi_horizon)
                N_s = np.zeros((len(n_hi_s), self.hi_horizon + 1))
                for t in reversed(range(self.hi_horizon)):
                    N_s[:, t] = n_hi_s[:, t] + self.args.gamma * N_s[:, t+1]
                N_s = N_s[:, :-1]
                # calcualte the min count of each episode
                min_count = np.min(n_hi_s, axis=1)
                min_count = np.reshape(min_count, (-1, 1))
                min_count = np.tile(min_count, self.hi_horizon)
                # combine N_s and min_count as the final count
                N_s = (1 - self.min_coeff) * N_s + self.min_coeff * min_count

                N_s = N_s.reshape(-1)
                self.future_hash.inc_multi(features, N_s)
                self.inc_number_hash.inc_hash(features)
                self.future_hash.tables = np.divide(self.future_hash.tables + 1, self.inc_number_hash.tables + 1)


    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        if self.add_phi:
            ag = self.representation(obs)
            obs = torch.cat([obs, ag], 1)
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        return obs, g

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        if action.shape == ():
            action = np.array([action])
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        if np.random.rand() < self.args.random_eps:
            action = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                       size=self.env_params['action'])
        return action

    def explore_policy(self, obs, goal):
        pi = self.low_actor_network(obs, goal)
        action = self._select_actions(pi)
        return action

    def update_hi(self, epoch, t):
        if self.args.gradient_flow or self.args.gradient_flow_value:
            sample_data, _ = self.slow_collect()
            sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        else:
            sample_data = None
        critic_1_loss, critic_2_loss, policy_loss, _, _ = self.hi_agent.update_parameters(self.hi_buffer,
                                                                                          self.args.batch_size,
                                                                                          self.env_params,
                                                                                          self.hi_sparse,
                                                                                          sample_data)
        if self.args.save and t == 0:
            self.writer.add_scalar('Loss/hi_critic_1', critic_1_loss, epoch)
            self.writer.add_scalar('Loss/hi_critic_2', critic_2_loss, epoch)
            self.writer.add_scalar('Loss/hi_policy', policy_loss, epoch)

    def random_policy(self, obs, goal):
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        return random_actions

    def test_policy(self, obs, goal):
        pi = self.low_actor_network(obs, goal)
        # convert the actions
        actions = pi.detach().cpu().numpy().squeeze()
        if actions.shape == ():
            actions = np.array([actions])
        return actions

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, epoch, buffer, actor_target, critic_target, actor, critic, T, actor_optim, critic_optim,
                        use_forward_loss=True, clip=True, n_batch=0):
        # sample the episodes
        transitions, ori_selected_idx = buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        obs_cur, obs_next, g_cur, ag = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag']
        ag_next = transitions['ag_next']
        g_next = g_cur.copy()

        # judge whether done at this step
        dist_cur = np.linalg.norm(ag - g_cur, axis=1)
        not_done_cur = (dist_cur > self.distance_threshold)
        selected_idxs = np.where(not_done_cur == True)[0]
        obs_cur, obs_next, g_cur, ag = obs_cur[selected_idxs], obs_next[selected_idxs], g_cur[selected_idxs], ag[selected_idxs]
        ag_next = ag_next[selected_idxs]
        g_next = g_cur.copy()
        if len(obs_next) != len(g_next):
            print("obs_next", obs_next.shape)
            print("g_next", g_next.shape)
        if ori_selected_idx is not None:
            after_selected_idx = ori_selected_idx[selected_idxs]

        # done
        dist = np.linalg.norm(ag_next - g_next, axis=1)
        not_done = (dist > self.distance_threshold).astype(np.int32).reshape(-1, 1)

        # transfer them into the tensor
        obs_cur = torch.tensor(obs_cur, dtype=torch.float32).to(self.device)
        g_cur = torch.tensor(g_cur, dtype=torch.float32).to(self.device)
        obs_next = torch.tensor(obs_next, dtype=torch.float32).to(self.device)
        g_next = torch.tensor(g_next, dtype=torch.float32).to(self.device)
        ag_next = torch.tensor(ag_next, dtype=torch.float32).to(self.device)
        not_done = torch.tensor(not_done, dtype=torch.int32).to(self.device)

        selected_action = transitions['actions'][selected_idxs]
        selected_r = transitions['r'][selected_idxs]

        actions_tensor = torch.tensor(selected_action, dtype=torch.float32).to(self.device)
        r_tensor = torch.tensor(selected_r, dtype=torch.float32).to(self.device)

        # calculate the target Q value function
        with torch.no_grad():
            actions_next = actor_target(obs_next, g_next)
            q_next_value = critic_target(obs_next, g_next, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + critic_target.gamma * q_next_value * not_done
            target_q_value = target_q_value.detach()
            # the low-level Q cannot be larger than 0
            target_q_value = torch.clamp(target_q_value, max=0.)
            if clip:
                clip_return = self.env_params[T]
                target_q_value = torch.clamp(target_q_value, -clip_return, 0.)
        # the q loss
        real_q_value = critic(obs_cur, g_cur, actions_tensor)

        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # add a L2 norm loss to the critic loss
        L2_reg = torch.tensor(0., requires_grad=True).to(self.args.device)
        for name, param in critic.named_parameters():
            L2_reg = L2_reg + torch.norm(param)
        critic_loss += 0. * L2_reg

        if use_forward_loss:
            forward_loss = critic(obs_cur, ag_next, actions_tensor).pow(2).mean()
            critic_loss += forward_loss
        # the actor loss
        actions_real = actor(obs_cur, g_cur)
        actor_loss = -critic(obs_cur, g_cur, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

        # start to update the network
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_actor_network.parameters(), 1.0)
        actor_optim.step()
        # update the critic_network
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_critic_network.parameters(), 1.0)
        critic_optim.step()

        if ori_selected_idx is not None:
            # write p to the buffer
            with torch.no_grad():
                p = (target_q_value - real_q_value).pow(2).detach().cpu().numpy()
                p = p.reshape(-1, 1)
                self.low_buffer.buffers['p_low'][after_selected_idx[:, 0], after_selected_idx[:, 1]] = p

        if self.args.save and n_batch == self.args.n_batches - 1:
            if T == 'max_timesteps':
                name = 'low'
            else:
                name = 'high'
            self.writer.add_scalar('Loss/' + name + '_actor_loss' + self.args.metric, actor_loss, epoch)
            self.writer.add_scalar('Loss/' + name + '_critic_loss' + self.args.metric, critic_loss, epoch)
            with torch.no_grad():
                target_q_mean = target_q_value.mean()
                real_q_mean = real_q_value.detach().mean()
                r_mean = r_tensor.detach().mean()
                q_next_mean = q_next_value.mean()
                L2_mean = L2_reg.detach().mean()
                self.writer.add_scalar('Loss/' + name + '_target_q', target_q_mean, epoch)
                # self.writer.add_scalar('Loss/' + name + '_real_q', real_q_mean, epoch)
                self.writer.add_scalar('Loss/' + name + '_r', r_mean, epoch)
                self.writer.add_scalar('Loss/' + name + '_target_q_next', q_next_mean, epoch)
                self.writer.add_scalar('Loss/' + name + '_weights_l2', L2_mean, epoch)

    def _eval_hier_agent(self, env, n_test_rollouts=10):
        total_success_rate = []
        if not self.args.eval:
            n_test_rollouts = self.args.n_test_rollouts
        discount_reward = np.zeros(n_test_rollouts)
        for roll in range(n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if num % self.c == 0:
                        hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                        hi_action = self.hi_agent.select_action(hi_act_obs, evaluate=True)
                        if self.old_sample:
                            new_hi_action = hi_action
                        else:
                            ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                            new_hi_action = ag + hi_action
                            new_hi_action = np.clip(new_hi_action, -SUBGOAL_RANGE, SUBGOAL_RANGE)
                        hi_action_tensor = torch.tensor(new_hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                observation_new, rew, done, info = env.step(action)
                if self.animate:
                    env.render()
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                if done:
                    per_success_rate.append(info['is_success'])
                    if bool(info['is_success']):
                        discount_reward[roll] = 1 - 1. / self.env_params['max_test_timesteps'] * num
                    break
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        global_reward = np.mean(discount_reward)
        if self.args.eval:
            print("hier success rate", global_success_rate, global_reward)
        return global_success_rate, global_reward

    def init_network(self):
        self.low_actor_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_actor_target_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_critic_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)
        self.low_critic_target_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)

        self.start_epoch = 0

        # create the optimizer
        self.low_actor_optim = torch.optim.Adam(self.low_actor_network.parameters(), lr=self.args.lr_actor)
        self.low_critic_optim = torch.optim.Adam(self.low_critic_network.parameters(), lr=self.args.lr_critic, weight_decay=self.args.weight_decay)

    def sync_target(self):
        # load the weights into the target networks
        self.low_actor_target_network.load_state_dict(self.low_actor_network.state_dict())
        self.low_critic_target_network.load_state_dict(self.low_critic_network.state_dict())

    def slow_update_phi(self, epoch, update_i):
        # sample_data, hi_action = self.slow_collect()
        # prioritized sampling
        if self.p_phi_old:
            # sample_data, idxs, reg_obs = self.prioritized_collect()
            sample_data, idxs, reg_obs = self.quick_prioritized_collect()
        else:
            sample_data, idxs = self.new_prioritized_collect()
        hi_action = None
        sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        if not self.args.use_prediction:
            obs, obs_next = self.representation(sample_data[0][:, :self.hi_dim]), self.representation(sample_data[1][:, :self.hi_dim])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            hi_obs, hi_obs_next = self.representation(sample_data[2][:, :self.hi_dim]), self.representation(sample_data[3][:, :self.hi_dim])
            max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            ini_representation_loss = (min_dist + max_dist).mean()
            # add l2 regularization
            ini_representation_loss += self.feature_reg * (obs / self.abs_range).pow(2).mean()
        else:
            hi_action = torch.tensor(hi_action, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                target_next_obs = self.representation.phi(sample_data[3])
            obs, obs_next = self.representation.phi(sample_data[0]), self.representation.phi(sample_data[1])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            hi_obs, hi_obs_next = self.representation.phi(sample_data[2]), self.representation.phi(sample_data[3])
            max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            representation_loss = (min_dist + max_dist).mean()
            # prediction loss
            if self.use_prob:
                predict_distribution = self.representation(sample_data[2], hi_action)
                prediction_loss = - predict_distribution.log_prob(target_next_obs).mean()
            else:
                predict_state = self.representation(sample_data[2], hi_action)
                prediction_loss = (predict_state - target_next_obs).pow(2).mean()
            representation_loss += self.prediction_coeff * prediction_loss

        # add a regularization term for phi learning
        if self.add_reg:
            reg_obs = torch.tensor(reg_obs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                reg_feature_old = self.target_phi_reg(reg_obs[:, :self.hi_dim])
            reg_feature_new = self.representation(reg_obs[:, :self.hi_dim])
            stable_loss = (reg_feature_new - reg_feature_old).pow(2).mean()
            if epoch > self.phi_interval:
                representation_loss = stable_loss * self.stable_coeff + ini_representation_loss
            else:
                representation_loss = ini_representation_loss
        else:
            representation_loss = ini_representation_loss

        self.representation_optim.zero_grad()
        representation_loss.backward()
        self.representation_optim.step()
        if self.args.save and update_i == 20000:
            self.writer.add_scalar('Loss/phi_loss' + self.args.metric, representation_loss, epoch)
            if self.add_reg:
                self.writer.add_scalar('Loss/phi_contrastive_loss', ini_representation_loss, epoch)
                self.writer.add_scalar('Loss/phi_stable_loss', stable_loss * self.stable_coeff, epoch)
        # write p to the buffer
        with torch.no_grad():
            p = (min_dist + max_dist).detach().cpu().numpy()
            if self.p_phi_old:
                # # old prioritized
                self.p_lst.append(p)
                self.idx_lst.append(idxs)
            else:
                self.low_buffer.update_priority(idxs, p)

    def slow_collect(self, batch_size=100):
        if self.args.use_prediction:
            transitions, _ = self.low_buffer.sample(batch_size)
            obs, obs_next = transitions['obs'], transitions['obs_next']

            hi_obs, hi_action, _, hi_obs_next, _ = self.hi_buffer.sample(batch_size)
            hi_obs, hi_obs_next = hi_obs[:, :self.env_params['obs']], hi_obs_next[:, :self.env_params['obs']]
            train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
            return train_data, hi_action
        else:
            # new negative samples
            episode_num = self.low_buffer.current_size
            obs_array = self.low_buffer.buffers['obs'][:episode_num]
            episode_idxs = np.random.randint(0, episode_num, batch_size)
            t_samples = np.random.randint(self.env_params['max_timesteps'] - self.k - self.delta_k, size=batch_size)
            if self.delta_k > 0:
                delta = np.random.randint(self.delta_k, size=batch_size)
            else:
                delta = 0

            hi_obs = obs_array[episode_idxs, t_samples]
            hi_obs_next = obs_array[episode_idxs, t_samples + self.k + delta]
            obs = hi_obs
            obs_next = obs_array[episode_idxs, t_samples + 1 + delta]

            # filter data when the robot is ant
            if self.args.env_name[:3] == "Ant":
                good_index = np.where((hi_obs[:, 2] >= 0.3) & (hi_obs_next[:, 2] >= 0.3) & (obs_next[:, 2] >= 0.3))[0]
                hi_obs = hi_obs[good_index]
                hi_obs_next = hi_obs_next[good_index]
                obs = hi_obs
                obs_next = obs_next[good_index]
                assert len(hi_obs) == len(hi_obs_next) == len(obs_next)

            train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
            return train_data, None

    def prioritized_collect(self, batch_size=100):
        # new negative samples
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]

        candidate_idxs = self.candidate_idxs[:episode_num * (self.low_buffer.T - self.k + 1)]
        p = self.low_buffer.get_all_data()['p']
        p = p[:, :self.low_buffer.T - self.k + 1]
        p = p.reshape(-1)
        p_old = p / p.sum()
        selected = np.random.choice(len(candidate_idxs), size=batch_size, replace=False, p=p_old)
        if self.add_reg:
            # select the regularization data
            p_new = 1. / np.sqrt(1 + p)
            p_new_norm = p_new / p_new.sum()
            selected_new = np.random.choice(len(candidate_idxs), size=batch_size, replace=False, p=p_new_norm)
            selected_idx_new = candidate_idxs[selected_new]
            episode_idxs_new = selected_idx_new[:, 0]
            t_samples_new = selected_idx_new[:, 1]
            reg_obs = obs_array[episode_idxs_new, t_samples_new]
        else:
            reg_obs = None

        selected_idx = candidate_idxs[selected]
        episode_idxs = selected_idx[:, 0]
        t_samples = selected_idx[:, 1]

        hi_obs = obs_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1]

        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        return train_data, selected_idx, reg_obs

    def quick_prioritized_collect(self, batch_size=100):
        # new negative samples
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]

        random_index = np.random.randint(len(self.high_p), size=batch_size)
        selected = self.high_p[random_index]
        if self.add_reg:
            random_index_new = np.random.randint(len(self.low_p), size=batch_size)
            selected_new = self.low_p[random_index_new]
            selected_idx_new = self.cur_candidate_idxs[selected_new]
            episode_idxs_new = selected_idx_new[:, 0]
            t_samples_new = selected_idx_new[:, 1]
            reg_obs = obs_array[episode_idxs_new, t_samples_new]
        else:
            reg_obs = None

        selected_idx = self.cur_candidate_idxs[selected]
        episode_idxs = selected_idx[:, 0]
        t_samples = selected_idx[:, 1]

        hi_obs = obs_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1]

        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        return train_data, selected_idx, reg_obs

    def new_prioritized_collect(self, batch_size=100):
        # new negative samples
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]

        candidate_idxs = self.candidate_idxs[:episode_num * (self.low_buffer.T - self.k + 1)]
        # sample in replay buffer
        selected = self.low_buffer._sample_for_phi(batch_size)
        selected_idx = candidate_idxs[selected]
        episode_idxs = selected_idx[:, 0]
        t_samples = selected_idx[:, 1]

        hi_obs = obs_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1]

        # filter data when the robot is ant
        if self.args.env_name[:3] == "Ant":
            good_index = np.where((hi_obs[:, 2] >= 0.3) & (hi_obs_next[:, 2] >= 0.3) & (obs_next[:, 2] >= 0.3))[0]
            hi_obs = hi_obs[good_index]
            hi_obs_next = hi_obs_next[good_index]
            obs = hi_obs
            obs_next = obs_next[good_index]
            selected_idx = selected_idx[good_index]
            selected = selected[good_index]
            assert len(hi_obs) == len(hi_obs_next) == len(obs_next) == len(selected_idx)

        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        return train_data, selected

    def visualize_representation(self, epoch):
        transitions, _ = self.low_buffer.sample(800)
        obs = transitions['obs']

        index1 = np.where((obs[:, 0] < 4) & (obs[:, 1] < 4))
        index2 = np.where((obs[:, 0] < 4) & (obs[:, 1] > 4))
        index3 = np.where((obs[:, 0] > 4) & (obs[:, 1] < 4))
        index4 = np.where((obs[:, 0] > 4) & (obs[:, 1] > 4))
        index_lst = [index1, index2, index3, index4]

        obs_tensor = torch.Tensor(obs).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        plt.scatter(features[:, 0], features[:, 1], color='green')
        plt.show()

    def plot_fig(self, rep, name, epoch):
        fig = plt.figure()
        axes = fig.add_subplot(111)
        rep1, rep2, rep3, rep4 = rep
        def scatter_rep(rep1, c, marker):
            if rep1.shape[0] > 0:
                l1 = axes.scatter(rep1[:, 0], rep1[:, 1], c=c, marker=marker)
            else:
                l1 = axes.scatter([], [], c=c, marker=marker)
            return l1
        l1 = scatter_rep(rep1, c='y', marker='s')
        l2 = scatter_rep(rep2, c='r', marker='o')
        l3 = scatter_rep(rep3, c='b', marker='1')
        l4 = scatter_rep(rep4, c='g', marker='2')

        plt.xlabel('x')
        plt.ylabel('y')
        axes.legend((l1, l2, l3, l4), ('space1', 'space2', 'space3', 'space4'))
        plt.savefig('fig/final/' + name + str(epoch) + '.png')
        plt.close()

    # evaluate the agent and show the track
    def vis_hier_policy(self, epoch=0, load_obs=None, path=None, representation=None):
        obs_vec = []
        hi_action_vec = []
        env = self.test_env
        observation = env.reset()
        obs = observation['observation']
        ag_record = observation['achieved_goal']
        obs_vec.append(obs.copy())
        if representation is None:
            representation = self.representation
        if self.args.image:
            obs_vec[-1][:2] = ag_record
        g = observation['desired_goal']
        if load_obs is None:
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if num % self.c == 0:
                        hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                        hi_action = self.hi_agent.select_action(hi_act_obs, evaluate=True)
                        if self.old_sample:
                            new_hi_action = hi_action
                        else:
                            ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                            new_hi_action = ag + hi_action  # subgoal in representation space
                            new_hi_action = np.clip(new_hi_action, -SUBGOAL_RANGE, SUBGOAL_RANGE)
                        hi_action_tensor = torch.tensor(new_hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                        hi_action_vec.append(hi_action)
                    action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                observation_new, rew, done, info = env.step(action)
                if self.animate:
                    env.render()
                obs = observation_new['observation']
                ag_record = observation_new['achieved_goal']
                obs_vec.append(obs.copy())
                if self.args.image:
                    obs_vec[-1][:2] = ag_record
                if done:
                    if info['is_success']:
                        print("success !!!")
                    break
        else:
            obs_vec = load_obs

        plt.figure(figsize=(12, 6))
        obs_vec = np.array(obs_vec)
        with open('fig/final/' + "img_maze_scale4_5.pkl", 'wb') as output:
            pickle.dump(obs_vec, output)
        self.plot_rollout(obs_vec, "XY_{}".format(epoch * self.env_params['max_timesteps']), 121, goal=g)

        if not self.learn_goal_space:
            features = obs_vec[:, :2]
            feature_goal = g[:2]
        else:
            if self.args.image:
                obs_vec[:, :2] = 0.
            obs_tensor = torch.Tensor(obs_vec[:, :self.hi_dim]).to(self.device)
            features = representation(obs_tensor).detach().cpu().numpy()
            rest = (self.env_params['obs'] - self.env_params['goal']) * [0.]
            g = np.concatenate((g, np.array(rest)))
            g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
            feature_goal = representation(g).detach().cpu().numpy()[0]
        hi_action_vec = np.array(hi_action_vec)
        if load_obs is None:
            self.plot_rollout(features, "Feature_{}".format(epoch * self.env_params['max_timesteps']), 122, feature_goal, hi_action_vec)
        else:
            self.plot_rollout(features, "Feature_{}".format(epoch * self.env_params['max_timesteps']), 122, use_lim=False)
        if path is None:
            file_name = 'fig/round2/rollout' + str(epoch) + '.png'
        else:
            file_name = path + '/rollout_' + str(epoch) + '.png'
        plt.savefig(file_name, bbox_inches='tight', transparent=False)
        # plt.show()
        plt.close()


    def plot_rollout(self, obs_vec, name, num, goal=None, hi_action_vec=None, no_axis=False, use_lim=False, fig=None):
        if fig is None:
            plt.subplot(num)
            cm = plt.cm.get_cmap('RdYlBu')
            num = np.arange(obs_vec.shape[0])
            plt.scatter(obs_vec[:, 0], obs_vec[:, 1], c=num, cmap=cm)
        else:
            ax = fig.add_subplot(1, 2, num)
            hi_horizon = int(obs_vec.shape[0] / self.args.c)
            cm = plt.cm.get_cmap('RdYlBu')
            num = np.arange(obs_vec.shape[0])
            plt.scatter(obs_vec[:, 0], obs_vec[:, 1], c=num, cmap=cm)
            for num in range(1, hi_horizon):
                ax.text(obs_vec[num*self.c, 0], obs_vec[num*self.c, 1], str(num), fontsize=10, color='k', alpha=0.8)
                plt.scatter(obs_vec[num*self.c, 0], obs_vec[num*self.c, 1], c='k', s=30)

        if goal is not None:
            plt.scatter([goal[0]], [goal[1]], marker='*',
                        color='green', s=200, label='goal')
        if hi_action_vec is not None:
            for num in range(len(hi_action_vec)):
                plt.scatter(hi_action_vec[num, 0], hi_action_vec[num, 1], c=self.color_set[num % len(self.color_set)], s=30)
                # ax.text(hi_action_vec[num, 0], hi_action_vec[num, 1], str(num+1), fontsize=10, color='#FF00FF', alpha=0.8)
        plt.title(name, fontsize=24)

        if use_lim:
            plt.ylim(-5, 25)
            plt.xlim(-30, 10)

        if not no_axis:
            plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                        color='green', s=200, label='start')
            plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                        color='red', s=200, label='end')
            plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), fontsize=14, borderaxespad=0.)


    def same_data_compare(self):
        with open('fig/final/' + "sampled_raw_states.pkl", 'rb') as output:
            Obs = pickle.load(output)

        features = []
        for obs in Obs:
            obs_tensor = torch.Tensor(obs[:, :29]).to(self.device)
            feature = self.representation(obs_tensor).detach().cpu().numpy()
            features.append(feature)

        self.plot_fig(Obs, 'obs', "raw")
        self.plot_fig(features, 'slow_feature', 'slow')

        with open('fig/final/' + "oracle_trajectory.pkl", 'rb') as output:
            obs_vec = pickle.load(output)

        xy = obs_vec[:, :2]
        dists = np.linalg.norm(xy - [[0, 8]], axis=1)
        success = np.argwhere(dists < 1.5)
        min_success = min(success)[0]
        print("min_success", min_success)
        print("min_dist", dists[min_success])

        plt.figure(figsize=(12, 6))
        self.plot_rollout(obs_vec[:min_success], "XY", 121, no_axis=True)

        obs_tensor = torch.Tensor(obs_vec[:min_success, :29]).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        rest = (self.env_params['obs'] - self.real_goal_dim) * [0.]
        g = np.concatenate(([0, 8], np.array(rest)))
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        feature_goal = self.representation(g).detach().cpu().numpy()[0]
        self.plot_rollout(features, "slow feature", 122, no_axis=True)

        file_name = 'fig/final/rollout_compare.png'
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    def picvideo(self):

        filenames = []
        for i in range(50, 5000, 50):
            filename = "fig/check_edge/rollout_{}.png".format(i)
            filenames.append(filename)

        frames = []
        for image_name in filenames:
            frames.append(imageio.imread(image_name))

        # druation: switching time of figure, the unit is second
        imageio.mimsave('fig/final/edge.gif', frames, 'GIF', duration=0.3)

    def cal_stable(self):
        transitions, _ = self.low_buffer.sample(100)
        obs = transitions['obs']

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                       '/phi_model_3000.pt', map_location='cuda:1')[0])

        feature1 = self.representation(obs).detach().cpu().numpy()

        self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                       '/phi_model_4000.pt', map_location='cuda:1')[0])

        feature2 = self.representation(obs).detach().cpu().numpy()
        distance = np.linalg.norm(feature1 - feature2)
        print("distance", distance)

    def plot_chain(self):
        state = np.eye(40)
        obs_tensor = torch.Tensor(state).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        plt.plot(features)
        plt.show()

    def cal_slow(self):
        num = 5
        record = np.zeros(num)
        state_lst = []
        for i in range(num):
            sample_data, hi_action = self.slow_collect()
            state_lst.append(sample_data[:2])
            print("sample", sample_data[:2].shape)
            sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
            obs, obs_next = self.representation(sample_data[0]), self.representation(sample_data[1])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            mean_dist = np.mean(min_dist.detach().cpu().numpy())
            record[i] = mean_dist

            print("mean_dist", mean_dist)

        arr_mean = np.mean(record)
        arr_std = np.std(record, ddof=1)
        print(self.args.env_name, arr_mean, arr_std)

        with open('fig/final/' + "img_push_states.pkl", 'wb') as output:
            pickle.dump(state_lst, output)

    def cal_random_slow(self):
        num = 5
        record = np.zeros(num)
        print("hi", self.hi_dim)

        for i in range(num):
            select_dim = np.random.randint(self.hi_dim, size=2)
            sample_data, hi_action = self.slow_collect()

            obs = sample_data[0][:, select_dim]
            obs_next = sample_data[1][:, select_dim]
            print("new", obs.shape)

            min_dist = np.square(obs - obs_next)
            mean_dist = np.mean(min_dist)
            record[i] = mean_dist

            print("mean_dist", mean_dist)

        arr_mean = np.mean(record)
        arr_std = np.std(record, ddof=1)
        print(self.args.env_name, arr_mean, arr_std)


    def select_by_count(self, hi_obs, t, epoch):
        transitions, _ = self.low_buffer.sample(1000)
        obs, ag_record = transitions['obs'], transitions['ag_record'][:, :self.real_goal_dim]
        obs_tensor = torch.Tensor(obs).to(self.device)
        features = self.representation(obs_tensor[:, :self.hi_dim]).detach().cpu().numpy()

        # current state
        hi_obs_tensor = torch.Tensor(hi_obs).to(self.device)
        hi_feature = self.representation(hi_obs_tensor).detach().cpu().numpy()

        distances = np.linalg.norm(features - hi_feature, axis=-1)
        near_indexs = np.where((distances < 20) & (distances > self.min_dist))[0]

        new_features = features[near_indexs]
        new_ag_record = ag_record[near_indexs]
        if self.future_count_coeff == 0.:
            count = np.array(self.hash.predict(new_features))
        else:
            count = np.array(self.future_hash.predict(new_features)) * self.future_count_coeff
        distances_new = distances[near_indexs]
        obs_new = obs[near_indexs]
        if len(count) == 0:
            # no nearby feature
            subgoal = features[0]
            xy_select = ag_record[0]
            obs_select = obs[0]
        else:
            # select subgoal with less count and larger distance
            score = count + 1. / np.sqrt(distances_new + 1) * self.distance_coeff

            if self.history_subgoal_coeff != 0:
                # select the subgoal that rarely selected in the past
                count_history = np.array(self.subgoal_hash.predict(new_features))
                score += count_history * self.history_subgoal_coeff

            # select the subgoal that can success
            if epoch > self.start_count_success:
                dis_to_goal = np.array(self.success_hash.predict(new_features))
                score += dis_to_goal * self.success_coeff

            min_index = score.argmin()
            subgoal = new_features[min_index]
            xy_select = new_ag_record[min_index]
            obs_select = obs_new[min_index]

        current_hi_step = int(t / self.c)
        self.count_xy_record[current_hi_step].append(xy_select)
        self.subgoal_record[current_hi_step].append(subgoal)
        self.valid_times += 1

        # record all history subgoal
        if self.history_subgoal_coeff != 0:
            self.subgoal_xy_hash.inc_hash(xy_select.copy()[None])
            self.subgoal_hash.inc_hash(subgoal.copy()[None])
            self.all_history_xy.append(xy_select)
            if self.usual_update_history:
                self.all_history_obs.append(obs_select)
            else:
                self.all_history_subgoal.append(subgoal)

        return subgoal, xy_select


    def cal_fall_over(self):
        self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')
        print("load buffer !!!")
        transitions, _ = self.low_buffer.sample(100000)
        obs, ag_record = transitions['obs'], transitions['ag_record']
        select_index = np.where((obs[:, 0] < -1.0) & (obs[:, 1] < -1.0))[0]
        print("select", len(select_index))
        total_count = len(select_index)
        obs_new = obs[select_index]
        not_fall = np.where((obs_new[:, 2] >= 0.3) & (obs_new[:, 2] <= 1.0))[0]
        smaller = np.where(obs_new[:, 2] < 0.3)[0]
        larger = np.where(obs_new[:, 2] > 1.0)[0]
        print("not fall over", len(not_fall))
        not_fall_rate = len(not_fall) / total_count
        print("not fall rate", not_fall_rate)
        print("smaller", len(smaller) / total_count)
        print("larger", len(larger) / total_count)
        print("#" * 20)

        # new negative samples
        batch_size = 10000
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]
        episode_idxs = np.random.randint(0, episode_num, batch_size)
        t_samples = np.random.randint(self.env_params['max_timesteps'] - self.k - self.delta_k, size=batch_size)
        if self.delta_k > 0:
            delta = np.random.randint(self.delta_k, size=batch_size)
        else:
            delta = 0

        hi_obs = obs_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k + delta]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1 + delta]

        # filter data when the robot is ant
        if self.args.env_name[:3] == "Ant":
            good_index = np.where((hi_obs[:, 2] >= 0.3) & (hi_obs_next[:, 2] >= 0.3) & (obs_next[:, 2] >= 0.3))[0]
            hi_obs = hi_obs[good_index]
            hi_obs_next = hi_obs_next[good_index]
            obs = hi_obs
            obs_next = obs_next[good_index]

            select_index = np.where((hi_obs[:, 0] < -1.0) & (hi_obs[:, 1] < -1.0) & (hi_obs_next[:, 0] < -1.0) & (hi_obs_next[:, 1] < -1.0) & \
                                    (obs_next[:, 0] < -1.0) & (obs_next[:, 1] < -1.0))[0]
            hi_obs = hi_obs[select_index]
            hi_obs_next = hi_obs_next[select_index]
            obs = hi_obs
            obs_next = obs_next[select_index]

            c_change = np.linalg.norm(hi_obs - hi_obs_next, axis=1)
            one_change = np.linalg.norm(obs - obs_next, axis=1)
            delta_change = c_change - one_change
            print("c_change", len(c_change))
            print("len_obs", len(hi_obs))
            x_change = np.linalg.norm(hi_obs[:, 0:2] - hi_obs_next[:, 0:2], axis=1) - np.linalg.norm(obs[:, 0:2] - obs_next[:, 0:2], axis=1)
            print("delta_change", delta_change)
            print("x_change", x_change)


        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])


    def edge_representation(self):
        self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')
        print("load buffer !!!")
        transitions, _ = self.low_buffer.sample(2000)
        obs, ag_record = transitions['obs'], transitions['ag_record']
        select_index = np.where(obs[:, 1] < -1.0)[0]
        print("selected num", len(select_index))

        obs_new = obs[select_index]
        # obs_new = obs_new[obs_new[:, 0].argsort()]

        # fix the xy position, only consider joints
        obs_new[:, :2] = obs_new[0, :2]
        # print("obs_new", obs_new)

        for i in range(50, 5000, 50):
            self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                           '/phi_model_{}.pt'.format(i), map_location='cuda:1')[0])
            self.pruned_phi = copy.deepcopy(self.representation)
            # pruning phi
            for name, module in self.pruned_phi.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.8)
            self.vis_hier_policy(epoch=i, load_obs=obs_new, path="fig/check_edge", representation=self.pruned_phi)

    def plot_density(self):
        self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')
        self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                       '/phi_model_{}.pt'.format(5000), map_location='cuda:1')[0])

        # pruning phi
        for name, module in self.representation.named_modules():
            print("name", name)
            print("module", module)
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.8)

        print(dict(self.representation.named_buffers()).keys())
        print("load buffer and phi !!!")

        state = self.low_buffer.get_all_data()['obs']
        state = state.reshape(-1, state.shape[2])
        obs_tensor = torch.Tensor(state[:, :self.hi_dim]).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        self.hash.inc_hash(features)


        transitions, _ = self.low_buffer.sample(500)
        state = transitions['obs']
        state = np.array(state)
        obs_tensor = torch.Tensor(state[:, :self.hi_dim]).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        count_xy = np.array(self.hash.predict(features)).astype(int)

        ax = plt.subplot(111, projection='3d')
        cm = plt.cm.get_cmap('RdYlBu')
        ax.scatter(features[:, 0], features[:, 1], count_xy, c=count_xy, cmap=cm)

        plt.show()

        bx = plt.subplot(111)
        bx.scatter(features[:, 0], features[:, 1], c=count_xy, cmap=cm)
        plt.show()

    def cal_phi_loss(self):
        self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')
        self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                       '/phi_model_{}.pt'.format(3000), map_location='cuda:1')[0])

        print("load buffer and phi !!!")

        # new negative samples
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]
        ag_record_array = self.low_buffer.buffers['ag_record'][:episode_num]

        candidate_idxs = np.array([[i, j] for i in range(episode_num) for j in range(self.low_buffer.T - self.k + 1)])
        p = np.ones(len(candidate_idxs)) * 1. / len(candidate_idxs)
        selected = np.random.choice(len(candidate_idxs), size=500000, replace=False, p=p)
        print("origin selected", len(selected))
        selected_idx = candidate_idxs[selected]
        episode_idxs = selected_idx[:, 0]
        t_samples = selected_idx[:, 1]

        hi_obs = obs_array[episode_idxs, t_samples]
        ag_record = ag_record_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1]

        # filter data when the robot is ant
        if self.args.env_name[:3] == "Ant":
            good_index = np.where((hi_obs[:, 2] >= 0.3) & (hi_obs_next[:, 2] >= 0.3) & (obs_next[:, 2] >= 0.3))[0]
            hi_obs = hi_obs[good_index]
            hi_obs_next = hi_obs_next[good_index]
            obs = hi_obs
            obs_next = obs_next[good_index]
            assert len(hi_obs) == len(hi_obs_next) == len(obs_next)
            selected_idx = selected_idx[good_index]
            print("selected data", len(good_index))

        sample_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        obs, obs_next = self.representation(sample_data[0]), self.representation(sample_data[1])
        min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
        hi_obs, hi_obs_next = self.representation(sample_data[2]), self.representation(sample_data[3])
        max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
        print("loss", max_dist + min_dist)
        representation_loss = (min_dist + max_dist).mean()
        print("phi loss near start: ", representation_loss)


