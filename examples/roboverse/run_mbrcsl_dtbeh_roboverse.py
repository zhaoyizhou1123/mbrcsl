# Behavior policy ablation study

import numpy as np
import torch
import roboverse

import argparse
import os
import random
import pickle
from copy import deepcopy
from typing import Dict, Tuple
from collections import defaultdict
import datetime

from offlinerlkit.modules import TransformerDynamicsModel, RcslModule
from offlinerlkit.dynamics import TransformerDynamics
from offlinerlkit.utils.roboverse_utils import PickPlaceObsWrapper, DoubleDrawerObsWrapper, \
    get_pickplace_dataset_dt, get_doubledrawer_dataset_dt, \
    get_pickplace_dataset, get_doubledrawer_dataset
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import RcslPolicyTrainer, DiffusionPolicyTrainer
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.policy import SimpleDiffusionPolicy, AutoregressivePolicy, RcslPolicy
from offlinerlkit.nets import MLP
from offlinerlkit.utils.dataset import TrajCtxFloatLengthDataset
from offlinerlkit.policy import DecisionTransformer
from offlinerlkit.policy_trainer import SequenceTrainer, TrainerConfig

'''
Recommended hyperparameters:
pickplace, horizon=40, behavior_epoch=30
doubledraweropen, horizon=50, behavior_epoch=40
doubledrawercloseopen, horizon=80, behavior_epoch=40
'''

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--algo-name", type=str, default="mbrcsl_dtbeh")
    parser.add_argument("--task", type=str, default="pickplace", help="task name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader workers, align with cpu number")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--last_eval", action="store_false")

    # env config
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=40, help="max path length for pickplace")

    # transformer_autoregressive dynamics
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=32)
    parser.add_argument("--dynamics_lr", type=float, default=1e-3)
    parser.add_argument("--dynamics_epoch", type=int, default=80)
    parser.add_argument("--load_dynamics_path", type=none_or_str, default=None)

    # Behavior policy
    parser.add_argument("--behavior_epoch", type=int, default=30)
    parser.add_argument('--behavior_batch', type=int, default=256)
    parser.add_argument('--load_diffusion_path', type=none_or_str, default=None)
    parser.add_argument('--task_weight', type=float, default=1.4, help="Weight on task data when training diffusion policy")
    parser.add_argument('--sample_ratio', type=float, default=0.8, help="Use (sample_ratio * num_total_data) data to train diffusion policy")
    parser.add_argument("--behavior_lr", type=float, default=1e-3)
    parser.add_argument("--behavior_weight_decay", type=float, default=0.1)
    # parser.add_argument("--n_layer", type=int, default=4)
    # parser.add_argument("--n_head", type=int, default=4)
    # parser.add_argument("--n_embd", type=int, default=32)
    parser.add_argument('--ctx', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=128, help="dt token embedding dimension")
    parser.add_argument('--goal_mul', type=float, default=1., help="goal = max_dataset_return * goal_mul")
    
    # Rollout 
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="file dir, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epoch', type=int, default=1000, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=5000, help="Needed valid trajs in rollout")
    parser.add_argument("--rollout-batch", type=int, default=200, help="Number of trajs to be sampled at one time")

    # RCSL policy (mlp)
    parser.add_argument("--rcsl_hidden_dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--rcsl_lr", type=float, default=1e-3)
    parser.add_argument("--rcsl_batch", type=int, default=256)
    parser.add_argument("--rcsl_epoch", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--holdout_ratio", type=float, default=0.2)

    return parser.parse_args()

def rollout_simple(
    init_obss: np.ndarray,
    dynamics: TransformerDynamics,
    rollout_policy: SimpleDiffusionPolicy,
    rollout_length: int
) -> Tuple[Dict[str, np.ndarray], Dict]:
    '''
    Only serves for non-terminal cases
    Sample a batch of trajectories at the same time.
    Output rollout_transitions contain keys:
    obss,
    next_obss,
    actions
    rewards, (N,1)
    rtgs, (N,1)
    traj_idxs, (N)
    '''
    num_transitions = 0
    rewards_arr = np.array([])
    rollout_transitions = defaultdict(list)
    batch_size = init_obss.shape[0]
    valid_idxs = np.arange(init_obss.shape[0]) # maintain current valid trajectory indexes
    returns = np.zeros(init_obss.shape[0]) # maintain return of each trajectory
    acc_returns = np.zeros(init_obss.shape[0]) # maintain accumulated return of each valid trajectory
    max_rewards = np.zeros(init_obss.shape[0]) # maintain max reward seen in trajectory
    rewards_full = np.zeros((init_obss.shape[0], rollout_length)) # full rewards (batch, H)

    # rollout
    observations = init_obss
    goal = np.zeros((init_obss.shape[0],1), dtype = np.float32)
    for t in range(rollout_length):
        actions = rollout_policy.select_action(observations, goal)
        next_observations, rewards, terminals, info = dynamics.step(observations, actions)
        rollout_transitions["observations"].append(observations)
        rollout_transitions["next_observations"].append(next_observations)
        rollout_transitions["actions"].append(actions)
        rollout_transitions["rewards"].append(rewards)
        rollout_transitions["terminals"].append(terminals)
        rollout_transitions["traj_idxs"].append(valid_idxs)
        rollout_transitions["acc_rets"].append(acc_returns)

        rewards = rewards.reshape(batch_size) # (B)
        rewards_full[:, t] = rewards

        num_transitions += len(observations)
        rewards_arr = np.append(rewards_arr, rewards.flatten())
        returns = returns + rewards.flatten() # Update return (for valid idxs only)
        max_rewards = np.maximum(max_rewards, rewards.flatten()) # Update max reward
        acc_returns = acc_returns + rewards.flatten()
        observations = deepcopy(next_observations)
    
    for k, v in rollout_transitions.items():
        rollout_transitions[k] = np.concatenate(v, axis=0)

    traj_idxs = rollout_transitions["traj_idxs"]
    rtgs = returns[traj_idxs] - rollout_transitions["acc_rets"]
    # rtgs = returns[traj_idxs] 
    rollout_transitions["rtgs"] = rtgs[..., None] # (N,1)

    return rollout_transitions, \
        {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean(), "returns": returns, "max_rewards": max_rewards, "rewards_full": rewards_full}

def rollout(    
    init_obss: np.ndarray,
    dynamics: TransformerDynamics,
    rollout_policy: DecisionTransformer,
    rollout_length: int,
    state_mean,
    state_std,
    device,
    action_dim,
    ctx):
    '''
    state_mean/std: Used for state normalization. Get from offline_dataset only currently
    '''
    # state_mean, state_std = self.offline_dataset.get_normalize_coef()
    rollout_policy.train(False)
    rets = [] # list of returns achieved in each epoch
    batch = init_obss.shape[0]
    states = torch.from_numpy(init_obss)
    states = states.type(torch.float32).to(device).unsqueeze(1) # (b,1,state_dim)
    rtgs = torch.zeros(batch, 1, 1).to(device) # (b,1,1)
    timesteps = torch.zeros(batch,1).to(device) # (b,1)
    
    # Initialize action
    actions = torch.empty((batch,0,action_dim)).to(device) # Actions are represented in one-hot

    num_transitions = 0
    rewards_arr = np.array([])
    rollout_transitions = defaultdict(list)
    valid_idxs = np.arange(init_obss.shape[0]) # maintain current valid trajectory indexes
    returns = np.zeros(init_obss.shape[0]) # maintain return of each trajectory
    acc_returns = np.zeros(init_obss.shape[0]) # maintain accumulated return of each valid trajectory
    max_rewards = np.zeros(init_obss.shape[0]) # maintain max reward seen in trajectory
    rewards_full = np.zeros((init_obss.shape[0], rollout_length)) # full rewards (batch, H)

    for h in range(rollout_length):
        # Get action
        pred_action = rollout_policy.select_action((states - state_mean) / state_std,
                                                actions.type(torch.float32),
                                                rtgs.type(torch.float32),
                                                timesteps.type(torch.float32)) # (b, act_dim)

        # Observe next states, rewards,
        # if self.is_gym:
        #     next_state, reward, terminated, _ = env.step(pred_action.detach().cpu().numpy()) # (state_dim), scalar
        # else:
        #     next_state, reward, terminated, _, _ = env.step(pred_action.detach().cpu().numpy()) # (state_dim), scalar
        next_state, reward, terminal, _ = dynamics.step(states[:, -1, :].detach().cpu().numpy(), pred_action.detach().cpu().numpy()) # (batch, )
        rollout_transitions["observations"].append(states[:,0,:].detach().cpu().numpy())
        rollout_transitions["next_observations"].append(next_state)
        rollout_transitions["actions"].append(pred_action.detach().cpu().numpy())
        rollout_transitions["rewards"].append(reward)
        rollout_transitions["terminals"].append(terminal)
        rollout_transitions["traj_idxs"].append(valid_idxs)
        rollout_transitions["acc_rets"].append(acc_returns)

        reward = reward.reshape(batch) # (B)
        rewards_full[:, h] = reward

        num_transitions += len(next_state)
        rewards_arr = np.append(rewards_arr, reward.flatten())
        returns = returns + reward.flatten() # Update return (for valid idxs only)
        max_rewards = np.maximum(max_rewards, reward.flatten()) # Update max reward
        acc_returns = acc_returns + reward.flatten()

        next_state = torch.from_numpy(next_state)
        # Calculate return
        # returns += reward
        
        # Update states, actions, rtgs, timesteps
        next_state = next_state.unsqueeze(1).to(device) # (b,1,state_dim)
        states = torch.cat([states, next_state], dim=1)
        states = states[:, -ctx: , :] # truncate to ctx_length

        pred_action = pred_action.unsqueeze(1).to(device) # (b, 1, action_dim)
        
        if ctx > 1:
            actions = torch.cat([actions, pred_action], dim=1)
            actions = actions[:, -ctx+1: , :] # actions length is ctx-1

        next_rtg = torch.zeros(batch,1,1).to(device)
        rtgs = torch.cat([rtgs, next_rtg], dim=1)
        rtgs = rtgs[:, -ctx: , :]

        # Update timesteps
        timesteps = torch.cat([timesteps, (h+1)*torch.ones(batch,1).to(device)], dim = 1) 
        timesteps = timesteps[:, -ctx: ]

    for k, v in rollout_transitions.items():
        rollout_transitions[k] = np.concatenate(v, axis=0)

    traj_idxs = rollout_transitions["traj_idxs"]
    final_rtgs = returns[traj_idxs] - rollout_transitions["acc_rets"]
    # rtgs = returns[traj_idxs] 
    rollout_transitions["rtgs"] = final_rtgs[..., None] # (N,1)

    return rollout_transitions, \
        {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean(), "returns": returns, "max_rewards": max_rewards, "rewards_full": rewards_full}

def train(args=get_args()):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # create env and dataset
    if args.task == 'pickplace':
        env = roboverse.make('Widow250PickTray-v0')
        env = PickPlaceObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)

        prior_data_path = os.path.join(args.data_dir, "pickplace_prior.npy")
        task_data_path = os.path.join(args.data_dir, "pickplace_task.npy")


        prior_data_path = os.path.join(args.data_dir, "pickplace_prior.npy")
        task_data_path = os.path.join(args.data_dir, "pickplace_task.npy")

        trajs = get_pickplace_dataset_dt(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
        
        dyn_dataset, init_obss_dataset = get_pickplace_dataset(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
    elif args.task == 'doubledraweropen':
        env = roboverse.make('Widow250DoubleDrawerOpenGraspNeutral-v0')
        env = DoubleDrawerObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)

        prior_data_path = os.path.join(args.data_dir, "closed_drawer_prior.npy")
        task_data_path = os.path.join(args.data_dir, "drawer_task.npy")

        trajs = get_doubledrawer_dataset_dt(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
        dyn_dataset, init_obss_dataset = get_doubledrawer_dataset(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
    elif args.task == 'doubledrawercloseopen':
        env = roboverse.make('Widow250DoubleDrawerCloseOpenGraspNeutral-v0')
        env = DoubleDrawerObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)

        prior_data_path = os.path.join(args.data_dir, "blocked_drawer_1_prior.npy")
        task_data_path = os.path.join(args.data_dir, "drawer_task.npy")

        trajs = get_doubledrawer_dataset_dt(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
        dyn_dataset, init_obss_dataset = get_doubledrawer_dataset(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
    elif args.task == 'doubledrawerpickplaceopen':
        env = roboverse.make('Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0')
        env = DoubleDrawerObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)

        prior_data_path = os.path.join(args.data_dir, "blocked_drawer_2_prior.npy")
        task_data_path = os.path.join(args.data_dir, "drawer_task.npy")

        trajs = get_doubledrawer_dataset_dt(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
        dyn_dataset, init_obss_dataset = get_doubledrawer_dataset(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
    else:
        raise NotImplementedError

    env.reset(seed=args.seed)
    behavior_dataset = TrajCtxFloatLengthDataset(trajs, ctx = args.ctx, single_timestep = False, with_mask=True)
    # goal = behavior_dataset.get_max_return() * args.goal_mul
    goal = 0

    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    exp_name = f"timestamp_{timestamp}&{args.seed}"
    log_dirs = make_log_dirs(args.task, args.algo_name, exp_name, vars(args), part = "dynamics")
    # key: output file name, value: output handler type
    print(f"Logging dynamics to {log_dirs}")
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    dynamics_model = TransformerDynamicsModel(
        obs_dim=obs_dim,
        act_dim=action_dim,
        obs_min = -1,
        obs_max = 1,
        act_min = -1,
        act_max = 1,
        r_min = 0,
        r_max = 1,
        ckpt_dir = logger.checkpoint_dir,
        device = args.device,
        n_layer = args.n_layer,
        n_head = args.n_head,
        n_embd = args.n_embd
    )
    
    dynamics_optim = dynamics_model.configure_optimizer(
        lr = args.dynamics_lr,
        weight_decay= 0. ,
        betas = (0.9, 0.999)
    )
    dynamics = TransformerDynamics(
        dynamics_model,
        dynamics_optim,
    )

    # create rollout policy
    behavior_policy = DecisionTransformer(
        state_dim=obs_dim,
        act_dim=action_dim,
        max_length=args.ctx,
        max_ep_len=args.horizon,
        action_tanh=False, # no tanh function
        hidden_size=args.embed_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4*args.embed_dim,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1)

    lr_scheduler=None

    behavior_log_dirs = make_log_dirs(args.task, args.algo_name, exp_name, vars(args), part="behavior_trans")
    print(f"Logging behavior to {behavior_log_dirs}")
    # key: output file name, value: output handler type
    behavior_output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    behavior_logger = Logger(behavior_log_dirs, behavior_output_config)
    behavior_logger.log_hyperparameters(vars(args))

    tconf = TrainerConfig(
                max_epochs=args.behavior_epoch, batch_size=args.behavior_batch, lr=args.behavior_lr,
                lr_decay=True, num_workers=1, horizon=args.horizon, 
                grad_norm_clip = 1.0, eval_repeat = 1, desired_rtg=goal, 
                env = env, ctx = args.ctx, device=args.device, 
                debug = False, logger = behavior_logger, last_eval = True)
    behavior_policy_trainer = SequenceTrainer(
        config=tconf,
        model=behavior_policy,
        offline_dataset=behavior_dataset,
        is_gym = True)

    # Training helper functions
    def get_dynamics():
        '''
        Load or train dynamics model
        '''
        if args.load_dynamics_path:
            print(f"Load dynamics from {args.load_dynamics_path}")
            dynamics.load(args.load_dynamics_path)
        else: 
            print(f"Train dynamics")
            dynamics.train(dyn_dataset, logger, max_epochs=args.dynamics_epoch)
        
    def get_rollout_policy():
        '''
        Load or train rollout policy

        Return:
            rollout policy
        '''
        if args.load_diffusion_path is not None:
            print(f"Load behavior policy from {args.load_diffusion_path}")
            with open(args.load_diffusion_path, 'rb') as f:
                behavior_policy_model = torch.load(f, map_location= args.device)
            behavior_policy.load_state_dict(behavior_policy_model.state_dict())
        else:
            print(f"Train behavior policy")
            behavior_policy_trainer.train() # save checkpoint periodically

    def get_rollout_trajs(logger: Logger, threshold = 0.9) -> Tuple[Dict[str, np.ndarray], float]:
        '''
        Rollout trajectories or load existing trajectories.
        If rollout, call `get_rollout_policy()` and `get_dynamics()` first to get rollout policy and dynamics

        Return:
            rollout trajectories
        '''
        '''
        diffusion behavior policy rollout

        - threshold: only keep trajs with ret > [threshold] (valid). Usually the max return in dataset
        - args.num_need_traj: number of valid trajectories needed. End rollout when get enough trajs
        - args.rollout_epoch: maximum rollout epoch. Should be large
        '''
        device = args.device
        num_need_traj = args.num_need_traj

        rollout_data_all = None # Initialize rollout_dataset as nothing
        num_traj_all = 0 # Initialize total number of rollout trajs
        start_epoch = 0 # Default starting epoch
        returns_all = []
        if args.rollout_ckpt_path is not None:
            print(f"Will save rollout trajectories to dir {args.rollout_ckpt_path}")
            os.makedirs(args.rollout_ckpt_path, exist_ok=True)
            data_path = os.path.join(args.rollout_ckpt_path, "rollout.dat")
            if os.path.exists(data_path): # Load ckpt_data
                ckpt_dict = pickle.load(open(data_path,"rb")) # checkpoint in dict type
                rollout_data_all = ckpt_dict['data'] # should be dict
                num_traj_all = ckpt_dict['num_traj']
                returns_all = ckpt_dict['return']
                start_epoch = ckpt_dict['epoch'] + 1
                # trajs = ckpt_dict
                print(f"Loaded checkpoint. Already have {num_traj_all} valid trajectories, start from epoch {start_epoch}.")

                if num_traj_all >= num_need_traj:
                    print(f"Checkpoint trajectories are enough. Skip rollout procedure.")
                    return rollout_data_all, max(returns_all)
        # Still need training, get dynamics and rollout policy
        get_dynamics()
        get_rollout_policy()

        state_mean, state_std = behavior_dataset.get_normalize_coef()

        with torch.no_grad():
            for epoch in range(start_epoch, args.rollout_epoch):
                batch_indexs = np.random.randint(0, init_obss_dataset.shape[0], size=args.rollout_batch)
                init_obss = init_obss_dataset[batch_indexs]
                rollout_data, rollout_info = rollout(init_obss, dynamics, behavior_policy, args.horizon,
                                                     state_mean, state_std, device, action_dim, args.ctx)
                    # print(pred_state)

                # Only keep trajs with returns > threshold
                returns = rollout_info['returns']
                rewards_full = rollout_info['rewards_full']
                min_last_rewards = np.min(rewards_full[:, -3:], axis = -1) # (B,), final steps must be large
                max_last_rewards = np.max(rewards_full[:, -3:], axis = -1)
                max_cond = np.logical_and(max_last_rewards > 0.9, max_last_rewards < 2)
                min_cond = min_last_rewards > 0.7
                valid_cond = np.logical_and(max_cond, min_cond)
                valid_trajs = np.arange(args.rollout_batch)[valid_cond] # np.array, indexs of all valid trajs

                valid_data_idxs = [rollout_data['traj_idxs'][i] in valid_trajs for i in range(rollout_data['traj_idxs'].shape[0])]
                for k in rollout_data:
                    rollout_data[k] = rollout_data[k][valid_data_idxs]

                # Add rollout_data to rollout_data_all
                if rollout_data_all is None: # No trajs collected
                    rollout_data_all = deepcopy(rollout_data)
                else:
                    for k in rollout_data:
                        rollout_data_all[k] = np.concatenate([rollout_data_all[k], rollout_data[k]], axis=0)
                
                num_traj_all += len(valid_trajs)
                returns_all += list(returns[valid_trajs])

                print(f"-----------\nEpoch {epoch}, get {len(valid_trajs)} new trajs")
                logger.logkv("Epoch", epoch)
                logger.logkv("num_new_trajs", len(valid_trajs))
                logger.logkv("num_total_trajs", num_traj_all)
                logger.dumpkvs()

                save_path = os.path.join(logger.checkpoint_dir, "rollout.dat")
                pickle.dump({'epoch': epoch, 
                                'data': rollout_data_all,
                                'num_traj': num_traj_all,
                                'return': returns_all}, open(save_path, "wb"))            

                if num_traj_all >= num_need_traj: # Get enough trajs, quit rollout
                    print(f"End rollout. Total epochs used: {epoch+1}")
                    break
            
        return rollout_data_all, max(returns_all)

    rollout_save_dir = make_log_dirs(args.task, args.algo_name, exp_name, vars(args), part="rollout")
    print(f"Logging diffusion rollout to {rollout_save_dir}")
    rollout_logger = Logger(rollout_save_dir, {"consoleout_backup": "stdout"})
    rollout_logger.log_hyperparameters(vars(args))
    rollout_dataset, max_offline_return = get_rollout_trajs(rollout_logger)

    # train
    rcsl_policy = AutoregressivePolicy(
        obs_dim=obs_dim,
        act_dim = action_dim,
        hidden_dims=args.rcsl_hidden_dims,
        lr = args.rcsl_lr,
        device = args.device
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_policy.rcsl_optim, args.rcsl_epoch)
    
    task_name = args.task
    rcsl_log_dirs = make_log_dirs(task_name, args.algo_name, exp_name, vars(args), part='rcsl')
    # key: output file name, value: output handler type
    print(f"Logging autoregressive gaussian rcsl to {rcsl_log_dirs}")
    rcsl_output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    rcsl_logger = Logger(rcsl_log_dirs, rcsl_output_config)
    rcsl_logger.log_hyperparameters(vars(args))

    policy_trainer = RcslPolicyTrainer(
        policy = rcsl_policy,
        eval_env = env,
        offline_dataset = rollout_dataset,
        rollout_dataset = None,
        goal = max_offline_return, # AutoregressivePolicy is not return-conditioned
        logger = rcsl_logger,
        seed = args.seed,
        epoch = args.rcsl_epoch,
        batch_size = args.rcsl_batch,
        offline_ratio = 1.,
        lr_scheduler = lr_scheduler,
        horizon = args.horizon,
        num_workers = args.num_workers,
        eval_episodes = args.eval_episodes
    )

    policy_trainer.train(holdout_ratio=args.holdout_ratio, last_eval=args.last_eval)


if __name__ == "__main__":
    train()