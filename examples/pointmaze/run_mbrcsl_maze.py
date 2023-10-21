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

from offlinerlkit.modules import EnsembleDynamicsModel, RcslModule
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.nets import MLP
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import RcslPolicyTrainer, DiffusionPolicyTrainer
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.utils.set_up_seed import set_up_seed
from offlinerlkit.policy import SimpleDiffusionPolicy, RcslPolicy
from envs.pointmaze.create_maze_dataset import create_env_dataset
from envs.pointmaze.utils.trajectory import get_pointmaze_dataset
from envs.pointmaze.utils.maze_utils import PointMazeObsWrapper

'''
task:
pointmaze
'''

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--algo-name", type=str, default="mbrcsl")
    parser.add_argument("--task", type=str, default="pointmaze", help="task name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader workers, align with cpu number")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--last_eval", action="store_true")

    # env config (general)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=200, help="max path length for pickplace")

    # env config (pointmaze)
    parser.add_argument('--maze_config_file', type=str, default='envs/pointmaze/config/maze_default.json')
    parser.add_argument('--data_file', type=str, default='pointmaze.dat')

    # ensemble dynamics
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--dynamics_hidden_dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics_weight_decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--dynamics_lr", type=float, default=1e-3)
    parser.add_argument("--load_dynamics_path", type=none_or_str, default=None)

    # Behavior policy (diffusion)
    parser.add_argument("--behavior_epoch", type=int, default=50)
    parser.add_argument("--num_diffusion_iters", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument('--behavior_batch', type=int, default=256)
    parser.add_argument('--load_diffusion_path', type=none_or_str, default=None)
    parser.add_argument('--sample_ratio', type=float, default=1., help="Use (sample_ratio * num_total_data) data to train diffusion policy")
    
    # Rollout 
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="file dir, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epoch', type=int, default=1000, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=100, help="Needed valid trajs in rollout")
    parser.add_argument("--rollout-batch", type=int, default=200, help="Number of trajs to be sampled at one time")

    # RCSL policy (mlp)
    parser.add_argument("--rcsl_hidden_dims", type=int, nargs='*', default=[1024,1024])
    parser.add_argument("--rcsl_lr", type=float, default=5e-5)
    parser.add_argument("--rcsl_batch", type=int, default=256)
    parser.add_argument("--rcsl_epoch", type=int, default=100)
    parser.add_argument("--rcsl_weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--holdout_ratio", type=float, default=0.1)
    parser.add_argument("--find_best_start", type=int, default=50)

    return parser.parse_args()

def rollout_simple(
    init_obss: np.ndarray,
    dynamics: EnsembleDynamicsModel,
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

def train(args=get_args()):
    set_up_seed(args.seed)

    # create env and dataset
    if args.task == 'pointmaze':
        env, trajs = create_env_dataset(args)
        env = PointMazeObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)
        diff_dataset, _, _ = get_pointmaze_dataset(
            trajs,
            sample_ratio =args.sample_ratio)
        dyn_dataset, init_obss_dataset, max_offline_return = get_pointmaze_dataset(trajs)
    else:
        raise NotImplementedError

    env.reset(seed=args.seed)

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

    dynamics_model = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )

    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )

    # create rollout policy
    diffusion_policy = SimpleDiffusionPolicy(
        obs_shape = args.obs_shape,
        act_shape= args.action_shape,
        feature_dim = 1,
        num_training_steps = args.behavior_epoch,
        num_diffusion_steps = args.num_diffusion_iters,
        device = args.device
    )

    diff_lr_scheduler = diffusion_policy.get_lr_scheduler()

    diff_log_dirs = make_log_dirs(args.task, args.algo_name, exp_name, vars(args), part="diffusion")
    print(f"Logging diffusion to {diff_log_dirs}")
    # key: output file name, value: output handler type
    diff_output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    diff_logger = Logger(diff_log_dirs, diff_output_config)
    diff_logger.log_hyperparameters(vars(args))

    diff_policy_trainer = DiffusionPolicyTrainer(
        policy = diffusion_policy,
        offline_dataset = diff_dataset,
        logger = diff_logger,
        seed = args.seed,
        epoch = args.behavior_epoch,
        batch_size = args.behavior_batch,
        lr_scheduler = diff_lr_scheduler,
        horizon = args.horizon,
        num_workers = args.num_workers,
        has_terminal = False,
    )

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
            dynamics.train(dyn_dataset, logger)
        
    def get_rollout_policy():
        '''
        Load or train rollout policy

        Return:
            rollout policy
        '''
        if args.load_diffusion_path is not None:
            print(f"Load behavior policy from {args.load_diffusion_path}")
            with open(args.load_diffusion_path, 'rb') as f:
                state_dict = torch.load(f, map_location= args.device)
            diffusion_policy.load_state_dict(state_dict)
        else:
            print(f"Train diffusion behavior policy")
            diff_policy_trainer.train() # save checkpoint periodically

    def get_rollout_trajs(logger: Logger, threshold) -> Tuple[Dict[str, np.ndarray], float]:
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

        with torch.no_grad():
            for epoch in range(start_epoch, args.rollout_epoch):
                batch_indexs = np.random.randint(0, init_obss_dataset.shape[0], size=args.rollout_batch)
                init_obss = init_obss_dataset[batch_indexs]
                rollout_data, rollout_info = rollout_simple(init_obss, dynamics, diffusion_policy, args.horizon)
                    # print(pred_state)

                # Only keep trajs with returns > threshold
                returns = rollout_info['returns']
                valid_cond = returns > threshold
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
    rollout_dataset, max_rollout_return = get_rollout_trajs(rollout_logger, threshold = max_offline_return)

    # train
    set_up_seed(args.seed)
    rcsl_backbone = MLP(
        input_dim = obs_dim + 1,
        hidden_dims = args.rcsl_hidden_dims,
        output_dim = action_dim,
        init_last = True
    )

    rcsl_module = RcslModule(
        rcsl_backbone,
        device = args.device
    )
    rcsl_optim = torch.optim.Adam(rcsl_module.parameters(), lr=args.rcsl_lr, weight_decay=args.rcsl_weight_decay)

    rcsl_policy = RcslPolicy(
        rcsl_module,
        rcsl_optim,
        device = args.device
    )
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_optim, args.rcsl_epoch)
    lr_scheduler=None
    
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
        goal = max_rollout_return, # AutoregressivePolicy is not return-conditioned
        logger = rcsl_logger,
        seed = args.seed,
        epoch = args.rcsl_epoch,
        batch_size = args.rcsl_batch,
        offline_ratio = 1.,
        lr_scheduler = lr_scheduler,
        horizon = args.horizon,
        num_workers = args.num_workers,
        eval_episodes = args.eval_episodes,
        binary_return=False
    )

    rcsl_logger.log(f"Desired return: {max_rollout_return}")

    policy_trainer.train(holdout_ratio=args.holdout_ratio, last_eval=args.last_eval, find_best_start=args.find_best_start)


if __name__ == "__main__":
    train()