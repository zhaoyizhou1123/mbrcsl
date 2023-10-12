import argparse
import os
import sys
import random

import gym
import d4rl

import numpy as np
import torch

import pickle
from copy import deepcopy
from typing import Dict, Tuple
import roboverse


# import __init__
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel, RcslModule
from offlinerlkit.dynamics import BaseDynamics, EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset, traj_rtg_datasets
from offlinerlkit.utils.config import Config
from offlinerlkit.utils.dataset import ObsActDataset
from offlinerlkit.utils.pickplace_utils import SimpleObsWrapper, get_pickplace_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.utils.diffusion_logger import setup_logger
from offlinerlkit.policy_trainer import RcslPolicyTrainer
from offlinerlkit.utils.trajectory import Trajectory
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.policy import DiffusionBC, RcslPolicy, SimpleDiffusionPolicy
from offlinerlkit.env.linearq import Linearq

# from rvs.policies import RvS


from pointmaze.utils.trajectory import Trajs2Dict


"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, cql-weight=0.5
hopper-medium-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-v2: rollout-length=1, cql-weight=5.0
halfcheetah-medium-replay-v2: rollout-length=5, cql-weight=0.5
hopper-medium-replay-v2: rollout-length=5, cql-weight=0.5
walker2d-medium-replay-v2: rollout-length=1, cql-weight=0.5
halfcheetah-medium-expert-v2: rollout-length=5, cql-weight=5.0
hopper-medium-expert-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-expert-v2: rollout-length=1, cql-weight=5.0
"""


def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--algo-name", type=str, default="rcsl_diffusion")
    parser.add_argument("--task", type=str, default="pickplace", help="maze") # Self-constructed environment
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers, align with cpu number")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # env config (pickplace)
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--horizon', type=int, default=40, help="max path length for pickplace")

    # env config (maze)
    # parser.add_argument('--maze_config_file', type=str, default='./pointmaze/config/maze2_simple_moredata.json')
    # parser.add_argument('--data_file', type=str, default='./pointmaze/dataset/maze2_smds_acc.dat')
    # parser.add_argument('--render', action='store_true')
    # parser.add_argument('--log_to_wandb',action='store_true', help='Set up wandb')
    # parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/, Folder to tensorboard logs" )
    # parser.add_argument('--env_type', type=str, default='pointmaze', help='pointmaze or ?')

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    # # parser.add_argument("--rollout-freq", type=int, default=1000)
    # parser.add_argument("--rollout-batch-size", type=int, default=50000)
    # parser.add_argument("--rollout-length", type=int, default=5)
    # parser.add_argument("--model-retain-epochs", type=int, default=5)
    # parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=none_or_str, default=None)

    # Behavior policy (diffusion)
    parser.add_argument("--behavior_epoch", type=int, default=1000)
    parser.add_argument("--num_diffusion_iters", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument('--behavior_batch', type=int, default=256)
    parser.add_argument('--load_diffusion_path', type=none_or_str, default=None, help = "path to .pth file")
    parser.add_argument('--diffusion_seed', type=str, default='0', help="Distinguish runs for diffusion policy, not random seed")
    parser.add_argument('--task_weight', type=float, default=1.)

    # Rollout 
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable, file path, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epochs', type=int, default=1000, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=100, help="Needed valid trajs in rollout")
    parser.add_argument("--rollout-batch", type=int, default=256, help="Number of trajs to be sampled at one time")

    # RCSL policy (mlp)
    parser.add_argument("--rcsl-hidden-dims", type=int, nargs='*', default=[20])
    parser.add_argument("--rcsl-lr", type=float, default=1e-3)
    parser.add_argument("--rcsl-batch", type=int, default=256)
    parser.add_argument("--rcsl-epoch", type=int, default=30)
    parser.add_argument("--rcsl-step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--fix_eval_seed", action="store_true", help="True to fix the seed for every eval")

    parser.add_argument("--batch-size", type=int, default=256)

    return parser.parse_args()

def train(args=get_args()):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # log
    # log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # # key: output file name, value: output handler type
    # output_config = {
    #     "consoleout_backup": "stdout",
    #     "policy_training_progress": "csv",
    #     "dynamics_training_progress": "csv",
    #     "tb": "tensorboard"
    # }
    # logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args))

    # create env and dataset
    if args.task == 'pickplace':
        # render_mode = 'human' if args.render else None
        # env = Linearq(size_param=args.env_param)
        # env2 = Linearq(size_param=args.env_param)
        # # dataset = qlearning_dataset(env, get_rtg=True)
        # dataset, init_obss_dataset, max_offline_return = traj_rtg_datasets(env)
        # obs_space = env.observation_space
        # args.obs_shape = (1,)
        # print(args.obs_shape)
        env = roboverse.make('Widow250PickTray-v0')
        env = SimpleObsWrapper(env)
        # v_env = gym.vector.SyncVectorEnv([lambda: SimpleObsWrapper(roboverse.make('Widow250PickTray-v0')) for t in range(args.eval_episodes)])
        # env2 = roboverse.make('Widow250PickTray-v0')
        # env2 = SimpleObsWrapper(env2)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)

        dataset, init_obss = get_pickplace_dataset(args.data_dir, task_weight=args.task_weight)
    else:
        raise NotImplementedError

    
    # elif args.task == 'maze': # self-constructed
        
    #     from pointmaze.envs.create_maze_dataset import create_env_dataset
    #     point_maze = create_env_dataset(args)
    #     env = point_maze.env_cls()
    #     trajs = point_maze.dataset[0] # first object is trajs
    #     dataset = Trajs2Dict(trajs)

    #     # Add a get_true_observation method for Env
    #     def get_true_observation(obs):
    #         '''
    #         obs, obs received from pointmaze Env. Dict.
    #         '''
    #         return obs['observation']
    
    #     setattr(env, 'get_true_observation', get_true_observation)

    #     obs_space = env.observation_space['observation']
    #     args.obs_shape = env.observation_space['observation'].shape
    # else:
    #     render_mode = 'human' if args.render else None
    #     env = gym.make(args.task, render_mode = render_mode)
    #     env2 = gym.make(args.task, render_mode = render_mode)
    #     # dataset = qlearning_dataset(env, get_rtg=True)
    #     dataset, init_obss_dataset, max_offline_return = traj_rtg_datasets(env)
    #     obs_space = env.observation_space
    #     args.obs_shape = env.observation_space.shape
    # obs_dim = np.prod(args.obs_shape)
    # args.action_dim = np.prod(env.action_space.shape)
    # obs_dim = int(1)
    # args.action_dim = int(1)
    # print(obs_dim, args.action_dim)
    # args.max_action = env.action_space.high[0]

    # seed
    env.reset(seed = args.seed)
    # env2.reset(seed = args.seed)

    # rcsl_backbone = MLP(input_dim=obs_dim+1, hidden_dims=args.rcsl_hidden_dims, output_dim=args.action_dim)

    # rcsl_module = RcslModule(rcsl_backbone, args.device)
    # rcsl_optim = torch.optim.Adam(rcsl_module.parameters(), lr=args.rcsl_lr)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_optim, args.rcsl_epoch)

    # rcsl_policy = RcslPolicy(
    #     dynamics = None,
    #     rollout_policy = None,
    #     rcsl = rcsl_module,
    #     rcsl_optim = rcsl_optim,
    #     device = args.device
    # )

    diffusion_policy = SimpleDiffusionPolicy(
        obs_shape = args.obs_shape,
        act_shape= args.action_shape,
        feature_dim = 1,
        num_training_steps = args.behavior_epoch,
        num_diffusion_steps = args.num_diffusion_iters,
        device = args.device
    )

    if args.load_diffusion_path is not None:
        with open(args.load_diffusion_path, 'rb') as f:
            state_dict = torch.load(f, map_location= args.device)
        diffusion_policy.load_state_dict(state_dict)


    diff_lr_scheduler = diffusion_policy.get_lr_scheduler()

    # create buffer
    # offline_buffer = ReplayBuffer(
    #     buffer_size=len(dataset["observations"]),
    #     obs_shape=args.obs_shape,
    #     obs_dtype=np.float32,
    #     action_dim=args.action_dim,
    #     action_dtype=np.float32,
    #     device=args.device
    # )
    # offline_buffer.load_dataset(dataset)

    # train

    # Creat policy trainer
    rcsl_log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    rcsl_output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    rcsl_logger = Logger(rcsl_log_dirs, rcsl_output_config)
    rcsl_logger.log_hyperparameters(vars(args))

    policy_trainer = RcslPolicyTrainer(
        policy = diffusion_policy,
        eval_env = env,
        # eval_env2 = v_env,
        offline_dataset = dataset,
        rollout_dataset = None,
        goal = 0,
        logger = rcsl_logger,
        seed = args.seed,
        epoch = args.rcsl_epoch,
        step_per_epoch = args.rcsl_step_per_epoch,
        batch_size = args.rcsl_batch,
        offline_ratio = 1.,
        lr_scheduler = diff_lr_scheduler,
        horizon = args.horizon,
        num_workers = args.num_workers,
        has_terminal = False,
        eval_episodes = args.eval_episodes
        # device = args.device
    )
    
    # print(f"Start evaluate")
    policy_trainer.train(last_eval=True)
    # result = policy_trainer._evaluate()
    # print(result)


if __name__ == "__main__":
    train()