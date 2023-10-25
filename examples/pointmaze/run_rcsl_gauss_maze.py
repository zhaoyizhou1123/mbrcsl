import numpy as np
import torch

import argparse
import os
import random
import pickle
from copy import deepcopy
from typing import Dict, Tuple
import datetime

from offlinerlkit.modules import RcslGaussianModule, DiagGaussian
from offlinerlkit.nets import MLP
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import RcslPolicyTrainer
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.utils.set_up_seed import set_up_seed
from offlinerlkit.policy import RcslGaussianPolicy
from envs.pointmaze.create_maze_dataset import create_env
from envs.pointmaze.utils.maze_utils import PointMazeObsWrapper

'''
task:
pointmaze
'''

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--algo_name", type=str, default="mbrcsl_gauss")
    parser.add_argument("--task", type=str, default="pointmaze", help="task name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader workers, align with cpu number")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--last_eval", action="store_true")

    # env config
    parser.add_argument('--horizon', type=int, default=200, help="max path length for pickplace")
    parser.add_argument('--rollout_ckpt_path', type=str, required=True, help="dir path, used to load mbrcsl rollout trajectories")
    parser.add_argument('--maze_config_file', type=str, default='envs/pointmaze/config/maze_default.json')

    # RCSL policy (mlp)
    parser.add_argument("--rcsl_hidden_dims", type=int, nargs='*', default=[1024,1024])
    parser.add_argument("--rcsl_lr", type=float, default=5e-5)
    parser.add_argument("--rcsl_batch", type=int, default=256)
    parser.add_argument("--rcsl_epoch", type=int, default=100)
    parser.add_argument("--rcsl_weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--holdout_ratio", type=float, default=0.1)

    return parser.parse_args()

def train(args=get_args()):
    set_up_seed(args.seed)

    # create env and dataset
    if args.task == 'pointmaze':
        env = create_env(args)
        env = PointMazeObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)
    else:
        raise NotImplementedError

    env.reset(seed=args.seed)

    # Get rollout dataset  
    data_path = os.path.join(args.rollout_ckpt_path, "rollout.dat")
    ckpt_dict = pickle.load(open(data_path,"rb")) # checkpoint in dict type
    
    rollout_dataset = ckpt_dict['data'] # should be dict
    num_traj_all = ckpt_dict['num_traj']
    print(f"Loaded {num_traj_all} rollout trajectories")

    returns_all = ckpt_dict['return']
    max_rollout_return = max(returns_all)

    # train
    set_up_seed(args.seed)
    rcsl_backbone = MLP(
        input_dim = obs_dim + 1,
        hidden_dims = args.rcsl_hidden_dims,
        output_dim = action_dim,
        init_last = True
    )

    dist = DiagGaussian(
        latent_dim=getattr(rcsl_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    rcsl_module = RcslGaussianModule(
        rcsl_backbone,
        dist,
        device = args.device
    )
    rcsl_optim = torch.optim.Adam(rcsl_module.parameters(), lr=args.rcsl_lr, weight_decay=args.rcsl_weight_decay)

    rcsl_policy = RcslGaussianPolicy(
        rcsl_module,
        rcsl_optim,
        device = args.device
    )
    lr_scheduler=None
    
    task_name = args.task
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    exp_name = f"timestamp_{timestamp}&{args.seed}"
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

    policy_trainer.train(holdout_ratio=args.holdout_ratio, last_eval=args.last_eval)


if __name__ == "__main__":
    train()