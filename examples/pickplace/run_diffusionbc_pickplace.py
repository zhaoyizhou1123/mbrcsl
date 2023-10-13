import argparse
import random
import datetime
from copy import deepcopy
from typing import Dict, Tuple
import roboverse

import numpy as np
import torch

from offlinerlkit.utils.pickplace_utils import SimpleObsWrapper, get_pickplace_dataset
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import RcslPolicyTrainer
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.policy import SimpleDiffusionPolicy

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--algo-name", type=str, default="diffusionbc")
    parser.add_argument("--task", type=str, default="pickplace", help="maze") # Self-constructed environment
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader workers, align with cpu number")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # env config (pickplace)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=40, help="max path length for pickplace")

    # Behavior Cloning Policy (diffusion)
    parser.add_argument('--load_diffusion_path', type=none_or_str, default=None, help = "path to .pth file")
    parser.add_argument("--num_diffusion_iters", type=int, default=5, help="Number of diffusion steps")
    parser.add_argument('--task_weight', type=float, default=1.)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=100)

    return parser.parse_args()

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
        env = SimpleObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        args.action_shape = env.action_space.shape

        dataset, init_obss = get_pickplace_dataset(args.data_dir, task_weight=args.task_weight)
    else:
        raise NotImplementedError
    diffusion_policy = SimpleDiffusionPolicy(
        obs_shape = args.obs_shape,
        act_shape= args.action_shape,
        feature_dim = 1,
        num_training_steps = args.epoch,
        num_diffusion_steps = args.num_diffusion_iters,
        device = args.device
    )

    if args.load_diffusion_path is not None:
        with open(args.load_diffusion_path, 'rb') as f:
            state_dict = torch.load(f, map_location= args.device)
        diffusion_policy.load_state_dict(state_dict)


    diff_lr_scheduler = diffusion_policy.get_lr_scheduler()

    # Creat policy trainer
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    exp_name = f"timestamp_{timestamp}&{args.seed}"
    rcsl_log_dirs = make_log_dirs(args.task, args.algo_name, exp_name, vars(args))
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
        epoch = args.epoch,
        batch_size = args.batch,
        offline_ratio = 1.,
        lr_scheduler = diff_lr_scheduler,
        horizon = args.horizon,
        num_workers = args.num_workers,
        has_terminal = False,
        eval_episodes = args.eval_episodes
    )
    policy_trainer.train(last_eval=True)


if __name__ == "__main__":
    train()