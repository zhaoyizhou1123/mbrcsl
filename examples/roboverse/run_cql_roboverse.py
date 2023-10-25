import argparse
import os
import sys
import random
import datetime
import roboverse

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import CQLPolicy
from offlinerlkit.utils.roboverse_utils import PickPlaceObsWrapper, DoubleDrawerObsWrapper, get_pickplace_dataset, get_doubledrawer_dataset

'''
Recommended hyperparameters:
pickplace, horizon=40, cql_weight=1.0
doubledraweropen, horizon=50, cql_weight=5.0
doubledrawercloseopen, horizon=80, behavior_epoch=2.0
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="cql")
    parser.add_argument("--task", type=str, default="pickplace")
    parser.add_argument("--last_eval", action="store_false")
    
    # env config (pickplace)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=40, help="max path length for pickplace")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

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
        env = PickPlaceObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        args.obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        args.action_dim = np.prod(args.action_shape)

        prior_data_path = os.path.join(args.data_dir, "pickplace_prior.npy")
        task_data_path = os.path.join(args.data_dir, "pickplace_task.npy")
        dataset, init_obss_dataset = get_pickplace_dataset(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
    elif args.task == 'doubledraweropen':
        env = roboverse.make('Widow250DoubleDrawerOpenGraspNeutral-v0')
        env = DoubleDrawerObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        args.obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        args.action_dim = np.prod(args.action_shape)

        prior_data_path = os.path.join(args.data_dir, "closed_drawer_prior.npy")
        task_data_path = os.path.join(args.data_dir, "drawer_task.npy")

        dataset, init_obss_dataset = get_doubledrawer_dataset(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
    elif args.task == 'doubledrawercloseopen':
        env = roboverse.make('Widow250DoubleDrawerCloseOpenGraspNeutral-v0')
        env = DoubleDrawerObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        args.obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        args.action_dim = np.prod(args.action_shape)

        prior_data_path = os.path.join(args.data_dir, "blocked_drawer_1_prior.npy")
        task_data_path = os.path.join(args.data_dir, "drawer_task.npy")

        dataset, init_obss_dataset = get_doubledrawer_dataset(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
    elif args.task == 'doubledrawerpickplaceopen':
        env = roboverse.make('Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0')
        env = DoubleDrawerObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        args.obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        args.action_dim = np.prod(args.action_shape)

        prior_data_path = os.path.join(args.data_dir, "blocked_drawer_2_prior.npy")
        task_data_path = os.path.join(args.data_dir, "drawer_task.npy")

        dataset, init_obss_dataset = get_doubledrawer_dataset(
            prior_data_path=prior_data_path,
            task_data_path=task_data_path)
    else:
        raise NotImplementedError
    env.reset(seed=args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create policy
    policy = CQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # log
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    exp_name = f"timestamp_{timestamp}&{args.seed}"
    log_dirs = make_log_dirs(args.task, args.algo_name, exp_name, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        horizon = args.horizon,
        has_terminal=False,
        binary_ret = True,
        last_eval_episodes=100
    )

    # train
    policy_trainer.train(last_eval=args.last_eval)


if __name__ == "__main__":
    train()