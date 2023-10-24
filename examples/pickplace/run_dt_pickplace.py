import numpy as np
import torch
import os
import argparse
import roboverse
import datetime

from offlinerlkit.policy import DecisionTransformer
from offlinerlkit.policy_trainer import SequenceTrainer, TrainerConfig
from offlinerlkit.utils.set_up_seed import set_up_seed
from offlinerlkit.utils.roboverse_utils import PickPlaceObsWrapper, DoubleDrawerObsWrapper, get_pickplace_dataset_dt, get_doubledrawer_dataset_dt
from offlinerlkit.utils.none_or_str import none_or_str
from offlinerlkit.utils.dataset import TrajCtxFloatLengthDataset
from offlinerlkit.utils.logger import Logger, make_log_dirs

'''
Recommended hyperparameters:
pickplace, horizon=40
doubledraweropen, horizon=50
doubledrawercloseopen, horizon=80
doubledrawerpickplaceopen, horizon=80
'''

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--algo-name", type=str, default="dt")
    parser.add_argument("--task", type=str, default="pickplace", help="task name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader workers, align with cpu number")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--last_eval", action="store_false")

    # env config
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=40, help="max path length for pickplace")

    # transformer mode
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=32)
    parser.add_argument('--ctx', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=128, help="dt token embedding dimension")

    # Train
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--lr', type=float, default=6e-3, help="learning rate of Trainer" )
    parser.add_argument('--goal_mul', type=float, default=1., help="goal = max_dataset_return * goal_mul")
    parser.add_argument('--sample', action='store_false', help="Sample action by probs, or choose the largest prob")

    return parser.parse_args()

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def train(args = get_args()):
    set_up_seed(args.seed)

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

        trajs = get_pickplace_dataset_dt(
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
    else:
        raise NotImplementedError

    env.reset(seed = args.seed)

    offline_train_dataset = TrajCtxFloatLengthDataset(trajs, ctx = args.ctx, single_timestep = False, with_mask=True)
    goal = offline_train_dataset.get_max_return() * args.goal_mul
    model = DecisionTransformer(
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

    # logger
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    exp_name = f"timestamp_{timestamp}&{args.seed}"
    rcsl_log_dirs = make_log_dirs(args.task, args.algo_name, exp_name, vars(args))
    # key: output file name, value: output handler type
    rcsl_output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    rcsl_logger = Logger(rcsl_log_dirs, rcsl_output_config)
    rcsl_logger.log_hyperparameters(vars(args))

    tconf = TrainerConfig(
                max_epochs=args.epoch, batch_size=args.batch_size, lr=args.lr,
                lr_decay=True, num_workers=1, horizon=args.horizon, 
                grad_norm_clip = 1.0, eval_repeat = args.eval_episodes, desired_rtg=goal, 
                env = env, ctx = args.ctx, device=args.device, 
                debug = False, logger = rcsl_logger, last_eval = args.last_eval)
    output_policy_trainer = SequenceTrainer(
        config=tconf,
        model=model,
        offline_dataset=offline_train_dataset,
        is_gym = True)
    
    output_policy_trainer.train()


if __name__ == '__main__':
    train()
