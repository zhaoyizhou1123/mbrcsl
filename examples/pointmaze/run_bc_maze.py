import numpy as np
import torch
import random
import datetime
import argparse

from envs.pointmaze.utils.evaluate_episodes import evaluate_episode
from offlinerlkit.policy import MLPBCModel
from offlinerlkit.policy_trainer import ActTrainer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.utils.set_up_seed import set_up_seed
from envs.pointmaze.create_maze_dataset import create_env_dataset
from envs.pointmaze.utils.maze_utils import PointMazeObsWrapper

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--algo_name", type=str, default="bc")
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

    # mlp bc policy
    parser.add_argument('--ctx', type=int, default=20)
    parser.add_argument('--data_ratio', type = float, default=0.1, help="%BC")
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)

    # training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate of Trainer" )
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument("--step_per_epoch", type=int, default=1000)
    
    args = parser.parse_args()
    return args

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def train(args = get_args()):
    variant = vars(args)
    device = variant.get('device', 'cuda')

    env_name = variant['task']
    model_type = variant['algo_name']

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
        scale = 1

        # re-format trajs
        trajs = [traj._asdict() for traj in trajs]
        for traj in trajs:
            for k in traj:
                traj[k] = np.asarray(traj[k])
            traj['dones'] = traj['terminated']
    else:
        raise NotImplementedError

    env.reset(seed = args.seed)

    if model_type == 'bc':
        # env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations
        env_targets = [0]
    else:
        raise NotImplementedError

    state_dim = obs_dim
    act_dim = action_dim

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajs:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        # returns.append(path['rewards'].sum())
        returns.append(sum(path['rewards']))
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['ctx']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['eval_episodes']
    pct_traj = variant.get('data_ratio')

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajs) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajs[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= args.horizon] = args.horizon-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'bc':
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=args.horizon,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        raise NotImplementedError
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['lr'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    # logger
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

    for iter in range(variant['epoch']):
        outputs = trainer.train_iteration(num_steps=variant['step_per_epoch'], iter_num=iter+1, print_logs=True)
        for k,v in outputs.items():
            logger.logkv(k,v)
        logger.set_timestep(iter)
        logger.dumpkvs()


if __name__ == '__main__':
    train()
