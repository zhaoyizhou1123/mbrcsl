
#general

import argparse
import json
import subprocess
import numpy as np
from tqdm import tqdm
import os
import glob
import tarfile
from comet_ml import Experiment
import datetime

from morel.morel import Morel
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from envs.pointmaze.create_maze_dataset import create_env_dataset
from envs.pointmaze.utils.trajectory import get_pointmaze_dataset
from envs.pointmaze.utils.maze_utils import PointMazeObsWrapper
from offlinerlkit.utils.set_up_seed import set_up_seed
from offlinerlkit.utils.logger import Logger, make_log_dirs

class Maze2DDataset(Dataset):

    def __init__(self, dataset):
        # self.env = gym.make('maze2d-umaze-v1')
        # dataset = self.env.get_dataset()

        # Input data
        self.source_observation = dataset["observations"][:-1]
        self.source_action = dataset["actions"][:-1]


        # Output data
        self.target_delta = dataset["observations"][1:] - self.source_observation
        self.target_reward = dataset["rewards"][:-1]

        # Normalize data
        self.delta_mean = self.target_delta.mean(axis=0)
        self.delta_std = self.target_delta.std(axis=0)

        self.reward_mean = self.target_reward.mean(axis=0)
        self.reward_std = self.target_reward.std(axis=0)

        self.observation_mean = self.source_observation.mean(axis=0)
        self.observation_std = self.source_observation.std(axis=0)

        self.action_mean = self.source_action.mean(axis=0)
        self.action_std = self.source_action.std(axis=0)

        self.source_action = (self.source_action - self.action_mean)/self.action_std
        self.source_observation = (self.source_observation - self.observation_mean)/self.observation_std
        self.target_delta = (self.target_delta - self.delta_mean)/self.delta_std
        self.target_reward = (self.target_reward - self.reward_mean)/self.reward_std

        # Get indices of initial states
        self.done_indices = dataset["terminals"][:-1]
        self.initial_indices = np.roll(self.done_indices, 1)
        self.initial_indices[0] = True

        # Calculate distribution parameters for initial states
        self.initial_obs = self.source_observation[self.initial_indices]
        self.initial_obs_mean = self.initial_obs.mean(axis = 0)
        self.initial_obs_std = self.initial_obs.std(axis = 0)

        # Remove transitions from terminal to initial states
        self.source_action = np.delete(self.source_action, self.done_indices, axis = 0)
        self.source_observation = np.delete(self.source_observation, self.done_indices, axis = 0)
        self.target_delta = np.delete(self.target_delta, self.done_indices, axis = 0)
        self.target_reward = np.delete(self.target_reward, self.done_indices, axis = 0)



    def __getitem__(self, idx):
        feed = torch.FloatTensor(np.concatenate([self.source_observation[idx], self.source_action[idx]])).to("cuda:0")
        target = torch.FloatTensor(np.concatenate([self.target_delta[idx], self.target_reward[idx:idx+1]])).to("cuda:0")

        return feed, target

    def __len__(self):
        return len(self.source_observation)

def upload_assets(comet_experiment, log_dir):
    tar_path = log_dir + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(log_dir, arcname=os.path.basename(log_dir))

    comet_experiment.log_asset(tar_path)
    os.remove(tar_path)

def main(args):
    set_up_seed(args.seed)

    tensorboard_writer = None
    comet_experiment = None

    if(not args.no_log):
        # Create necessary directories
        if(not os.path.isdir(args.log_dir)):
            os.mkdirs(args.log_dir)

        # Create log_dir for run
        run_log_dir = os.path.join(args.log_dir,args.exp_name)
        if(os.path.isdir(run_log_dir)):
            cur_count = len(glob.glob(run_log_dir + "_*"))
            run_log_dir = run_log_dir + "_" + str(cur_count)
        os.mkdir(run_log_dir)

        # Create tensorboard writer if requested

        if(args.tensorboard):
            tensorboard_dir = os.path.join(run_log_dir, "tensorboard")
            writer = SummaryWriter(log_dir = tensorboard_dir)


    # Create comet experiment if requested
    if(args.comet_config is not None):
        with open(args.comet_config, 'r') as f:
            comet_dict = json.load(f)
            comet_experiment = Experiment(
                api_key = comet_dict["api_key"],
                project_name = comet_dict["project_name"],
                workspace = comet_dict["workspace"],
            )
            comet_experiment.set_name(args.exp_name)

            # Get hash for latest git commit for logging
            last_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").rstrip()
            comet_experiment.log_parameter("git_commit_id", last_commit_hash)

    # Instantiate dataset

    # create env and dataset
    if args.task == 'pointmaze':
        env, trajs = create_env_dataset(args)
        env = PointMazeObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)
        dataset, _, _ = get_pointmaze_dataset(trajs)
    else:
        raise NotImplementedError

    dynamics_data = Maze2DDataset(dataset)

    dataloader = DataLoader(dynamics_data, batch_size=128, shuffle = True)

    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    exp_name = f"timestamp_{timestamp}&{args.seed}"
    log_dirs = make_log_dirs(args.task, args.algo_name, exp_name, vars(args))
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

    agent = Morel(obs_dim, action_dim, args.horizon, tensorboard_writer = tensorboard_writer, comet_experiment = comet_experiment, logger = logger)

    agent.train(dataloader, dynamics_data)

    if(not args.no_log):
        agent.save(os.path.join(run_log_dir, "models"))
        if comet_experiment is not None:
            upload_assets(comet_experiment, run_log_dir)

    agent.eval(env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')

    parser.add_argument('--log_dir', type=str, default='logs/pointmaze/morel')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--comet_config', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='exp_test')
    parser.add_argument('--no_log', action='store_false')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="pointmaze", help="task name")
    parser.add_argument("--algo_name", type=str, default="morel")

    # env config (general)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=200, help="max path length for pickplace")

    # env config (pointmaze)
    parser.add_argument('--maze_config_file', type=str, default='envs/pointmaze/config/maze_default.json')
    parser.add_argument('--data_file', type=str, default='pointmaze.dat')


    args = parser.parse_args()
    main(args)