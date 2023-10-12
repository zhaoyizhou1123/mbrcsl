'''
Add trajectory dataset, for rcsl methods
'''

import numpy as np
import torch
import collections
import gym
import d4rl
import pickle
import os
from typing import Optional

from offlinerlkit.utils.cumsum import discount_cumsum


def qlearning_dataset(env, dataset=None, terminate_on_end=False, get_rtg = False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        get_rtg (bool): Include key 'rtgs' in the return dict if True. Default is False
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            rtgs: An N-dim float array of rtgs. (has this key if get_rtg=True)
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    # print(f"Successfully load dataset")
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    rtg_ = []

    acc_ret_traj_ = [] # Maintain accumulate return for one trajectory
    ret = 0 # Maintain acc ret in one trajectory

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        # print(f"Data point {i} / {N}")
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        if (not terminate_on_end) and final_timestep: # reach timeout, usually throw away timeout data
            # Skip this transition and don't apply terminals on the last step of an episode

            # Update rtg for this traj
            if get_rtg:
                rtg_traj_ = [ret - acc_ret for acc_ret in acc_ret_traj_]
                rtg_ += rtg_traj_

                rtg_traj_ = []
            ret = 0
            episode_step = 0
            continue  # skip this data

        # terminate_on_end (rare) or not timeout
        if done_bool or final_timestep: # Most cases: done_bool, i.e., reach terminal
            episode_step = 0
            if not has_next_obs: # if no next_obs, just throw away; else use this data
                if get_rtg:
                    rtg_traj_ = [ret - acc_ret for acc_ret in acc_ret_traj_]
                    rtg_ += rtg_traj_
                    
                    rtg_traj_ = []
                ret = 0
                continue # skip this data

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

        if get_rtg:
            acc_ret_traj_.append(ret)
        ret += reward
        episode_step += 1

        if done_bool or final_timestep: # for (done_bool or final_timestep) and has_next_obs
            assert has_next_obs, f"Should has_next_obs = True, is actually False!"
            # episode_step = 0
            if get_rtg:
                rtg_traj_ = [ret - acc_ret for acc_ret in acc_ret_traj_]
                rtg_ += rtg_traj_
                
                rtg_traj_ = []
            ret = 0

    if get_rtg:
        assert len(obs_) == len(rtg_), f"Obs {len(obs_)} and Rtg {len(rtg_)} should be same length!"
    if not get_rtg:
        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
        }
    else:
        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
            'rtgs': np.array(rtg_)
        }


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, max_ep_len=1000, device="cpu"):
        super().__init__()

        self.obs_dim = dataset["observations"].shape[-1]
        self.action_dim = dataset["actions"].shape[-1]
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.device = torch.device(device)
        self.input_mean = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).mean(0)
        self.input_std = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).std(0) + 1e-6

        data_ = collections.defaultdict(list)
        
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        self.trajs = []
        for i in range(dataset["rewards"].shape[0]):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000-1)
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                self.trajs.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1
        
        indices = []
        for traj_ind, traj in enumerate(self.trajs):
            end = len(traj["rewards"])
            for i in range(end):
                indices.append((traj_ind, i, i+self.max_len))

        self.indices = np.array(indices)
        

        returns = np.array([np.sum(t['rewards']) for t in self.trajs])
        num_samples = np.sum([t['rewards'].shape[0] for t in self.trajs])
        print(f'Number of samples collected: {num_samples}')
        print(f'Num trajectories: {len(self.trajs)}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_ind, start_ind, end_ind = self.indices[idx]
        traj = self.trajs[traj_ind].copy()
        obss = traj['observations'][start_ind:end_ind]
        actions = traj['actions'][start_ind:end_ind]
        next_obss = traj['next_observations'][start_ind:end_ind]
        rewards = traj['rewards'][start_ind:end_ind].reshape(-1, 1)
        delta_obss = next_obss - obss
    
        # padding
        tlen = obss.shape[0]
        inputs = np.concatenate([obss, actions], axis=1)
        inputs = (inputs - self.input_mean) / self.input_std
        inputs = np.concatenate([inputs, np.zeros((self.max_len - tlen, self.obs_dim+self.action_dim))], axis=0)
        targets = np.concatenate([delta_obss, rewards], axis=1)
        targets = np.concatenate([targets, np.zeros((self.max_len - tlen, self.obs_dim+1))], axis=0)
        masks = np.concatenate([np.ones(tlen), np.zeros(self.max_len - tlen)], axis=0)

        inputs = torch.from_numpy(inputs).to(dtype=torch.float32, device=self.device)
        targets = torch.from_numpy(targets).to(dtype=torch.float32, device=self.device)
        masks = torch.from_numpy(masks).to(dtype=torch.float32, device=self.device)

        return inputs, targets, masks
    

# From https://github.com/kzl/decision-transformer/blob/master/gym/data/download_d4rl_datasets.py 

def traj_rtg_datasets(env, input_path: Optional[str] =None, data_path: Optional[str] = None):
    '''
    Download all datasets needed for experiments, and re-combine them as trajectory datasets
    Throw away the last uncompleted trajectory

    Args:
        data_dir: path to store dataset file

    Return:
        dataset: Dict,
        initial_obss: np.ndarray
        max_return: float
    '''
    dataset = env.get_dataset(h5path=input_path)

    N = dataset['rewards'].shape[0] # number of data (s,a,r)
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    # obs_ = []
    # next_obs_ = []
    # action_ = []
    # reward_ = []
    # done_ = []
    # rtg_ = []

    for i in range(N): # Loop through data points

        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000-1)
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(dataset[k][i])
            
        # obs_.append(dataset['observations'][i].astype(np.float32))
        # next_obs_.append(dataset['next_observations'][i].astype(np.float32))
        # action_.append(dataset['actions'][i].astype(np.float32))
        # reward_.append(dataset['rewards'][i].astype(np.float32))
        # done_.append(bool(dataset['terminals'][i]))

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            # Update rtg
            rtg_traj = discount_cumsum(np.array(data_['rewards']))
            episode_data['rtgs'] = rtg_traj
            # rtg_ += rtg_traj

            paths.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    init_obss = np.array([p['observations'][0] for p in paths]).astype(np.float32)

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    if data_path is not None:
        with open(data_path, 'wb') as f:
            pickle.dump(paths, f)

    # print(f"N={N},len(obs_)={len(obs_)},len(reward_)={len(reward_)},len(rtg_)={len(rtg_)}!")
    # assert len(obs_) == len(rtg_), f"Got {len(obs_)} obss, but {len(rtg_)} rtgs!"

    # Concatenate paths into one dataset
    full_dataset = {}
    for k in ['observations', 'next_observations', 'actions', 'rewards', 'rtgs', 'terminals']:
        full_dataset[k] = np.concatenate([p[k] for p in paths], axis=0)

    return full_dataset, init_obss, np.max(returns)

# def traj_rtg_custom_datasets(env, input_path: str, data_path: Optional[str] = None):
#     '''
#     Download all datasets needed for experiments, and re-combine them as trajectory datasets
#     Throw away the last uncompleted trajectory

#     Args:
#         data_dir: path to store dataset file
#         input_path: path to input dataset file

#     Return:
#         dataset: Dict,
#         initial_obss: np.ndarray
#         max_return: float
#     '''
#     dataset = env.get_dataset(h5path=input_path)

#     N = dataset['rewards'].shape[0] # number of data (s,a,r)
#     data_ = collections.defaultdict(list)

#     use_timeouts = False
#     if 'timeouts' in dataset:
#         use_timeouts = True

#     episode_step = 0
#     paths = []
#     # obs_ = []
#     # next_obs_ = []
#     # action_ = []
#     # reward_ = []
#     # done_ = []
#     # rtg_ = []

#     for i in range(N): # Loop through data points

#         done_bool = bool(dataset['terminals'][i])
#         if use_timeouts:
#             final_timestep = dataset['timeouts'][i]
#         else:
#             final_timestep = (episode_step == 1000-1)
#         for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
#             data_[k].append(dataset[k][i])
            
#         # obs_.append(dataset['observations'][i].astype(np.float32))
#         # next_obs_.append(dataset['next_observations'][i].astype(np.float32))
#         # action_.append(dataset['actions'][i].astype(np.float32))
#         # reward_.append(dataset['rewards'][i].astype(np.float32))
#         # done_.append(bool(dataset['terminals'][i]))

#         if done_bool or final_timestep:
#             episode_step = 0
#             episode_data = {}
#             for k in data_:
#                 episode_data[k] = np.array(data_[k])
#             # Update rtg
#             rtg_traj = discount_cumsum(np.array(data_['rewards']))
#             episode_data['rtgs'] = rtg_traj
#             # rtg_ += rtg_traj

#             paths.append(episode_data)
#             data_ = collections.defaultdict(list)

#         episode_step += 1

#     init_obss = np.array([p['observations'][0] for p in paths]).astype(np.float32)

#     returns = np.array([np.sum(p['rewards']) for p in paths])
#     num_samples = np.sum([p['rewards'].shape[0] for p in paths])
#     print(f'Number of samples collected: {num_samples}')
#     print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

#     if data_path is not None:
#         with open(data_path, 'wb') as f:
#             pickle.dump(paths, f)

#     # print(f"N={N},len(obs_)={len(obs_)},len(reward_)={len(reward_)},len(rtg_)={len(rtg_)}!")
#     # assert len(obs_) == len(rtg_), f"Got {len(obs_)} obss, but {len(rtg_)} rtgs!"

#     # Concatenate paths into one dataset
#     full_dataset = {}
#     for k in ['observations', 'next_observations', 'actions', 'rewards', 'rtgs', 'terminals']:
#         full_dataset[k] = np.concatenate([p[k] for p in paths], axis=0)

#     return full_dataset, init_obss, np.max(returns)