'''
Create the Dataset Class from csv file
'''

import torch
from torch.utils.data import Dataset
# import pandas as pd
import numpy as np
from typing import List, Dict
from offlinerlkit.utils.trajectory import Trajectory

    

class ObsActDataset(Dataset):
    '''
    For diffusion policy training
    '''
    def __init__(self, dataset: Dict[str, np.ndarray]):
        '''
        dataset: A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
        '''
        
        self.observations = dataset['observations']
        self.actions = dataset['actions']

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs_sample = self.observations[idx]
        act_sample = self.actions[idx]
        return dict(obs=obs_sample, action=act_sample)

class DictDataset(Dataset):
    '''
    From Dict to dataset
    '''
    def __init__(self, dict_dataset: Dict[str, np.ndarray]):
        self.dataset = dict_dataset

        # 'obss' and 'next_obss' key may have different names, store its name
        if 'obss' in self.dataset.keys():
            self.obss_key = 'obss'
            self.next_obss_key = 'next_obss'
        else:
            self.obss_key = 'observations'
            self.next_obss_key = 'next_observations'


    def __len__(self):
        return len(self.dataset[self.obss_key])
    
    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        '''
        Return: Dict, with keys same as dict_dataset. Contain:
            observations:
            actions:
            next_observations:
            terminals:
            rewards:
            rtgs:
            (weights):
        '''
        if 'weights' in self.dataset:
            return {
                'observations': self.dataset[self.obss_key][index],
                'next_observations': self.dataset[self.next_obss_key][index],
                'actions': self.dataset['actions'][index],
                'terminals': self.dataset['terminals'][index],
                'rewards': self.dataset['rewards'][index],
                'rtgs': self.dataset['rtgs'][index],
                'weights': self.dataset['weights'][index]
            }
        else:
            return {
                'observations': self.dataset[self.obss_key][index],
                'next_observations': self.dataset[self.next_obss_key][index],
                'actions': self.dataset['actions'][index],
                'terminals': self.dataset['terminals'][index],
                'rewards': self.dataset['rewards'][index],
                'rtgs': self.dataset['rtgs'][index]
            }            

class TrajCtxFloatLengthDataset(Dataset):
    '''
    Son of the pytorch Dataset class
    Provides context length, no next state.
    Trajectory length is uncertain
    '''

    def __init__(self, trajs, ctx = 1, single_timestep = False, keep_ctx = True, with_mask=False, state_normalize=False):    
        '''
        trajs: list(traj), namedtuple with attributes "observations", "actions", "rewards", "returns", "timesteps" \n
        single_timestep: bool. If true, timestep only keep initial step; Else (ctx,) \n
        keep_ctx: If False, ctx must be set 1, and we will not keep ctx dimension.
        with_mask: If true, also return attention mask. For DT
        state_normalize: If true, normalize states
        Note: Each traj must have same number of timesteps
        '''    
        self._trajs = trajs
        self._trajectory_num = len(self._trajs)
        self._horizon = len(self._trajs[0].observations)
        self.keep_ctx = keep_ctx
        self.with_mask = with_mask

        if not keep_ctx:
            assert ctx == 1, f"When keep_ctx = False, ctx must be 1"

        self.ctx = ctx
        self.single_timestep = single_timestep

        self.state_normalize = state_normalize

        if state_normalize:
            states_list = []
            for traj in trajs:
                states_list += traj.observations
            states = np.concatenate(states_list, axis = 0)
            self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        else:
            self.state_mean = 0
            self.state_std = 1

        self.traj_start_idxs = [] # The index of each traj's start
        cnt = 0
        self.traj_idx_list = [] # maintain the traj_idx of each idx
        for i,traj in enumerate(trajs):
            self.traj_start_idxs.append(cnt)
            traj_len = len(traj.rewards)
            self.traj_idx_list += [i for _ in range(traj_len)]
            cnt += traj_len
        self.traj_start_idxs.append(cnt) # Last idx is the total number of data
    
    def __len__(self):
        return self.traj_start_idxs[-1]
    
    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        '''
        Update: Also train incomplete contexts. Incomplete contexts pad 0.
        Input: idx, int, index to get an RTG trajectory slice from dataset \n
        Return: An RTG trajectory slice with length ctx_length \n
        - states: Tensor of size [ctx_length, state_space_size]
        - actions: Tensor of size [ctx_length, action_dim], here action is converted to one-hot representation
        - rewards: Tensor of size [ctx_length, 1]
        - rtgs: Tensor of size [ctx_length, 1]
        - timesteps: (ctx_length) if single_timestep=False; else (1,), only keep the first timestep
        Note: if keep_ctx = False, all returns above will remove the first dim. In particular, timesteps becomes scalar.
        '''

        ctx = self.ctx # context length
        trajectory_idx = self.traj_idx_list[idx]
        res_idx = idx - self.traj_start_idxs[trajectory_idx]

        # Test whether it is full context length
        if res_idx - ctx + 1 < 0:
            start_idx = 0
            pad_len = ctx - res_idx - 1 # number of zeros to pad
        else:
            start_idx = res_idx - ctx + 1
            pad_len = 0

        traj = self._trajs[trajectory_idx]
        states_slice = torch.from_numpy(np.array(traj.observations)[start_idx : res_idx + 1, :])
        states_slice = (states_slice - self.state_mean) / self.state_std

        actions_slice = torch.from_numpy(np.array(traj.actions)[start_idx : res_idx + 1, :])
        rewards_slice = torch.from_numpy(np.array(traj.rewards)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)
        rtgs_slice = torch.from_numpy(np.array(traj.returns)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)

        # pad 0
        states_slice = torch.cat([torch.zeros(pad_len, states_slice.shape[-1]), states_slice], dim = 0)
        actions_slice = torch.cat([torch.zeros(pad_len, actions_slice.shape[-1]), actions_slice], dim = 0)
        rewards_slice = torch.cat([torch.zeros(pad_len, rewards_slice.shape[-1]), rtgs_slice], dim = 0)
        rtgs_slice = torch.cat([torch.zeros(pad_len, rtgs_slice.shape[-1]), rtgs_slice], dim = 0)

        if self.single_timestep: # take the last step
            timesteps_slice = torch.from_numpy(np.array(traj.timesteps)[res_idx : res_idx + 1]) # (1,)
        else: 
            timesteps_slice = torch.from_numpy(np.array(traj.timesteps)[start_idx : res_idx + 1]) #(real_ctx_len, )
            timesteps_slice = torch.cat([torch.zeros(pad_len), timesteps_slice], dim = 0)

        if not self.keep_ctx:
            states_slice = states_slice[0,:]
            actions_slice = actions_slice[0,:]
            rewards_slice = rewards_slice[0,:]
            rtgs_slice = rtgs_slice[0,:]
            timesteps_slice = timesteps_slice[0]

        assert states_slice.shape[0] != 0, f"{idx}, {states_slice.shape}"
        if self.with_mask:
            attn_mask = torch.cat([torch.zeros((pad_len)), torch.ones((ctx-pad_len))], dim=-1)
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask
        else:
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice
    
    def getitem(self, idx):
        if self.with_mask:
            states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask = self.__getitem__(idx)
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask
        else:
            states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice = self.__getitem__(idx)
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice           

    
    def get_max_return(self):
        traj_rets = [traj.returns[0] for traj in self._trajs]
        return max(traj_rets)
    
    def get_normalize_coef(self):
        '''
        Get state normalization mean and std
        '''
        return self.state_mean, self.state_std

class TrajCtxMixSampler:
    '''
    Sample trajs from mixed dataset
    '''
    def __init__(self, datasets: List[List[Trajectory]], weights: List[float], ctx: int) -> None:
        assert len(datasets) == len(weights), f"Datasets {len(datasets)} and weights {len(weights)} must be of same length!"
        assert all(w>=0 for w in weights) and sum(weights)==1, f"Weights must be valid prob. distribution!"
        self.datasets = datasets
        self.weights = weights
        self.ctx = ctx
    def get_batch_traj(self, batch_size: int, with_mask = False):
        '''
        Get a batch from several datasets using weighted sampling.
        ctx: Context length. Pad 0
        with_mask: If True, also returns mask for DT training.
        '''
        datasets = self.datasets
        weights = self.weights
        ctx = self.ctx

        num_samples = [int(batch_size * w) for w in weights]
        num_samples[-1] = batch_size - sum(num_samples[:-1]) # Make num_samples sum up to batch_size

        batch_s, batch_a, batch_r, batch_rtg, batch_t, batch_mask = [], [], [], [], [], []
        for idx, dataset in enumerate(datasets):
            # dataset is a collection of trajs
            num_sample = num_samples[idx]
            num_trajs = len(dataset)
            horizon = len(dataset[0].observations)
            # Get the data indexs for one dataset
            batch_inds = np.random.choice(
                np.arange(num_trajs * horizon),
                size=num_sample,
                replace=True,
            )
            for i in range(num_sample):
                traj_idx = batch_inds[i] // horizon # which trajectory to read, row
                res_idx = batch_inds[i] - traj_idx * horizon # column index to read
                traj = dataset[traj_idx]

                # Test whether it is full context length
                if res_idx - ctx + 1 < 0:
                    start_idx = 0
                    pad_len = ctx - res_idx - 1 # number of zeros to pad
                else:
                    start_idx = res_idx - ctx + 1
                    pad_len = 0

                states_slice = torch.from_numpy(np.array(traj.observations)[start_idx : res_idx + 1, :])

                actions_slice = torch.from_numpy(np.array(traj.actions)[start_idx : res_idx + 1, :])
                rewards_slice = torch.from_numpy(np.array(traj.rewards)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)
                rtgs_slice = torch.from_numpy(np.array(traj.returns)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)

                # pad 0
                states_slice = torch.cat([torch.zeros(pad_len, states_slice.shape[-1]), states_slice], dim = 0)
                actions_slice = torch.cat([torch.zeros(pad_len, actions_slice.shape[-1]), actions_slice], dim = 0)
                rewards_slice = torch.cat([torch.zeros(pad_len, rewards_slice.shape[-1]), rtgs_slice], dim = 0)
                rtgs_slice = torch.cat([torch.zeros(pad_len, rtgs_slice.shape[-1]), rtgs_slice], dim = 0)

                timesteps_slice = torch.from_numpy(np.array(traj.timesteps)[start_idx : res_idx + 1]) #(real_ctx_len, )
                timesteps_slice = torch.cat([torch.zeros(pad_len), timesteps_slice], dim = 0)

                batch_s.append(states_slice.unsqueeze(0))
                batch_a.append(actions_slice.unsqueeze(0))
                batch_r.append(rewards_slice.unsqueeze(0))
                batch_rtg.append(rtgs_slice.unsqueeze(0))
                batch_t.append(timesteps_slice.unsqueeze(0))

                if with_mask:
                    attn_mask = torch.cat([torch.zeros((pad_len)), torch.ones((ctx-pad_len))], dim=-1)
                    batch_mask.append(attn_mask.unsqueeze(0))
        # print(batch_s[0].shape, batch_s[1].shape)
        batch_s = torch.cat(batch_s, dim=0)
        batch_a = torch.cat(batch_a, dim=0)
        batch_r = torch.cat(batch_r, dim=0)
        batch_rtg = torch.cat(batch_rtg, dim=0)
        batch_t = torch.cat(batch_t, dim=0)
        if with_mask:
            batch_mask.append(batch_mask)
            return batch_s, batch_a, batch_r, batch_rtg, batch_t, batch_mask
        else:
            return batch_s, batch_a, batch_r, batch_rtg, batch_t