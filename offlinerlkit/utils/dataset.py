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