from collections import namedtuple
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import torch

Trajectory = namedtuple(
    "Trajectory", ["observations", "actions", "next_observations", "rewards", "returns", "timesteps", "terminals", "timeouts"])
'''
Each attribute is also a list, element shape according to env, length equal to horizon
'''
SimpleTrajectory = namedtuple(
    "SimpleTrajectory", ["observations", "actions", "next_observations", "rewards", "returns", "timesteps", "terminals"])

def Dict2Traj(dict_dataset: Dict[str, np.ndarray], num_trajs: int, horizon: int) -> List[SimpleTrajectory]:
    '''
    Convert dict_dataset with keys 'observations', 'next_observations', 'actions', 'rewards', "traj_idxs", "rtgs", "terminals"
    '''
    # Maintain a mapping from traj_idx to idx in trajs. Warning: different trajs may share the same traj_idx
    traj_idxs = list(dict_dataset['traj_idxs']) # sort according to idx
    traj_idxs.sort()
    cur_min = -1
    trajidx2pos = {} # starting position of each traj_idx, may occupy several slots
    for i, idx in enumerate(traj_idxs):
        if idx > cur_min:
            cur_min = idx
            trajidx2pos[idx] = i // horizon

    # Use list first, we will turn to np.ndarray later
    trajs = [
        SimpleTrajectory(
            observations=[],
            actions=[],
            next_observations=[],
            rewards=[],
            returns=[],
            timesteps=[],
            terminals=[]
        )
        for _ in range(num_trajs)
    ]

    idx_cnt = [int(0) for _ in range(len(traj_idxs))] # Count appearance time of each idx
    for i in range(len(dict_dataset['traj_idxs'])):
        traj_idx = dict_dataset['traj_idxs'][i]
        list_pos = trajidx2pos[traj_idx] + idx_cnt[traj_idx] // horizon
        idx_cnt[traj_idx] += 1
        obs = dict_dataset['observations'][i]
        next_obs = dict_dataset['next_observations'][i]
        act = dict_dataset['actions'][i]
        r = dict_dataset['rewards'][i].squeeze()
        rtg = dict_dataset['rtgs'][i].squeeze()
        terminal = dict_dataset['terminals'][i].squeeze()
        trajs[list_pos].observations.append(obs)
        trajs[list_pos].actions.append(act)
        trajs[list_pos].next_observations.append(next_obs)
        trajs[list_pos].rewards.append(r)
        trajs[list_pos].returns.append(rtg)
        trajs[list_pos].timesteps.append(len(trajs[list_pos].timesteps))
        trajs[list_pos].terminals.append(terminal)
    
    # Convert to np.ndarray
    final_trajs = [
        SimpleTrajectory(
            observations=np.asarray(traj.observations),
            actions=np.asarray(traj.actions),
            next_observations=np.asarray(traj.next_observations),
            rewards=np.asarray(traj.rewards),
            returns=np.asarray(traj.returns),
            timesteps=np.asarray(traj.timesteps),  
            terminals=np.asarray(traj.terminals)       
        )
        for traj in trajs
    ]

    return final_trajs
