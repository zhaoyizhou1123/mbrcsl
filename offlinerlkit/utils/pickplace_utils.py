# Helper functions for pick place environment

from typing import Dict, Optional, Union, Tuple, List
import numpy as np
import gym
from gym.spaces import Box
import os
from collections import namedtuple

def get_true_obs(obs: Dict[str, np.ndarray]) -> np.ndarray :
    '''
    Convert pick place observation to observation needed for training.
    our obs = object_position + object_orientation + state
    Question: Do we need to normalize state?
    '''
    # if type(obs) is not np.ndarray:
    obj_pos = obs['object_position']
    obj_ori = obs['object_orientation']
    state = obs['state']
    return np.concatenate([obj_pos, obj_ori, state], axis = 0)
    # else:
    #     obss = []
    #     for idx in range(len(obs)):
    #         obj_pos = obs[idx]['object_position']
    #         obj_ori = obs[idx]['object_orientation']
    #         state = obs[idx]['state']
    #         obss.append(np.concatenate([obj_pos, obj_ori, state], axis = 0))
    #     return np.asarray(obss)


class SimpleObsWrapper(gym.ObservationWrapper):
    '''
    Wrap pick place environment to return desired obs
    '''
    def __init__(self, env):
        super().__init__(env)
        # Get observation space
        tmp_obs = env.reset()

        tmp_true_obs = get_true_obs(tmp_obs)
        low = env.observation_space['state'].low[0]
        high = env.observation_space['state'].high[0]
        self.observation_space = Box(shape = tmp_true_obs.shape, low = low, high = high)

    def observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        return get_true_obs(observation)

    def reset(self, seed = None):
        if seed is not None:
            np.random.seed(seed) # controls env seed
        return self.observation(self.env.reset())

def get_pickplace_dataset(data_dir: str, prior_weight: float =1., task_weight: float = 1., set_type: str = 'full', sample_ratio: float = 1.) -> Tuple[Dict, np.ndarray]:
    '''
    Concatenate prior_data and task_data
    prior_weight and task_weight: weight of data point

    Args:
        set_type: 'prior', 'task', 'full'
        sample_ratio: Ratio of trajectories sampled. Sometimes we want to train on a smaller dataset.

    Return:
        dataset: Dict, additional key 'weights'
        init_obss: np.ndarray (num_traj, obs_dim)
    '''
    with open(os.path.join(data_dir, 'pickplace_prior.npy'), "rb") as fp:
        prior_data = np.load(fp, allow_pickle=True)
    with open(os.path.join(data_dir, 'pickplace_task.npy'), "rb") as ft:
        task_data = np.load(ft, allow_pickle=True)
    set_weight(prior_data, prior_weight)
    set_weight(task_data, task_weight)

    # Sample trajectories
    num_trajs_prior = int(len(prior_data) * sample_ratio)
    idxs_prior = np.random.choice(len(prior_data), size=(num_trajs_prior), replace = False)
    prior_data = prior_data[idxs_prior]

    num_trajs_task = int(len(task_data) * sample_ratio)
    idxs_task = np.random.choice(len(task_data), size=(num_trajs_task), replace = False)
    task_data = task_data[idxs_task]

    if set_type == 'full':
        full_data = np.concatenate([prior_data, task_data], axis=0) # list of dict
    elif set_type == 'prior':
        full_data = prior_data
    elif set_type =='task':
        full_data = task_data
    keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'weights']

    init_obss = []
    for d in prior_data:
        obs_list = d['observations']
        init_obss.append(get_true_obs(obs_list[0]))
        
    dict_data  = {}
    for key in keys:
        values = []
        for d in full_data: # trajectory, dict of lists
            value_list = d[key] # list of timesteps data
            if key == 'observations':
                values += [get_true_obs(obs) for obs in value_list] # element is list
                # init_obss.append(get_true_obs(value_list[0])) # Get initial observation
            elif key == 'next_observations':
                # print(get_true_obs(value_list[0]))
                values += [get_true_obs(obs) for obs in value_list] # element is list
            else:
                values += value_list # element is list
        values = np.asarray(values)
        dict_data[key] = values
    # dict_data = flatten_dict(full_data, keys)
    rtgs = np.zeros_like(dict_data['rewards']) # no return
    dict_data['rtgs'] = rtgs

    init_obss = np.asarray(init_obss)
    return dict_data, init_obss

def set_weight(dataset: np.ndarray, weight: float):
    for traj in list(dataset):
        traj_len = len(traj['rewards'])
        weights = [weight for _ in range(traj_len)]
        traj['weights'] = weights

def set_weight_dict(dataset: Dict[str, np.ndarray], weight: float):
    dataset_len = len(dataset['rewards'])
    weights = [weight for _ in range(dataset_len)]
    dataset['weights'] = np.asarray(weights)

def merge_dataset(dataset_list: List[Dict[str, np.ndarray]]):
    large_dataset = {}
    for k in ['observations', 'actions', 'rewards', 'next_observations', 'rtgs', 'terminals', 'weights']:
        v_list = [dataset[k].reshape(dataset[k].shape[0],-1) for dataset in dataset_list] # element: (T, dim) or (T,)
        large_dataset[k] = np.concatenate(v_list, axis = 0)
    return large_dataset

# From https://github.com/avisingh599/cog/blob/master/rlkit/data_management/obs_dict_replay_buffer.py
def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {
        key: flatten_n([d[key] for d in dicts])
        for key in keys
    }

if __name__ == '__main__':
    import roboverse
    env = roboverse.make('Widow250PickTray-v0')
    env = SimpleObsWrapper(env)
    dict_data, _ = get_pickplace_dataset("./dataset")
    for k,v in dict_data.items():
        print(k)
        print(v.shape)
    # dic = {1: np.array([1,2])}
    # dicts = [dic, dic, dic]
    # print(flatten_dict(dicts,[1]))

def get_pickplace_dataset_dt(data_dir: str, prior_weight: float =1., task_weight: float = 1., set_type: str = 'full', sample_ratio: float = 1.) -> Tuple[Dict, np.ndarray]:
    '''
    Concatenate prior_data and task_data
    prior_weight and task_weight: weight of data point

    Args:
        set_type: 'prior', 'task', 'full'
        sample_ratio: Ratio of trajectories sampled. Sometimes we want to train on a smaller dataset.

    Return:
        dataset: list trajs: namedtuple with keys "observations", "actions", "rewards", "returns", "timesteps" 
        init_obss: np.ndarray (num_traj, obs_dim)
    '''
    SimpleTrajectory = namedtuple(
    "SimpleTrajectory", ["observations", "actions", "rewards", "returns", "timesteps"])
    with open(os.path.join(data_dir, 'pickplace_prior.npy'), "rb") as fp:
        prior_data = np.load(fp, allow_pickle=True)
    with open(os.path.join(data_dir, 'pickplace_task.npy'), "rb") as ft:
        task_data = np.load(ft, allow_pickle=True)
    set_weight(prior_data, prior_weight)
    set_weight(task_data, task_weight)

    # Sample trajectories
    num_trajs_prior = int(len(prior_data) * sample_ratio)
    idxs_prior = np.random.choice(len(prior_data), size=(num_trajs_prior), replace = False)
    prior_data = prior_data[idxs_prior]

    num_trajs_task = int(len(task_data) * sample_ratio)
    idxs_task = np.random.choice(len(task_data), size=(num_trajs_task), replace = False)
    task_data = task_data[idxs_task]

    if set_type == 'full':
        full_data = np.concatenate([prior_data, task_data], axis=0) # list of dict
    elif set_type == 'prior':
        full_data = prior_data
    elif set_type =='task':
        full_data = task_data
    keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'weights']

    trajs = []
    for traj in full_data:
        last_reward = traj['rewards'][-1]
        rewards = traj['rewards']
        # print(f"obs: {type(traj['observations'][0])}")
        # print(f"actions: {traj['actions'].shape}")
        # print(f"rewards: {traj['rewards'].shape}")
        simple_traj = SimpleTrajectory(
            observations= [get_true_obs(obs) for obs in traj['observations']],
            actions = traj['actions'],
            rewards = rewards,
            returns = [last_reward for _ in range(len(rewards))],
            timesteps= list(range(len(rewards)))
        )
        trajs.append(simple_traj)
    # print(f"Collected {len(trajs)} trajs")
    return trajs