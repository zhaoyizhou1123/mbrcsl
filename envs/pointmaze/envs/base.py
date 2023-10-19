'''
Modified from https://github.dev/keirp/stochastic_offline_envs
'''

from os import path
import pickle
from ..samplers.trajectory_sampler import BaseSampler


class BaseOfflineEnv:
    '''
    Encapsulates env and dataset
    '''
    def __init__(self, data_path, env_cls, horizon, sampler=None, sample_args=None):
        '''
        data_path: str, path to dataset file
        env_cls: function, return the environment
        data_policy_fn: A function that returns BasePolicy class, can reset/sample/update, as data sampling policy. 
        horizon: int, horizon of each episode
        sampler: BaseSampler | None. Specifies a dataset sampler
        sample_args: arguments for sampler data collection. 
        '''
        self.env_cls = env_cls
        self.horizon = horizon
        self.data_path = data_path
        self.sample_args = sample_args
        if sampler is None:
            self.sampler = BaseSampler()
        else:
            self.sampler = sampler
        if self.data_path is not None and path.exists(self.data_path):
            print('Dataset file found. Loading existing trajectories.')
            with open(self.data_path, 'rb') as file:
                self.dataset = pickle.load(file) # Dataset may not be trajs, might contain other infos
        else:
            print('Dataset file not found. Generating trajectories.')
            self.generate_and_save()

    def generate_and_save(self):
        self.dataset = self.sampler.collect_trajectories(self.sample_args)

        if self.data_path is not None:
            with open(self.data_path, 'wb') as file:
                pickle.dump(self.dataset, file)
                print('Saved trajectories to dataset file.')


def default_path(name):
    # Get the path of the current file
    file_path = path.dirname(path.realpath(__file__))
    # Go up 3 directories
    root_path = path.abspath(path.join(file_path, '..', '..', '..'))
    # Go to offline data directory
    offline_data_path = path.join(root_path, 'offline_data')
    # append the name of the dataset
    return path.join(offline_data_path, name)
