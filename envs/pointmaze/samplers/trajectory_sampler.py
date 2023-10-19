'''
Modified from https://github.dev/keirp/stochastic_offline_envs
'''

from ..samplers.base import BaseSampler
from collections import namedtuple
from random import randint
from tqdm.autonotebook import tqdm
from copy import deepcopy

Trajectory = namedtuple(
    "Trajectory", ["obs", "actions", "rewards", "infos", "policy_infos"])


class TrajectorySampler(BaseSampler):
    """Collects full trajectories. If there is an unfinished trajectory
    (t < horizon or not `done`), don't return those transitions."""

    def __init__(self,
                 env_cls,
                 policy,
                 horizon,
                 obs_key=None
                 ):
        '''
        obs_key: used to take part of the obs. I think we just use None
        '''
        self.env_cls = env_cls
        self.env = self.env_cls()
        self.policy = policy
        self.horizon = horizon
        self.obs_key = obs_key

    def collect_trajectories(self, n_interactions, n_trajs=None):
        """Collect at most n_interactions. If n_trajs is not None,
        collecta at most n_trajs trajectories."""

        if n_interactions is not None:
            print(
                f'Using {self.policy.name} to gather {n_interactions} interactions.')

        trajs = []

        n_gathered = 0

        self.policy.reset()

        obs_ = []
        actions_ = []
        rewards_ = []
        infos_ = []
        policy_infos_ = []
        t = 0

        env = self.env

        obs = env.reset()
        reward = None

        pbar = tqdm(total=n_interactions)

        while n_interactions is None or n_gathered < n_interactions:
            if self.obs_key is not None:
                obs = obs[self.obs_key]
            obs_.append(deepcopy(obs))

            action, policy_info = self.policy.sample(obs, reward, t)

            policy_infos_.append(policy_info)
            actions_.append(action)

            obs, reward, done, info = env.step(action)

            t += 1
            infos_.append(info)
            rewards_.append(reward)

            n_gathered += 1
            pbar.update(1)

            if t == self.horizon or done:
                trajs.append(Trajectory(obs=obs_,
                                        actions=actions_,
                                        rewards=rewards_,
                                        infos=infos_,
                                        policy_infos=policy_infos_))
                t = 0
                obs_ = []
                actions_ = []
                rewards_ = []
                infos_ = []
                policy_infos_ = []

                if n_trajs is not None and len(trajs) == n_trajs:
                    break

                obs = env.reset()
                self.policy.reset()
                reward = None

        pbar.close()

        return trajs
