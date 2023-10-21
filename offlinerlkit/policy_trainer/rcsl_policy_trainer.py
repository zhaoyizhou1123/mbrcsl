import time
import os

import numpy as np
import torch
import gym
import gymnasium

from typing import Optional, Dict, List, Tuple, Union
from tqdm import tqdm
from collections import deque
from torch.utils.data import DataLoader
from copy import deepcopy

from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy
from offlinerlkit.utils.dataset import DictDataset


class RcslPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: Union[gym.Env, gymnasium.Env],
        offline_dataset: Dict[str, np.ndarray],
        rollout_dataset: Optional[Dict[str, np.ndarray]],
        goal: float,
        logger: Logger,
        seed,
        eval_env2: Optional[Union[gym.Env, gymnasium.Env]] = None,
        epoch: int = 1000,
        batch_size: int = 256,
        offline_ratio: float = 0,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        horizon: Optional[int] = None,
        num_workers = 1,
        has_terminal = False,
        binary_return = True
    ) -> None:
        '''
        offline_ratio = 0: rollout only, 1: offline only
        '''
        self.policy = policy
        self.eval_env = eval_env
        self.eval_env2 = eval_env2
        self.horizon = horizon
        self.offline_dataset = offline_dataset
        self.rollout_dataset = rollout_dataset
        self.goal = goal
        self.logger = logger

        self._epoch = epoch
        self._batch_size = batch_size
        self._offline_ratio = offline_ratio
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        self.num_workers = num_workers
        self.env_seed = seed
        self.binary_return = binary_return

        self.is_gymnasium_env = hasattr(self.eval_env, "get_true_observation")
        assert (not self.is_gymnasium_env) or (self.horizon is not None), "Horizon must be specified for Gymnasium env"
        self.has_terminal = has_terminal

    def train(self, holdout_ratio: float = 0.1, last_eval = False, find_best_start: Optional[int] = None) -> Dict[str, float]:
        '''
        last_eval: If True, only evaluates at the last epoch
        find_best_start: If >=0, begin to find the best epoch by holdout loss
        '''
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)

        dataset = DictDataset(self.offline_dataset)

        if holdout_ratio == 0.:
            has_holdout = False
            train_dataset = dataset
        else:
            has_holdout = True
            holdout_size = int(len(dataset) * holdout_ratio)
            train_size = len(dataset) - holdout_size
            train_dataset, holdout_dataset = torch.utils.data.random_split(dataset, [train_size, holdout_size], 
                                                                        generator=torch.Generator().manual_seed(self.env_seed))
        data_loader = DataLoader(
            train_dataset,
            batch_size = self._batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = self.num_workers
        )
        best_ep_reward_mean = 1e10
        best_policy_dict = self.policy.state_dict()
        best_holdout_loss = 1e10
        old_train_loss = 1e10
        epochs_since_upd = 0
        stop_by_holdout = (find_best_start is not None)
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(enumerate(data_loader), desc=f"Epoch #{e}/{self._epoch}")
            # losses = []
            for it, batch in pbar:
                '''
                batch: dict with keys
                    'observations'
                    'next_observations'
                    'actions'
                    'terminals'
                    'rewards'
                    'rtgs'

                '''
                loss_dict = self.policy.learn(batch)
                # losses.append(loss_dict['loss'])
                pbar.set_postfix(**loss_dict)

                for k, v in loss_dict.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Test validation loss
            if has_holdout:
                holdout_loss = self.validate(holdout_dataset)
                if stop_by_holdout and e >= find_best_start: # test holdout improvement
                    # loss = sum(losses) / len(losses)
                    if (best_holdout_loss - holdout_loss) / best_holdout_loss > 0.01:
                        best_holdout_loss = holdout_loss
                        best_policy_dict = deepcopy(self.policy.state_dict())
                        # old_train_loss = loss
                        epochs_since_upd = 0
                    # elif best_holdout_loss > holdout_loss and (old_train_loss - loss) / old_train_loss > 0.005:
                    #     best_holdout_loss = holdout_loss
                    #     best_policy_dict = deepcopy(self.policy.state_dict())
                    #     old_train_loss = loss
                    #     epochs_since_upd = 0
                    else:
                        epochs_since_upd += 1

            if last_eval and e < self._epoch: # When last_eval is True, only evaluate on last epoch
                pass
            else:
                eval_info = self._evaluate()
                ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                ep_reward_max, ep_reward_min = np.max(eval_info["eval/episode_reward"]), np.min(eval_info["eval/episode_reward"])
                ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])

                if not hasattr(self.eval_env, "get_normalized_score"): # gymnasium_env does not have normalized score
                    last_10_performance.append(ep_reward_mean)
                    self.logger.logkv("eval/episode_reward", ep_reward_mean)
                    self.logger.logkv("eval/episode_reward_std", ep_reward_std)                    
                else:       
                    norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                    norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                    norm_ep_rew_max = self.eval_env.get_normalized_score(ep_reward_max) * 100
                    norm_ep_rew_min = self.eval_env.get_normalized_score(ep_reward_min) * 100
                    last_10_performance.append(norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
                    self.logger.logkv("eval/normalized_episode_reward_max", norm_ep_rew_max)
                    self.logger.logkv("eval/normalized_episode_reward_min", norm_ep_rew_min)
                self.logger.logkv("eval/episode_length", ep_length_mean)
                self.logger.logkv("eval/episode_length_std", ep_length_std)

                # save checkpoint
                # if ep_reward_mean >= best_ep_reward_mean:
                #     best_ep_reward_mean = ep_reward_mean
                #     best_policy_dict = self.policy.state_dict()
                #     torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy_best.pth"))

            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])

            if stop_by_holdout and epochs_since_upd >= 5: # Stop, evaluate for the last time
                self.policy.load_state_dict(best_policy_dict)
                eval_info = self._evaluate()
                ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                self.logger.log(f"Final evaluation: Mean {ep_reward_mean}, std {ep_reward_std}\n")
                break
        
        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy_final.pth"))
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self, eval_episodes: int = -1) -> Dict[str, List[float]]:
        '''
        Always set desired rtg to 0
        '''
        # Pointmaze obs has different format, needs to be treated differently
        if eval_episodes == -1:
            real_eval_episodes = self._eval_episodes
        else:
            real_eval_episodes = eval_episodes
        is_gymnasium_env = self.is_gymnasium_env

        self.eval_env.reset(seed=self.env_seed) # Fix seed
        
        self.policy.eval()
        if is_gymnasium_env:
            obs, _ = self.eval_env.reset()
            obs = self.eval_env.get_true_observation(obs)
        else:
            obs = self.eval_env.reset()
            
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        if not self.has_terminal: # don't use terminal signal, terminate when reach horizon
            while num_episodes < real_eval_episodes:
                rtg = torch.tensor([[self.goal]]).type(torch.float32)
                for timestep in range(self.horizon): # One epoch
                    action = self.policy.select_action(obs.reshape(1, -1), rtg)
                    if hasattr(self.eval_env, "get_true_observation"): # gymnasium env 
                        next_obs, reward, terminal, _, _ = self.eval_env.step(action.flatten())
                    else:
                        next_obs, reward, terminal, info = self.eval_env.step(action.flatten())
                    if is_gymnasium_env:
                        next_obs = self.eval_env.get_true_observation(next_obs)
                    episode_reward += reward
                    rtg = rtg - reward
                    episode_length += 1

                    obs = next_obs
                if self.binary_return:
                    episode_reward = 1 if episode_reward > 0 else 0 # Clip to 1
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                if is_gymnasium_env:
                    obs, _ = self.eval_env.reset()
                    obs = self.eval_env.get_true_observation(obs)
                else:
                    obs = self.eval_env.reset()
        else:
            rtg = torch.tensor([[self.goal]]).type(torch.float32)
            while num_episodes < self._eval_episodes:
                action = self.policy.select_action(obs.reshape(1, -1), rtg)
                if hasattr(self.eval_env, "get_true_observation"): # gymnasium env 
                    next_obs, reward, terminal, _, _ = self.eval_env.step(action.flatten())
                else:
                    next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
                if is_gymnasium_env:
                    next_obs = self.eval_env.get_true_observation(next_obs)
                episode_reward += reward
                episode_length += 1

                obs = next_obs

                if terminal: # Episode finishes
                    if self.binary_return:
                        episode_reward = 1 if episode_reward > 0 else 0 # Clip to 1
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward, "episode_length": episode_length}
                    )
                    episode_reward, episode_length = 0, 0
                    if is_gymnasium_env:
                        obs, _ = self.eval_env.reset()
                        obs = self.eval_env.get_true_observation(obs)
                    else:
                        obs = self.eval_env.reset()
                    rtg = torch.tensor([[self.goal]]).type(torch.float32)
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
    
    @ torch.no_grad()
    def validate(self, holdout_dataset: torch.utils.data.Dataset) -> Optional[float]:
        data_loader = DataLoader(
            holdout_dataset,
            batch_size = self._batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = self.num_workers
        )
        self.policy.eval()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        losses = []
        for it, batch in pbar:
            '''
            batch: dict with keys
                'observations'
                'next_observations'
                'actions'
                'terminals'
                'rewards'
                'rtgs'
            '''
            loss_dict = self.policy.validate(batch)

            for k, v in loss_dict.items():
                self.logger.logkv_mean(k, v)

            if "holdout_loss" in loss_dict:
                loss = loss_dict["holdout_loss"]
                losses.append(loss)

        if len(losses) > 0:
            return(sum(losses) / len(losses))
        else:
            return None