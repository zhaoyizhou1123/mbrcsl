import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy


# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        horizon: Optional[int] = None,
        has_terminal = False,
        binary_ret = False
    ) -> None:
        '''
        binary_ret: If True, only output 0/1 for return
        '''
        self.policy = policy
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.is_gymnasium_env = hasattr(self.eval_env, "get_true_observation")
        assert (not self.is_gymnasium_env) or (self.horizon is not None), "Horizon must be specified for Gymnasium env"
        self.has_terminal = has_terminal
        self.horizon = horizon
        self.binary_ret = binary_ret

    def train(self, last_eval = False) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                batch = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            if last_eval and e < self._epoch: # When last_eval is True, only evaluate on last epoch
                pass
            else:
                eval_info = self._evaluate()
                ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
                if hasattr(self.eval_env, "get_normalized_score"):
                    norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                    norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                    last_10_performance.append(norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
                self.logger.logkv("eval/episode_reward", ep_reward_mean)
                self.logger.logkv("eval/episode_reward_std", ep_reward_std)

                self.logger.logkv("eval/episode_length", ep_length_mean)
                self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        if not self.has_terminal: # Finite horizon, terminal is unimportant
            while num_episodes < self._eval_episodes:
                for timestep in range(self.horizon): # One epoch
                    # print(f"Timestep {timestep}, obs {obs}")
                    action = self.policy.select_action(obs.reshape(1, -1), deterministic=True)
                    if hasattr(self.eval_env, "get_true_observation"): # gymnasium env 
                        next_obs, reward, terminal, _, _ = self.eval_env.step(action.flatten())
                    else:
                        next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
                    if self.is_gymnasium_env:
                        next_obs = self.eval_env.get_true_observation(next_obs)
                    episode_reward += reward
                    episode_length += 1

                    obs = next_obs

                if self.binary_ret:
                    episode_reward = 1 if episode_reward >= 1 else 0
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                if self.is_gymnasium_env:
                    obs, _ = self.eval_env.reset()
                    obs = self.eval_env.get_true_observation(obs)
                else:
                    obs = self.eval_env.reset()
        else:
            while num_episodes < self._eval_episodes:
                action = self.policy.select_action(obs.reshape(1, -1), deterministic=True)
                if hasattr(self.eval_env, "get_true_observation"): # gymnasium env 
                    next_obs, reward, terminal, _, _ = self.eval_env.step(action.flatten())
                else:
                    next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
                if self.is_gymnasium_env:
                    next_obs = self.eval_env.get_true_observation(next_obs)
                episode_reward += reward
                episode_length += 1

                obs = next_obs

                if terminal: # Episode finishes
                    if self.binary_ret:
                        episode_reward = 1 if episode_reward >= 1 else 0
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward, "episode_length": episode_length}
                    )
                    num_episodes +=1
                    episode_reward, episode_length = 0, 0
                    if self.is_gymnasium_env:
                        obs, _ = self.eval_env.reset()
                        obs = self.eval_env.get_true_observation(obs)
                    else:
                        obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
