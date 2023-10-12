import time
import os

import numpy as np
import torch
import gym
import gymnasium

from typing import Optional, Dict, List, Tuple, Union
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy


# model-based policy trainer
class MBPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: Union[gym.Env, gymnasium.Env],
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0,
        horizon: Optional[int] = None,
        has_terminal = False,
        binary_ret = False
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.horizon = horizon
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger

        self._rollout_freq, self._rollout_batch_size, \
            self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.is_gymnasium_env = hasattr(self.eval_env, "get_true_observation")
        assert (not self.is_gymnasium_env) or (self.horizon is not None), "Horizon must be specified for Gymnasium env"
        self.has_terminal = has_terminal
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
                if num_timesteps % self._rollout_freq == 0: # rollout periodically
                    init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
                    self.fake_buffer.add_batch(**rollout_transitions)
                    self.logger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    )
                    for _key, _value in rollout_info.items():
                        self.logger.logkv_mean("rollout_info/"+_key, _value)

                # Sample from both real (offline data) and fake (rollout data) according to real_ratio
                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size
                real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                    dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
                    for k, v in dynamics_update_info.items():
                        self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            if last_eval and e < self._epoch: # When last_eval is True, only evaluate on last epoch
                pass
            else:
                # evaluate current policy
                eval_info = self._evaluate()
                ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])

                if not hasattr(self.eval_env, "get_normalized_score"): # gymnasium_env does not have normalized score
                    last_10_performance.append(ep_reward_mean)
                    self.logger.logkv("eval/episode_reward", ep_reward_mean)
                    self.logger.logkv("eval/episode_reward_std", ep_reward_std)         
                else:       
                    norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                    norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                    last_10_performance.append(norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
                self.logger.logkv("eval/episode_length", ep_length_mean)
                self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        is_gymnasium_env = self.is_gymnasium_env
        
        self.policy.eval()
        if is_gymnasium_env:
            obs, _ = self.eval_env.reset()
            obs = self.eval_env.get_true_observation(obs)
        else:
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
                    if is_gymnasium_env:
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
                if is_gymnasium_env:
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
                if is_gymnasium_env:
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
                    if is_gymnasium_env:
                        obs, _ = self.eval_env.reset()
                        obs = self.eval_env.get_true_observation(obs)
                    else:
                        obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }