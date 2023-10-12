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

from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy
from offlinerlkit.utils.dataset import DictDataset

class DiffusionPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        offline_dataset: Dict[str, np.ndarray],
        logger: Logger,
        seed,
        epoch: int = 25,
        batch_size: int = 256,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        horizon: Optional[int] = None,
        num_workers = 1,
        has_terminal = False
    ) -> None:
        '''
        offline_ratio = 0: rollout only, 1: offline only
        '''
        self.policy = policy
        self.horizon = horizon
        self.offline_dataset = offline_dataset
        self.logger = logger

        self._epoch = epoch
        self._batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.num_workers = num_workers
        self.env_seed = seed
        self.has_terminal = has_terminal

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)

        data_loader = DataLoader(
            DictDataset(self.offline_dataset),
            batch_size = self._batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = self.num_workers
        )       

        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(enumerate(data_loader), desc=f"Epoch #{e}/{self._epoch}")
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
                pbar.set_postfix(**loss_dict)

                for k, v in loss_dict.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance)}

