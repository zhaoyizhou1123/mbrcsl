import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.modules import TransformerDynamicsModel

class TransformerDynamics(BaseDynamics):
    def __init__(
        self,
        model: TransformerDynamicsModel,
        optim: torch.optim.Optimizer,
    ) -> None:
        super().__init__(model, optim)

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        '''
        Return:
            reward (B,1) (if obs has batch)
            terminal (B,1)
        '''
        "imagine single forward step"
        next_obs, reward, _ = self.model.sample(obs, action) # (batch, obs_dim + 1) [reward, obs]

        next_obs = next_obs.cpu().numpy()
        reward = reward.cpu().numpy()

        terminal = np.array([False for _ in range(reward.shape[0])])
        
        return next_obs, reward, terminal, {}

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        rewards = rewards.reshape(rewards.shape[0], -1)
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((rewards, next_obss), axis=-1) # estimate reward first
        if 'weights' in data:
            weights = data['weights']
            weights = weights.reshape(weights.shape[0], -1) # (N,1)
        else:
            weights = None
        return inputs, targets, weights
    
    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: int = 80,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
    ) -> None:
        inputs, targets, weights = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]
        if weights is not None:
            train_weights, holdout_weights = weights[train_splits.indices], weights[holdout_splits.indices]
        else: 
            train_weights, holdout_weights = None, None

        data_idxes = np.arange(train_size)
        np.random.shuffle(data_idxes)

        epoch = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            if train_weights is not None:
                train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], train_weights[data_idxes], batch_size)
            else:
                train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], None, batch_size)
            new_holdout_loss = self.validate(holdout_inputs, holdout_targets, holdout_weights)
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", new_holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            np.random.shuffle(data_idxes)
            
            if epoch >= max_epochs:
                break

        self.save(logger.model_dir)
        self.model.eval()
    
    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray],
        batch_size: int = 256,
    ) -> float:
        '''
        inputs, targets: (N, dim). N is sampled with replacement
        weights: None / (N, 1)
        '''
        self.model.train()
        assert inputs.ndim == 2, f"{inputs.shape}"
        train_size = inputs.shape[0]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[batch_num * batch_size:(batch_num + 1) * batch_size]
            inputs_batch = torch.as_tensor(inputs_batch).type(torch.float32).to(self.model.device)
            targets_batch = targets[batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).type(torch.float32).to(self.model.device)
            if weights is not None:
                weights_batch = weights[batch_num * batch_size:(batch_num + 1) * batch_size]
                weights_batch = torch.as_tensor(weights_batch).type(torch.float32).to(self.model.device)
            else:
                weights_batch is None
            
            loss = self.model.fit(inputs_batch, targets_batch, weights_batch)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray, weights: Optional[np.ndarray]) -> float:
        inputs = torch.as_tensor(inputs).type(torch.float32).to(self.model.device)
        targets = torch.as_tensor(targets).type(torch.float32).to(self.model.device)
        if weights is not None:
            weights = torch.as_tensor(weights).type(torch.float32).to(self.model.device)
        else:
            weights = None
        val_loss = self.model.fit(inputs, targets, weights)
        return val_loss.item()
    

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
    
    def load(self, load_path: str) -> None:
        '''
        load_type: 'all', 'obs', 'r'
        '''
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
