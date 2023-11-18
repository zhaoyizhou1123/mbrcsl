# Modified from combo.py

# TODO: Implement classic RCSL policy from actor_module and dist_module. Follow COMBO class

import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Optional
from collections import defaultdict
from offlinerlkit.policy import BasePolicy
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.modules.rcsl_module import RcslModule


class MLPPolicy(BasePolicy):
    """
    wrapped rcsl policy
    """

    def __init__(
        self,
        rcsl: RcslModule,
        rcsl_optim: torch.optim.Optimizer,
        device: Union[str, torch.device]
    ) -> None:
        super().__init__()

        self.rcsl = rcsl      
        self.rcsl_optim = rcsl_optim

        self.device = device
   
    # One batch update
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, rtgs = batch["observations"], batch["actions"], batch["rtgs"]

        pred_actions = self.rcsl.forward(obss, rtgs)
        # Average over batch and dim, sum over ensembles.
        loss = torch.pow(pred_actions - actions.to(pred_actions.device), 2).mean() # MSE error

        self.rcsl_optim.zero_grad()
        loss.backward()
        self.rcsl_optim.step()

        result =  {
            "loss": loss.item(),
        }
        
        return result

    @ torch.no_grad()
    def validate(self, batch: Dict) -> Dict[str, float]:
        obss, actions, rtgs = batch["observations"], batch["actions"], batch["rtgs"]

        pred_actions = self.rcsl.forward(obss, rtgs)
        # Average over batch and dim, sum over ensembles.
        loss = torch.pow(pred_actions - actions.to(pred_actions.device), 2).mean() # MSE error

        result =  {
            "holdout_loss": loss.item(),
        }
        
        return result
    
    def select_action(self, obs: np.ndarray, rtg: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action = self.rcsl.forward(obs, rtg)
        return action.cpu().numpy()
    
    def train(self) -> None:
        self.rcsl.train()

    def eval(self) -> None:
        self.rcsl.eval()