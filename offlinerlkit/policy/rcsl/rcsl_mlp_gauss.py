# Modified from combo.py

import numpy as np
import torch
from typing import Dict, Union, Tuple, Optional
from offlinerlkit.modules import RcslGaussianModule
from offlinerlkit.policy import BasePolicy


class RcslGaussianPolicy(BasePolicy):
    """
    wrapped rcsl policy
    """

    def __init__(
        self,
        rcsl: RcslGaussianModule,
        rcsl_optim: torch.optim.Optimizer,
        device: Union[str, torch.device]
    ) -> None:
        super().__init__()

        self.rcsl = rcsl      
        self.rcsl_optim = rcsl_optim

        self.device = device

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, rtgs = batch["observations"], batch["actions"], batch["rtgs"]

        mu, logvar = self.rcsl.get_dist_params(obss, rtgs)
        inv_var = torch.exp(-logvar)
        # Average over batch and dim, sum over ensembles.
        mse_loss_inv = (torch.pow(mu - actions.to(mu.device), 2) * inv_var).mean() # MLE for Gaussian
        var_loss = logvar.mean()
        loss = mse_loss_inv + var_loss

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

        mu, logvar = self.rcsl.get_dist_params(obss, rtgs)
        inv_var = torch.exp(-logvar)
        # Average over batch and dim, sum over ensembles.
        mse_loss_inv = (torch.pow(mu - actions.to(mu.device), 2) * inv_var).mean() # MLE for Gaussian
        var_loss = logvar.mean()
        loss = mse_loss_inv + var_loss

        result =  {
            "loss": loss.item(),
        }
        
        return result
    
    def select_action(self, obs: np.ndarray, rtg: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            dist = self.rcsl.forward(obs, rtg)
            action = dist.rsample()
        return action.cpu().numpy()
    
    def train(self) -> None:
        self.rcsl.train()

    def eval(self) -> None:
        self.rcsl.eval()