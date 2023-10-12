import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional
from offlinerlkit.modules.dist_module import DiagGaussian

class RcslModule(nn.Module):
    '''
    rcsl policy network
    '''
    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor], rtg: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        '''
        obs: (batch, obs_dim) 
        rtg: (batch,) / (batch,1)
        '''
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        rtg = torch.as_tensor(rtg, device=self.device, dtype=torch.float32)
        if rtg.dim() == 1:
            rtg = rtg.unsqueeze(-1)
        in_tensor = torch.cat([obs,rtg], dim=-1) #(batch, obs_dim + 1)
        action = self.backbone(in_tensor)
        return action