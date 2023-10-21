import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None,
        init_last: bool = False
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            last_layer = nn.Linear(hidden_dims[-1], output_dim)
            if init_last:
                nn.init.xavier_uniform_(last_layer.weight, gain=1e-2)
                nn.init.constant_(last_layer.bias, 0.0)
            model += [last_layer]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)