


import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import wandb
import os
import torch.nn as nn

from typing import Callable, List, Tuple, Dict


from torch.utils.tensorboard import SummaryWriter  

logger = logging.getLogger(__name__)

class Config:
    '''
    A simple class used for configuration
    '''
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)