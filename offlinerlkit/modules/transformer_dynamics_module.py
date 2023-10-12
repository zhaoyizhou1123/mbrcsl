# Predict both next_obs and reward

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional, Dict, Tuple

from offlinerlkit.modules.gpt import GPTConfig
from offlinerlkit.utils.transformer_utils import top_k_logits, Discretizer
from offlinerlkit.utils.logger import Logger
# from offlinerlkit.dynamics import BaseDynamics


class TransformerDynamicsModel(nn.Module):
    def __init__(
        self, obs_dim, act_dim, obs_min, obs_max, act_min, act_max,
        r_min, r_max,
        ckpt_dir: str, device="cpu",
        n_layer = 4, n_head= 4, n_embd = 32
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self.device = device
        # self.logger = logger

        block_size = (
            self._obs_dim + self._act_dim + self._obs_dim + 1 - 1
        )  # -1 since only need n-1 autoregressive steps, + 1 for reward
        vocab_size = 500

        conf = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            discrete_dim=self._obs_dim + self._act_dim + self._obs_dim + 1,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd * n_head,
            savepath=ckpt_dir,
        )

        self._model = conf.make(self.device)
        self._target_model = copy.deepcopy(self._model)
        self._obs_discretizer = Discretizer(obs_min, obs_max, vocab_size)
        self._act_discretizer = Discretizer(act_min, act_max, vocab_size)
        self._r_discretizer = Discretizer(r_min, r_max, vocab_size)

    def configure_optimizer(self, lr, weight_decay, betas):
        return self._model.configure_optimizer(lr, weight_decay, betas)

    def fit(self, obs_act, r_next_obs, weight):
        # State marginal conditioned on the initial state
        obs = obs_act[:, :self._obs_dim]
        act = obs_act[:, self._obs_dim:]
        next_obs = r_next_obs[:, 1:]
        rew = r_next_obs[:, :1]
        obs_discrete, obs_recon, obs_error = self._obs_discretizer(obs)
        act_discrete, act_recon, act_error = self._act_discretizer(act)
        next_obs_discrete, _ , _ = self._obs_discretizer(next_obs)
        rew_discrete, _, _ = self._r_discretizer(rew)
        target_discrete = torch.cat([rew_discrete, next_obs_discrete], dim = -1)
        input_discrete = torch.cat([obs_discrete, act_discrete], dim = -1)
        logits, loss_p = self._model(input_discrete, targets=target_discrete)
        return loss_p

    @torch.no_grad()
    def sample(self, obs, act, temperature=1.0, top_k=None):
        # Discretize observation
        obs = self._obs_discretizer.discretize(obs)
        act = self._act_discretizer.discretize(act)

        batch_size = len(obs)
        total_probs = torch.zeros(batch_size, device=self.device)
        block_size = self._model.get_block_size()
        self._model.eval()

        x = np.concatenate([obs, act], axis= -1)
        x = torch.as_tensor(x).to(self.device)
        for k in range(self._obs_dim + 1):
            x_cond = x
            if x_cond.shape[1] > block_size:
                raise RuntimeError("Sequence length greater than block size")
            logits, _ = self._model(x_cond)
            # Pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            probs_for_calc = F.softmax(logits, dim=-1)
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample next token
            ix = torch.multinomial(probs, num_samples=1)
            # Compute conditional probability
            pr = probs_for_calc[torch.arange(batch_size), ix.squeeze()]
            terminated = (ix == self._model.vocab_size).squeeze()
            pr[terminated] = -10
            total_probs += pr
            # Append action to the sequence and continue
            x = torch.cat((x, ix), dim=1)

        # Reconstruct next_obs
        next_obs = x[:, -self._obs_dim:]
        rew = x[:, -self._obs_dim-1:-self._obs_dim]
        next_obs = self._obs_discretizer.reconstruct_torch(next_obs)
        rew = self._r_discretizer.reconstruct_torch(rew)
        return next_obs, rew, total_probs
