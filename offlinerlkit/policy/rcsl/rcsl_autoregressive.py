import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict
import numpy as np
from offlinerlkit.utils.soft_clamp import soft_clamp


class AutoregressivePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims, lr, device):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Input is obs + act + one-hot for the predicted dimension
        # Output is the mean and standard deviation of the predicted dimension
        input_dim = obs_dim + 1 + act_dim + act_dim # also depend on return
        all_dims = [input_dim] + hidden_dims + [2]
        self.model = nn.ModuleList()
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            self.model.append(nn.Linear(in_dim, out_dim))
            self.model.append(nn.LeakyReLU())

        self.rcsl_optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.register_parameter(
            "max_logstd",
            nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logstd",
            nn.Parameter(torch.ones(1) * -10, requires_grad=True)
        )
        self.to(self.device)

    def forward(self, obs, rtg, deterministic: bool = False):
        batch_size = obs.size(0)
        rtg = rtg.reshape(batch_size, 1)

        # Initialize action to zeros
        act = torch.zeros((batch_size, self.act_dim), device=obs.device)

        # One-hot encoding for all dimensions
        one_hot_all = torch.eye(self.act_dim, device=obs.device)

        # Predict each dimension autoregressively
        for i in range(self.act_dim):
            one_hot = one_hot_all[i][None, :].repeat(batch_size, 1)
            x = torch.cat([obs, rtg, act, one_hot], dim=1)
            for layer in self.model:
                x = layer(x)
            mean, logstd = torch.chunk(x, 2, dim=-1)
            logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)

            # logstd might be too small
            if deterministic:
                next_dim = mean
            else:
                assert logstd.exp() != float('nan'), f"{logstd}"
                if logstd.exp() == 0:
                    next_dim = mean
                else:
                    dist = Normal(mean, logstd.exp())
                    next_dim = dist.sample()
            act = torch.cat([act[:, :i], next_dim, act[:, i + 1 :]], dim=1)

        return act

    def select_action(self, obs: np.ndarray, rtg: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            rtg = torch.as_tensor(rtg).type(torch.float32).to(self.device)
            action = self.forward(obs, rtg, deterministic)
        return action.cpu().numpy()

    def fit(self, obs, rtg, act, weights = None):
        batch_size = obs.size(0)

        # Generate all the one-hot vectors, expand by repeat
        one_hot_all = torch.eye(self.act_dim, device=obs.device)
        one_hot_full = one_hot_all.repeat_interleave(batch_size, dim=0)

        # Repeat act by act_dim times and mask by one-hot encoding
        mask = (
            torch.tril(torch.ones((self.act_dim, self.act_dim), device=obs.device))
            - one_hot_all
        )  # lower trig - diag
        mask_full = mask.repeat_interleave(batch_size, dim=0)
        act_full = act.repeat(self.act_dim, 1) # (batch*act_dim, act_dim)
        act_masked = act_full * mask_full

        # Repeat obs by act_dim times
        rtg = rtg.reshape(batch_size, 1)
        obs_rtg = torch.cat([obs, rtg], dim = 1)
        obs_rtg_full = obs_rtg.repeat(self.act_dim, 1)

        # Concatenate everything to get input
        input_full = torch.cat([obs_rtg_full, act_masked, one_hot_full], dim=1)

        # Use the one-hot vector as boolean mask to get target
        target = act_full[one_hot_full.bool()].unsqueeze(1)

        # Forward through model and compute loss
        x = input_full
        for layer in self.model:
            x = layer(x)
        mean, logstd = torch.chunk(x, 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        if any(torch.isnan(mean)):
            torch.save(self.model.state_dict(), "model_debug.pth")
            torch.save(input_full, "input_debug.pth")
            raise Exception(f"Mean is nan, input_full {input_full.detach().cpu().numpy()}")
        dist = Normal(mean, logstd.exp())
        loss = -dist.log_prob(target)
        if weights is None:
            loss = loss.mean()
        else:
            loss = loss.reshape(loss.shape[0], -1) # (batch * act_dim, 1)
            weights = weights.reshape(weights.shape[0], -1) # (batch, 1)
            weights = weights.repeat(self.act_dim, 1) # (batch * act_dim, 1)
            loss = torch.sum(loss * weights) / (torch.sum(weights) * loss.shape[-1])
        return loss
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, rtgs = batch["observations"], batch["actions"], batch["rtgs"]
        obss = obss.type(torch.float32).to(self.device)
        actions = actions.type(torch.float32).to(self.device)
        rtgs = rtgs.type(torch.float32).to(self.device)
        if 'weights' in batch:
            weights = batch['weights'].type(torch.float32).to(self.device) # (batch, )
        else:
            weights = None
        loss = self.fit(obss, rtgs, actions,weights)

        self.rcsl_optim.zero_grad()
        loss.backward()
        self.rcsl_optim.step()

        result =  {
            "loss": loss.item(),
        }
        
        return result

    def validate(self, batch: Dict) -> Dict[str, float]:
        obss, actions, rtgs = batch["observations"], batch["actions"], batch["rtgs"]
        obss = obss.type(torch.float32).to(self.device)
        actions = actions.type(torch.float32).to(self.device)
        rtgs = rtgs.type(torch.float32).to(self.device)
        if 'weights' in batch:
            weights = batch['weights'].type(torch.float32).to(self.device) # (batch, )
        else:
            weights = None
        with torch.no_grad():
            loss = self.fit(obss, rtgs, actions, weights)
        return {
            "holdout_loss": loss.item()
        }


if __name__ == "__main__":
    model = AutoregressivePolicy(10, 5, [32, 32])
    obs = torch.randn(32, 10)
    act = torch.randn(32, 5)

    # Test forward
    act_pred = model(obs)
    print(act_pred.shape)

    # Test fit
    loss = model.fit(obs, act)
    print(loss)
