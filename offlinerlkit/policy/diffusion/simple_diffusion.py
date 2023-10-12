import math

import numpy as np
import torch
import torch.nn as nn
from typing import Dict

from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.training_utils import EMAModel
from .ema import EMAModel


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualLinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                LinearBlock(input_dim, output_dim),
                LinearBlock(output_dim, output_dim),
            ]
        )

        # FiLM modulation
        self.film = nn.Linear(cond_dim, output_dim * 2)

        # Make sure dimensions are compatible
        self.residual_fc = (
            nn.Linear(input_dim, output_dim, 1)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.blocks[0](x)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        out = gamma * out + beta
        out = self.blocks[1](out)
        out = out + self.residual_fc(x)
        return out


class ConditionalResNet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        embed_dim=256,
        hidden_dims=[512, 512, 512],
    ):
        super().__init__()
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        all_dims = [input_dim] + hidden_dims + [input_dim]
        cond_dim = embed_dim + global_cond_dim

        self.blocks = nn.ModuleList()
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            self.blocks.append(
                ConditionalResidualLinearBlock(in_dim, out_dim, cond_dim)
            )

    def forward(self, sample, timestep, global_cond):
        # Expand integer timestep
        if len(timestep.shape) == 0:
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
            timestep = timestep.expand(sample.shape[0])

        # Encode timestep
        embed = self.diffusion_step_encoder(timestep)
        cond = torch.cat([embed, global_cond], axis=-1)

        # Forward through model
        x = sample
        for block in self.blocks:
            x = block(x, cond)
        return x


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim, # should be obs_dim + 1
        embed_dim=256,
        down_dims=[256, 512, 1024],
    ):
        # print(f"Unet: global_cond_dim = {global_cond_dim}")
        super().__init__()
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        start_dim = down_dims[0]
        mid_dim = down_dims[-1]
        all_dims = [input_dim] + down_dims
        in_out_dims = list(zip(all_dims[:-1], all_dims[1:]))
        cond_dim = embed_dim + global_cond_dim # 256 + obs_dim + 1 = 274

        # Down modules
        self.down_modules = nn.ModuleList()
        for in_dim, out_dim in in_out_dims:
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualLinearBlock(in_dim, out_dim, cond_dim),
                        ConditionalResidualLinearBlock(out_dim, out_dim, cond_dim),
                    ]
                )
            )

        # Mid modules
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualLinearBlock(mid_dim, mid_dim, cond_dim),
                ConditionalResidualLinearBlock(mid_dim, mid_dim, cond_dim),
            ]
        )

        # Up modules
        self.up_modules = nn.ModuleList()
        for in_dim, out_dim in reversed(in_out_dims[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualLinearBlock(out_dim * 2, in_dim, cond_dim),
                        ConditionalResidualLinearBlock(in_dim, in_dim, cond_dim),
                    ]
                )
            )

        # Final linear layers
        self.final_linear = nn.Sequential(
            LinearBlock(start_dim, start_dim),
            nn.Linear(start_dim, input_dim),
        )

    def forward(self, sample, timestep, global_cond):
        # Expand integer timestep
        if len(timestep.shape) == 0:
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
            timestep = timestep.expand(sample.shape[0])

        # Encode timestep
        embed = self.diffusion_step_encoder(timestep)
        cond = torch.cat([embed, global_cond], axis=-1)

        # Forward through model
        x = sample
        h = []
        for resnet, resnet2 in self.down_modules:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            h.append(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, cond)

        for resnet, resnet2 in self.up_modules:
            x = torch.cat((x, h.pop()), dim=-1)
            x = resnet(x, cond)
            x = resnet2(x, cond)

        x = self.final_linear(x)
        return x


class ConditionalDiffusionModel:
    def __init__(
        self,
        input_dim,
        cond_shape_dict, # {"obs": obs_shape, "feat": (feature_dim,)}
        num_training_steps,
        num_diffusion_steps,
        clip_sample,
        device,
        lr=1e-4,
        weight_decay=1e-6,
        num_warmup_steps=500,
        model_cls="unet",
    ):
        self.input_dim = input_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device

        # Condition encoders
        self.cond_encoders = nn.ModuleDict()
        self.cond_dim = 0
        for name, shape in cond_shape_dict.items():
            if len(shape) == 1:
                self.cond_encoders[name] = nn.Identity()
                self.cond_dim += shape[0]
            else:
                raise NotImplementedError
        self.cond_encoders.to(self.device)

        # Noise prediction net
        model_cls = ConditionalUnet1D if model_cls == "unet" else ConditionalResNet1D
        self.noise_pred_net = model_cls(
            input_dim=self.input_dim,
            global_cond_dim=self.cond_dim, # should be obs_dim + 1
        ).to(self.device)

        # Optimizer
        self.model_params = list(self.noise_pred_net.parameters()) + list(
            self.cond_encoders.parameters()
        )
        self.optimizer = torch.optim.AdamW(
            self.model_params, lr=lr, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=clip_sample,
        )

        # Exponential moving average
        self.ema = EMAModel(self.noise_pred_net.parameters(), power=0.75)
        # self.ema = EMAModel(self.noise_pred_net, power=0.75)

    def encode_conditions(self, cond_dict):
        return torch.cat(
            [self.cond_encoders[k](v) for k, v in cond_dict.items()], dim=-1
        )

    @ torch.no_grad()
    def validate(self, x, cond_dict, weights = None):
        noise = torch.randn_like(x)

        # Sample a diffusion timestep for each data point
        timestep = torch.randint(
            low=0,
            high=self.num_diffusion_steps,
            size=(x.shape[0],),
            device=self.device,
        ).long()

        # Forward diffusion process
        x_t = self.noise_scheduler.add_noise(x, noise, timestep)

        # Predict the noise residual
        cond = self.encode_conditions(cond_dict)
        noise_pred = self.noise_pred_net(x_t, timestep, cond)

        # Compute loss
        if weights is None:
            loss = nn.functional.mse_loss(noise_pred, noise)
        else:
            loss = self.weighted_mse_loss(noise_pred, noise, weights)
        result =  {
            "holdout_loss": loss.item(),
        }
        
        return result


    def learn(self, x, cond_dict, weights=None):
        '''
        train on one batch

        weights: tensor (batch,)
        '''
        # Sample noise to add to data
        noise = torch.randn_like(x)

        # Sample a diffusion timestep for each data point
        timestep = torch.randint(
            low=0,
            high=self.num_diffusion_steps,
            size=(x.shape[0],),
            device=self.device,
        ).long()

        # Forward diffusion process
        x_t = self.noise_scheduler.add_noise(x, noise, timestep)

        # Predict the noise residual
        cond = self.encode_conditions(cond_dict)
        noise_pred = self.noise_pred_net(x_t, timestep, cond)

        # Compute loss
        if weights is None:
            loss = nn.functional.mse_loss(noise_pred, noise)
        else:
            loss = self.weighted_mse_loss(noise_pred, noise, weights)

        # Step optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Step lr scheduler every batch
        # self.lr_scheduler.step()

        # Update exponential moving average
        # self.ema.step(self.noise_pred_net)
        self.ema.step(self.model_params)

        result =  {
            "loss": loss.item(),
        }
        
        return result

    def get_lr_scheduler(self):
        '''
        Return lr_scheduler
        '''
        return self.lr_scheduler

    def sample(self, cond_dict, w=None):
        '''
        cond_dict: {"obs": obs, "feat": feat}
        '''
        # Copy EMA model weights
        self.ema.store(self.model_params)
        self.ema.copy_to(self.model_params)

        # Encode conditions
        cond = self.encode_conditions(cond_dict)

        # print(f"Diff model: cond shape {cond.shape}")

        # Initialize sample
        sample = torch.randn((len(cond), self.input_dim), device=self.device)

        # Initialize scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_steps)

        # Reverse diffusion process
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.noise_pred_net(sample, t, cond)

            # Add guidance
            if w is not None:
                alpha_bar = self.noise_scheduler.alphas_cumprod[t]
                noise_pred += -(1 - alpha_bar).sqrt() * w

            # Reverse diffusion step
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample

        # Restore original model weights
        self.ema.restore(self.model_params)
        return sample

    def state_dict(self):
        return {
            "noise_pred_net": self.noise_pred_net.state_dict(),
            "cond_encoders": self.cond_encoders.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "ema": self.ema.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.noise_pred_net.load_state_dict(state_dict["noise_pred_net"])
        self.cond_encoders.load_state_dict(state_dict["cond_encoders"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.ema.load_state_dict(state_dict["ema"])

    def weighted_mse_loss(self, input, target, weight):
        '''
        input: (batch, dim)
        target: (batch, dim)
        weight: (batch) or (batch, 1)
        '''
        assert input.dim() == 2 and target.dim() == 2
        dim = input.shape[1]
        weight = weight.reshape(weight.shape[0], 1)
        return torch.sum(weight * (input - target) ** 2) / (torch.sum(weight) * dim)


class SimpleDiffusionPolicy(ConditionalDiffusionModel):
    '''
    Note: When loading DiffusionPolicy, also need to load scaler manually
    '''
    def __init__(
        self,
        obs_shape,
        act_shape,
        feature_dim,
        num_training_steps,
        num_diffusion_steps,
        device,
        **kwargs,
    ):
        super().__init__(
            input_dim=np.prod(act_shape),
            cond_shape_dict={"obs": obs_shape, "feat": (feature_dim,)},
            num_training_steps=num_training_steps,
            num_diffusion_steps=num_diffusion_steps,
            clip_sample=True,
            device=device,
            **kwargs,
        )

    def learn(self, batch: Dict):
        '''
        Update one batch
        '''
        obss = batch['observations'].type(torch.float32).to(self.device)
        actions = batch['actions'].type(torch.float32).to(self.device)
        rtgs = batch['rtgs']
        rtgs = rtgs.reshape(rtgs.shape[0], -1).type(torch.float32).to(self.device)
        if 'weights' in batch:
            weights = batch['weights'].type(torch.float32).to(self.device) # (batch, )
        else:
            weights = None

        return super().learn(actions, {"obs": obss, "feat": rtgs}, weights)

    def validate(self, batch: Dict):
        '''
        Update one batch
        '''
        obss = batch['observations'].type(torch.float32).to(self.device)
        actions = batch['actions'].type(torch.float32).to(self.device)
        rtgs = batch['rtgs']
        rtgs = rtgs.reshape(rtgs.shape[0], -1).type(torch.float32).to(self.device)
        if 'weights' in batch:
            weights = batch['weights'].type(torch.float32).to(self.device) # (batch, )
        else:
            weights = None

        return super().validate(actions, {"obs": obss, "feat": rtgs}, weights)

    def select_action(self, obs, feat):
        # print(f"DiffusionPolicy: select action with obs shape {obs.shape}, feat(rtg) shape {feat.shape}")
        obs = torch.as_tensor(obs, dtype = torch.float32, device = self.device)
        feat = torch.as_tensor(feat, dtype = torch.float32, device = self.device)

        with torch.no_grad():
            action = super().sample({"obs": obs, "feat": feat})
        # print(action)
        return action.cpu().numpy()

    def train(self) -> None:
        self.noise_pred_net.train()
        self.cond_encoders.train()

    def eval(self) -> None:
        self.noise_pred_net.eval()
        self.cond_encoders.eval()
