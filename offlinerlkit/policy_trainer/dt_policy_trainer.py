import numpy as np
import torch
import time
from tqdm import tqdm
import wandb
import os

from torch.utils.data.dataloader import DataLoader
from offlinerlkit.policy import DecisionTransformer
import torch.nn.functional as F

class TrainerConfig:
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    ckpt_path = None
    num_workers = 1 # for DataLoader
    tb_log = None
    log_to_wandb = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class SequenceTrainer:
    def __init__(self, config: TrainerConfig, model: DecisionTransformer, offline_dataset, rollout_dataset = None, is_gym = False):
        '''
        offline_trajs / rollout_trajs: List[Trajectory]
        config members:
        - batch_size
        - lr
        - device
        '''
        self.config = config
        self.device = self.config.device
        self.model = model.to(self.device)
        self.batch_size = config.batch_size
        self.diagnostics = dict()
        self.offline_dataset = offline_dataset
        self.rollout_dataset = rollout_dataset

        warmup_steps = 10000
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )

        self.logger = config.logger
        self.is_gym = is_gym

    def loss_fn(self, pred_action, true_action):
        '''
        Compute the MSE loss.
        - pred_action: (batch, action_dim), logits of the predicted action (don't do softmax)
        - true_action: (batch, action_dim), the true action in 1-dim representation
        Return: scalar tensor. The mean of each loss
        '''

        return F.mse_loss(pred_action, true_action)

    def eval(self, desired_rtg, train_epoch):
        '''
        state_mean/std: Used for state normalization. Get from offline_dataset only currently
        '''
        state_mean, state_std = self.offline_dataset.get_normalize_coef()
        self.model.train(False)
        rets = [] # list of returns achieved in each epoch
        env = self.config.env
        action_dim = env.action_space.shape[0]
        for epoch in range(self.config.eval_repeat):
            if self.is_gym:
                states = env.reset()
            else:
                states, _ = env.reset()
            if hasattr(env, 'get_true_observation'): # For pointmaze
                states = env.get_true_observation(states)
            states = torch.from_numpy(states)
            states = states.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) # (1,1,state_dim)
            rtgs = torch.Tensor([[[desired_rtg]]]).to(self.device) # (1,1,1)
            timesteps = torch.Tensor([[0]]).to(self.device) # (1,1)
            
            # Initialize action
            actions = torch.empty((1,0,action_dim)).to(self.device) # Actions are represented in one-hot

            ret = 0 # total return 
            for h in range(self.config.horizon):
                # Get action
                pred_action = self.model.get_action((states - state_mean) / state_std,
                                                      actions.type(torch.float32),
                                                      rtgs.type(torch.float32),
                                                      timesteps.type(torch.float32)) # (act_dim)

                # Observe next states, rewards,
                if self.is_gym:
                    next_state, reward, terminated, _ = env.step(pred_action.detach().cpu().numpy()) # (state_dim), scalar
                else:
                    next_state, reward, terminated, _, _ = env.step(pred_action.detach().cpu().numpy()) # (state_dim), scalar
                if hasattr(env, 'get_true_observation'): # For pointmaze
                    next_state = env.get_true_observation(next_state)
                if epoch == 0 and self.config.debug:
                    print(f"Step {h+1}, action is {pred_action.detach().cpu()}, observed next state {next_state}, reward {reward}")   
                next_state = torch.from_numpy(next_state)
                # Calculate return
                ret += reward
                
                # Update states, actions, rtgs, timesteps
                next_state = next_state.unsqueeze(0).unsqueeze(0).to(self.device) # (1,1,state_dim)
                states = torch.cat([states, next_state], dim=1)
                states = states[:, -self.config.ctx: , :] # truncate to ctx_length

                pred_action = pred_action.unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, action_dim)
                
                if self.config.ctx > 1:
                    actions = torch.cat([actions, pred_action], dim=1)
                    actions = actions[:, -self.config.ctx+1: , :] # actions length is ctx-1

                next_rtg = rtgs[0,0,-1] - reward
                next_rtg = next_rtg * torch.ones(1,1,1).to(self.device) # (1,1,1)
                rtgs = torch.cat([rtgs, next_rtg], dim=1)
                rtgs = rtgs[:, -self.config.ctx: , :]

                # Update timesteps
                timesteps = torch.cat([timesteps, (h+1)*torch.ones(1,1).to(self.device)], dim = 1) 
                timesteps = timesteps[:, -self.config.ctx: ]

            # Add the ret to list
            rets.append(ret)

        ep_reward_mean, ep_reward_std = np.mean(rets), np.std(rets)

        # logging
        self.logger.logkv("epoch", train_epoch + 1)
        self.logger.logkv("eval/target_return", desired_rtg)
        self.logger.logkv("eval/episode_return", ep_reward_mean)
        self.logger.logkv("eval/episode_return_std", ep_reward_std)  

        # Set the model back to training mode
        self.model.train(True)
        return ep_reward_mean
    
    def _run_epoch(self, epoch_num):
        '''
        Run one epoch in the training process \n
        Epoch_num: int, epoch number, used to display in progress bar. \n
        During training, we convert action to one_hot_hash
        '''
        if self.rollout_dataset is None: # Only offline dataset, don't differ
            dataset = self.offline_dataset
        else:
            if epoch_num < self.config.pre_epochs:
                dataset = self.offline_dataset
                if self.config.debug:
                    print(f"Pretraining") 
            else:
                dataset = self.rollout_dataset
                if self.config.debug:
                    print(f"Training on rollout data")
        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size= self.config.batch_size,
                            num_workers= self.config.num_workers)
        
        # losses = []
        pbar = tqdm(enumerate(loader), total=len(loader))
        losses = []
        for it, (states, actions, _, rtgs, timesteps, attention_mask) in pbar:
            '''
            states, (batch, ctx, state_dim)
            actions, (batch, ctx, action_dim)
            rtgs, (batch, ctx, 1)
            timesteps, (batch, ctx)
            attention_mask, (batch, ctx)
            '''    

            states = states.type(torch.float32).to(self.device)
            actions = actions.type(torch.float32).to(self.device)
            rtgs = rtgs.type(torch.float32).to(self.device)
            timesteps = timesteps.to(self.device).long()
            attention_mask = attention_mask.to(self.device)

            action_target = torch.clone(actions)

            # forward the model
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rtgs, timesteps, attention_mask=attention_mask,
            )

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            loss = self.loss_fn(
                action_preds,
                action_target
            )

            losses.append(loss.item())
            
            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()

            self.logger.logkv_mean("loss", loss.item())
            pbar.set_description(f"Epoch {epoch_num+1}, iter {it}: train loss {loss.item():.5f}.")
        

    def train(self):
        start_time = time.time()
        for epoch in range(self.config.max_epochs):
            self._run_epoch(epoch)
            if self.config.last_eval and epoch < self.config.max_epochs - 1:
                pass
            else:
                self.eval(self.config.desired_rtg, train_epoch=epoch)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        self._save_checkpoint(os.path.join(self.logger.model_dir, "policy_final.pth"))

    def _save_checkpoint(self, ckpt_path):
        '''
        ckpt_path: str, path of storing the model
        '''
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model, ckpt_path)
