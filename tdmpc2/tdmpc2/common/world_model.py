from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from tdmpc2.common import layers, math, init
from hf_transformer.tdmpc_transformer import TDMPCDecisionTransformerModel, TDMPCDecisionTransformerConfig


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
            for i in range(len(cfg.tasks)):
                self._action_masks[i, : cfg.action_dims[i]] = 1.0
        self._encoder = layers.enc(cfg)
        self._dynamics = layers.mlp(
            cfg.latent_dim + cfg.action_dim + cfg.task_dim,
            2 * [cfg.mlp_dim],
            cfg.latent_dim,
            act=layers.SimNorm(cfg),
        )
        self._reward = layers.mlp(
            cfg.latent_dim + cfg.action_dim + cfg.task_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
        )
        self._pi = layers.mlp(
            cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim
        )
        self.tf_config = TDMPCDecisionTransformerConfig(num_tasks=cfg.task_dim,
                                                        state_dim=cfg.latent_dim,
                                                        act_dim=cfg.action_dim,
                                                        use_horizon_batchsize_dimensioning=True,
                                                        num_bins=cfg.num_bins,
                                                        )
        self._transformer = TDMPCDecisionTransformerModel(self.tf_config)
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        if self.cfg.multitask:
            self._action_masks = self._action_masks.to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def track_q_grad(self, mode=True):
        """
        Enables/disables gradient tracking of Q-networks.
        Avoids unnecessary computation during policy optimization.
        This method also enables/disables gradients for task embeddings.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.cfg.multitask:
            for p in self._task_emb.parameters():
                p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_target.data.lerp_(p.data, self.cfg.tau)

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        #a shape 512,61 expected 1 512 61
        tf= self.get_z_only_tf_input(z, a.unsqueeze(0))
        _,_,_,_,rew = self._transformer(**tf)
        print("Reward shape :", rew.shape)
        z = torch.cat([z, a], dim=-1)
        #print("Reward shape :", self._reward(z).shape)
        #reward shape [512, 101]
        return self._reward(z)
    
    def get_z_only_tf_input(self, z, actions=None, rewards=None, timesteps=None, returns_to_go=None):
        if len(z.shape)<3:
            z = z.unsqueeze(0)

        horizon = z.shape[0]
        batch_size = z.shape[1]
        if actions is not None:
            assert actions.shape[0] == horizon and actions.shape[1] == batch_size and actions.shape[2]==self.cfg.action_dim, f"shape assertion failed: actions shape: {actions.shape} expected shape: ({horizon},{batch_size},{self.cfg.action_dim})"
        else:
            actions = torch.zeros((horizon,batch_size,self.cfg.action_dim)).to(z.device)
        
        if rewards is not None:
            assert rewards.shape[0] == horizon and rewards.shape[1] == batch_size and rewards.shape[2]==1, f"shape assertion failed: rewards shape: {rewards.shape} expected shape: ({horizon},{batch_size},1)"
        else:
            rewards = torch.zeros((horizon,batch_size,1)).to(z.device)

        if returns_to_go is not None:
            assert returns_to_go.shape[0] == horizon and returns_to_go.shape[1] == batch_size and returns_to_go.shape[2]==1, f"shape assertion failed: returns_to_go shape: {returns_to_go.shape} expected shape: ({horizon},{batch_size},1)"
        else:
            returns_to_go = torch.zeros((horizon,batch_size,1)).to(z.device)

        """
        tf_inputs = {
            'states': z,
            'actions': torch.zeros((horizon,batch_size,self.cfg.action_dim)).to(z.device),
            'rewards': torch.zeros((horizon,batch_size,1)).to(z.device),
            'returns_to_go': torch.zeros((horizon,batch_size,1)).to(z.device),
            'timesteps': torch.arange(0,horizon).unsqueeze(0).repeat(batch_size,1).T.unsqueeze(2).to(z.device),
            # 'attention_mask': torch.zeros((horizon,batch_size,1)).to(z.device)
        }
        """
        tf_inputs = {
            'states': z,
            'actions': actions,
            'rewards': rewards,
            'returns_to_go': returns_to_go,
            'timesteps': torch.arange(0,horizon).unsqueeze(0).repeat(batch_size,1).T.unsqueeze(2).to(z.device),
            # 'attention_mask': torch.zeros((horizon,batch_size,1)).to(z.device)
        }
        return tf_inputs


    def pi(self, z, task, actions=None):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        # mu, log_std = self._pi(z).chunk(2, dim=-1)
        if actions is not None:
            actions = actions.unsqueeze(0)
        tf_input = self.get_z_only_tf_input(z, actions)
        _, _, mu, log_std, _ = self._transformer(**tf_input)
        if len(z.shape) < 3:
            mu = mu.squeeze()
            log_std = log_std.squeeze()
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask:  # Mask out unused action dimensions
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type="min", target=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
                - `min`: return the minimum of two randomly subsampled Q-values.
                - `avg`: return the average of two randomly subsampled Q-values.
                - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {"min", "avg", "all"}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)

        if return_type == "all":
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == "min" else (Q1 + Q2) / 2
