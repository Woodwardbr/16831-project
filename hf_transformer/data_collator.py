import random
from dataclasses import dataclass
from hf_transformer.encoder import Encoder

import numpy as np
import torch

@dataclass
class DecisionTransformerGymDataCollator:
    return_tensors: str = "pt"
    max_len: int = 20 #subsets of the episode we use for training
    state_dim: int = 17  # size of state space
    act_dim: int = 6  # size of action space
    max_ep_len: int = 1000 # max episode length in the dataset
    scale: float = 1000.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset
    
    def __init__(self, dataset, config) -> None:
        self.act_dim = config.action_dim
        self.state_dim = config.obs_shape['state'][0]
        self.latent_dim = config.latent_dim
        self.task_dim = config.task_dim
        self.max_len = config.horizon
        self.dataset = dataset

        # calculate dataset stats for normalization of states
        self.n_traj = dataset['obs'].shape[0]
        # TODO(woodwardbr): probably can delete this bc of encoding
        # self.state_mean = dataset['obs'].mean(dim=0).mean(dim=0).numpy()
        # self.state_std = dataset['obs'].std(dim=0).std(dim=0).numpy()+1e-9
        self.p_sample = np.ones((self.n_traj)) / self.n_traj

        self.encoder = Encoder(config)
        self.only_state_p = config.only_state_p
        self.use_horizon_batchsize_dimensioning = config.use_horizon_batchsize_dimensioning

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, feature):
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_size = len(feature)
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        z, a, r, rtg, timesteps, mask = [], [], [], [], [], []
        
        for ind in batch_inds:
            # for feature in features:
            feature = self.dataset[int(ind)]

            si = random.randint(0, len(feature["reward"]) - 1)

            # Data cleaning
            # For some reason there are NaNs
            feature["obs"][torch.isnan(feature["obs"])] = 1e-12
            feature["action"][torch.isnan(feature["action"])] = 1e-12
            feature["reward"][torch.isnan(feature["reward"])] = 1e-12
            feature["task"][torch.isnan(feature["task"])] = 1e-12

            # For some reason some of the actions are enormous
            feature['action'][feature['action']> 10] =  10
            feature['action'][feature['action']<-10] = -10
            

            # get sequences from dataset
            s_b = feature["obs"][si : si + self.max_len].clone().detach().unsqueeze(0)
            a_b = np.array(feature["action"][si : si + self.max_len])
            r_b = np.array(feature["reward"][si : si + self.max_len])
            t_b = feature["task"][si : si + self.max_len].clone().detach()

            # reshape to maximum dimension by adding 0s
            s_b = torch.cat([s_b, torch.zeros((s_b.shape[0], s_b.shape[1], self.state_dim-s_b.shape[2]))], axis=2)
            a_b = np.concatenate([a_b, np.zeros((a_b.shape[0], self.act_dim-a_b.shape[1]))], axis=1)
            
            # Encode into latent space
            # Append to sequences
            z.append(self.encoder.encode(s_b, t_b).detach().numpy())
            a.append(a_b.reshape(1, -1, self.act_dim))
            r.append(r_b.reshape(1, -1, 1))

            timesteps.append(np.arange(si, si + z[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["reward"][si:]), gamma=0.99)[
                    : z[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < z[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = z[-1].shape[1]
            z[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.latent_dim)), z[-1]], axis=1)
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.expand_dims(np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1),2)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

            # Use some examples where only state is populated
            # So that the transformer is prepared for TDMPC2
            if np.random.rand() < self.only_state_p:
                a[-1] = np.zeros_like(a[-1])
                r[-1] = np.zeros_like(r[-1])
                rtg[-1] = np.zeros_like(rtg[-1])



        z = torch.from_numpy(np.concatenate(z, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        if self.use_horizon_batchsize_dimensioning:
            z = z.permute(1,0,2)
            a = a.permute(1,0,2)
            r = r.permute(1,0,2)
            rtg = rtg.permute(1,0,2)
            timesteps = timesteps.permute(1,0,2)
            mask = mask.permute(1,0)

        return {
            "states": z,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }