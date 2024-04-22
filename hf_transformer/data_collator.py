import random
import os
from dataclasses import dataclass

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
    
    def __init__(self, dataset) -> None:
        self.act_dim = dataset["action"].shape[-1]
        self.state_dim = dataset["obs"].shape[-1]
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        self.n_traj = dataset['obs'].shape[0]
        self.state_mean = dataset['obs'].mean(dim=0).mean(dim=0).numpy()
        self.state_std = dataset['obs'].std(dim=0).std(dim=0).numpy()+1e-9
        self.p_sample = np.ones((self.n_traj)) / self.n_traj

        self.num_tasks = torch.max(dataset['task']).item()
        self.state_dim += self.num_tasks

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, rtg, timesteps, mask = [], [], [], [], [], []
        
        for ind in batch_inds:
            # for feature in features:
            feature = self.dataset[int(ind)]
            si = random.randint(0, len(feature["reward"]) - 1)

            # get sequences from dataset
            s_obs = np.array(feature["obs"][si : si + self.max_len]).reshape(1, -1, self.state_dim-self.num_tasks)
            task_arr = np.array(feature["task"][si : si + self.max_len]).reshape(1, -1)
            # add task encoding to state
            t_enc = np.zeros((1,task_arr.shape[1], self.num_tasks))
            t_enc[0, np.arange(task_arr.shape[1]), task_arr.squeeze()-1] = 1
            s.append(np.concatenate([t_enc, s_obs], axis=2))
            
            a.append(np.array(feature["action"][si : si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature["reward"][si : si + self.max_len]).reshape(1, -1, 1))

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["reward"][si:]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1][:,:,self.num_tasks:] = (s[-1][:,:,self.num_tasks:] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }