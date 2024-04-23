import torch
import torch.nn as nn
from tdmpc2.common import layers, math, init

tasks = ['humanoid_h1hand-sit_simple-v0'] # Make this be as long as num tasks + 1
cfg = {
    "multitask":True,
    "obs": 'state',
    "obs_shape":{'state': [151]},
    "tasks":tasks,
    "task_dim": 64, # 96 if mt80 else 64. 0 if single task
}

# woodwardbr
# Encoder mimics the TDMPC2 Encoder 
# To be used for pretraining a transformer
class Encoder():
    def __init__(self, cfg):
        self.cfg = cfg
        self._encoder = layers.enc(cfg)
        self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)

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