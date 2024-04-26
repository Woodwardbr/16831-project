import torch
import torch.functional as F
from torch.utils.data import DataLoader
from tdmpc2.common.scale import RunningScale
import numpy as np

class TF_Trainer():
    def __init__(self, model, cfg, dataset, collator):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Trainer is on device:', self.device)
        self.cfg = cfg
        self.model = model.to(self.device)
        self.collator = collator

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                             max_lr=cfg.lr, 
                                                             total_steps=cfg.num_epochs*len(self.data_loader.dataset), 
                                                             final_div_factor=1e3)
        
        self.epochs = cfg.num_epochs
        self.batches = cfg.num_batches
        self.batch_size = cfg.batch_size

        dataloader_params = {
            "batch_size": self.batch_size,
            "collate_fn": collator,
            "num_workers": 0, # self.args.dataloader_num_workers,
            "pin_memory": True, # self.args.dataloader_pin_memory,
            "persistent_workers": 0, # self.args.dataloader_persistent_workers,
        }
        self.dataloader = DataLoader(dataset, **dataloader_params)
        self.q_scaler = RunningScale(cfg)

    def train(self):
        train_history = []
        test_history = []
        loss_buffer = []
        for epoch in range(self.epochs):
            self.model.train()
            for i, seq in enumerate(self.dataloader):
                seq = seq[0].to(self.device)
                self.optimizer.zero_grad()
                
                pred_seq, pred_ac, pred_log_pi, pred_rew = self.model(seq[:-1])

                pi_loss = self.pi_loss(pred_ac, seq[1:])
                log_pi_loss = self.log_pi_loss()
                loss = self.loss_fun(pred)

                loss.backward()
                self.optimizer.step()
                loss_buffer.append(loss.item())
                
                if i % 20 == 0:
                    print("epoch %d, iter %d, loss %.3f" % (epoch, i, np.mean(loss_buffer)))
                    loss_buffer = []
            train_history.append(loss.item())
            self.scheduler.step()

            # self.model.eval()
            # test_losses = []
            # with torch.no_grad():
            #     for i, seq in enumerate(test_loader):
            #         seq = seq[0].to(self.device)

            #         pred = self.model(seq)
            #         loss = self.loss_fun(pred)

            #         test_losses.append(loss.item())
            # print("epoch %d, test loss %.3f" % (epoch, np.mean(test_losses)))
            # test_history.append(np.mean(test_losses))

        return self.model, train_history # , test_history
    
    def tmpc_loss(self):
        # Compute losses
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t + 1] = z

        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += (
                math.soft_ce(reward_preds[t], reward[t], self.cfg).mean()
                * self.cfg.rho**t
            )
            for q in range(self.cfg.num_q):
                value_loss += (
                    math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean()
                    * self.cfg.rho**t
                )
        consistency_loss *= 1 / self.cfg.horizon
        reward_loss *= 1 / self.cfg.horizon
        value_loss *= 1 / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
        )
        return total_loss

    def policy_loss(self, log_pis, qs):
        self.q_scaler.update(qs[0])
        qs = self.q_scaler(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()