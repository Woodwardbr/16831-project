import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tdmpc2.common.scale import RunningScale
import numpy as np
import tdmpc2.common.math as tdmpc_math

class TF_Trainer():
    def __init__(self, model, cfg, train_dataset, collator):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Trainer is on device:', self.device)
        self.cfg = cfg
        self.model = model.to(self.device)
        self.collator = collator

        dataloader_params = {
            "batch_size": self.cfg.batch_size,
            "collate_fn": self.collator
        }
        self.dataloader = DataLoader(self.collator.dataset, **dataloader_params)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                             max_lr=cfg.lr, 
                                                             total_steps=cfg.num_epochs*len(self.dataloader.dataset), 
                                                             final_div_factor=1e3)
        
        self.epochs = cfg.num_epochs
        self.batch_size = cfg.batch_size
        self.q_scaler = RunningScale(cfg)

    def train(self):
        train_history = []
        test_history = []
        loss_buffer = []
        for epoch in range(self.epochs):
            self.model.train()
            for i, seq in enumerate(self.dataloader):
                for k in seq.keys():
                    seq[k] = seq[k].to(self.device)

                pred_z, pred_q, pred_ac, pred_log_pi, pred_rew = self.model(
                                                        seq['states'][:-1],
                                                        seq['actions'][:-1],
                                                        seq['rewards'][:-1],
                                                        seq['returns_to_go'][:-1],
                                                        seq['timesteps'][:-1],
                                                        seq['attention_mask'][:-1]
                                                    )
                # Cloning to do loss for the log_pi output independantly
                # since it is dependant on Qs, another output of the TF
                # if i%2==0:
                self.optimizer.zero_grad()
                loss = self.tdmpc_loss(
                                        pred_z, 
                                        seq['states'][1:], 
                                        pred_q,
                                        pred_ac,
                                        seq['actions'][1:],
                                        pred_log_pi,
                                        pred_rew, 
                                        seq['rewards'][1:], 
                                        seq['returns_to_go'][1:]
                                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.clip_grad_norm
                )
                self.optimizer.step()

                loss_buffer.append(loss.item())
                
                if i % 50 == 0:
                    print("epoch %d, iter %d, loss %.3f" % (epoch, i, np.mean(loss_buffer)))
                    loss_buffer = []
            train_history.append(loss_buffer[-1])
            self.scheduler.step()
            self.model.save(self.model.state_dict(),f"pretrained_models/model_{epoch}/model.keras")

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
    
    def tdmpc_loss(self, 
                   z_pred, 
                   z_real, 
                   q_pred, 
                   ac_pred, 
                   ac_real,
                   log_pi_pred,
                   rew_pred, 
                   rew_real, 
                   rtg_real):
        # Compute losses
        consistency_loss, reward_loss, value_loss, action_loss, action_log_pi_loss = 0, 0, 0, 0, 0
        q_clone = q_pred.detach().clone().to(self.device)
        for t in range(self.cfg.horizon):
            consistency_loss += F.mse_loss(z_pred[:,t], z_real[:,t]) \
                        * self.cfg.rho**t
            reward_loss += torch.mean(tdmpc_math.soft_ce(rew_pred[:,t], rew_real[:,t], self.cfg)) \
                        * self.cfg.rho**t
            value_loss += torch.mean(tdmpc_math.soft_ce(q_pred[:,t], rtg_real[:,t], self.cfg)) \
                        * self.cfg.rho**t
            action_loss += F.mse_loss(ac_pred[:,t], ac_real[:,t]) \
                        * self.cfg.rho**t
            action_log_pi_loss += torch.mean(self.cfg.entropy_coef * log_pi_pred[:,t] - q_clone[:,t]) \
                        * self.cfg.rho**t

        # Normalize by horizon length
        consistency_loss *= 1 / self.cfg.horizon
        reward_loss *= 1 / self.cfg.horizon
        value_loss *= 1 / self.cfg.horizon
        action_loss *= 1 / self.cfg.horizon
        action_log_pi_loss *= 1 / self.cfg.horizon

        # Combine with weighted average
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
            + self.cfg.action_coef * action_loss
            + self.cfg.log_pi_coef * action_log_pi_loss
        )
        return total_loss