import os
import torch
from transformers import Trainer, TrainingArguments
from hf_transformer.data_collator import DecisionTransformerGymDataCollator
from hf_transformer.tdmpc_transformer import TDMPCDecisionTransformerConfig, TDMPCDecisionTransformerModel
from hf_transformer.config import Config
from hf_transformer.tf_trainer import TF_Trainer
from hf_transformer.hf_hub_files import get_tdmpc2_mt30

os.environ["WANDB_DISABLED"] = "true" # we diable weights and biases logging for this tutorial

print("Collecting data...")
data_path = get_tdmpc2_mt30() # Download the TDMPC 30 task dataset and return path

# TODO: Make this use all chunks
torch.cuda.empty_cache()
dataset = torch.load(os.path.join(os.getcwd(),data_path,'chunk_0.pt'))

cfg = Config()
collator = DecisionTransformerGymDataCollator(dataset, cfg)

print("Initializing Transformer...")
tf_config = TDMPCDecisionTransformerConfig(num_tasks=cfg.numtasks,
                                            state_dim=cfg.latent_dim,
                                            act_dim=cfg.action_dim,
                                            use_horizon_batchsize_dimensioning=cfg.use_horizon_batchsize_dimensioning)
model = TDMPCDecisionTransformerModel(tf_config)

trainer = TF_Trainer(
    model=model,
    cfg=cfg,
    train_dataset=dataset,
    collator=collator
)

print("Beginning training...")
model, train_history = trainer.train()
model.save('pretrained_models/model.keras')