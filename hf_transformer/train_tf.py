import os
import torch
from transformers import Trainer, TrainingArguments
from hf_transformer.data_collator import DecisionTransformerGymDataCollator
from hf_transformer.tdmpc_transformer import TDMPCDecisionTransformerConfig, TDMPCDecisionTransformerModel
from hf_transformer.config import Config
from hf_transformer.hf_hub_files import get_tdmpc2_mt30

os.environ["WANDB_DISABLED"] = "true" # we diable weights and biases logging for this tutorial

print("Collecting data...")
data_path = get_tdmpc2_mt30() # Download the TDMPC 30 task dataset and return path

# TODO: Make this use all chunks
torch.cuda.empty_cache()
dataset = torch.load(os.path.join(os.getcwd(),data_path,'chunk_3.pt'))
# for i in range(3):
#     d = torch.load(os.path.join(os.getcwd(),data_path,f'chunk_{i}.pt'))
#     for k in dataset.keys():
#         dataset[k] = torch.cat((dataset[k],d[k]),0)

# Custom config object
# stored in config.py
cfg = Config()
collator = DecisionTransformerGymDataCollator(dataset, cfg)

print("Initializing Transformer...")
tf_config = TDMPCDecisionTransformerConfig(num_tasks=cfg.numtasks,
                                            state_dim=cfg.latent_dim,
                                            act_dim=cfg.action_dim,
                                            use_horizon_batchsize_dimensioning=cfg.use_horizon_batchsize_dimensioning)
model = TDMPCDecisionTransformerModel(tf_config)

training_args = TrainingArguments(
    output_dir="pretrained_models/",
    remove_unused_columns=False,
    num_train_epochs=120,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

print("Beginning training...")
trainer.train()