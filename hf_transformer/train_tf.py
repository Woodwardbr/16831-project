import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, load_dataset
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments
from hf_transformer.data_collator import DecisionTransformerGymDataCollator
from hf_transformer.trainable_transformer import TrainableDT
from hf_transformer.hf_hub_files import get_tdmpc2_mt30

os.environ["WANDB_DISABLED"] = "true" # we diable weights and biases logging for this tutorial

print("Collecting data...")
data_path = get_tdmpc2_mt30() # Download the TDMPC 30 task dataset and return path

# TODO: Make this use all chunks
dataset = torch.load(os.path.join(os.getcwd(),data_path,'chunk_0.pt'))
collator = DecisionTransformerGymDataCollator(dataset)

print("Initializing Transformer...")
config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
model = TrainableDT(config)

training_args = TrainingArguments(
    output_dir="output/",
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