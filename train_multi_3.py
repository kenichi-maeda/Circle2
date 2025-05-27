#### To run: torchrun --standalone --nnodes=1 --nproc_per_node=6 train_multi_3.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split, DistributedSampler
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from torch.distributed.elastic.multiprocessing.errors import record

dist.init_process_group(backend="gloo")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.set_num_threads(20 // int(os.environ["WORLD_SIZE"]))


# Dataset class
class CircleDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = torch.tensor(row[:10].values, dtype=torch.float32)
        outputs = row[10:].values.reshape(4, 3)
        targets = torch.tensor(outputs.flatten(), dtype=torch.float32)
        return inputs, targets


data = pd.read_csv("circles_data_100M.csv")

# Split dataset
full_dataset = CircleDataset(data)
train_size = int(0.9 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])


# **Optimized Model with Skip Connections**
class CircleNet(torch.nn.Module):
    def __init__(self):
        super(CircleNet, self).__init__()

        self.input_layer = torch.nn.Linear(10, 1024)
        self.batchnorm1 = torch.nn.BatchNorm1d(1024)
        self.activation = torch.nn.SiLU()  # Swish activation

        self.hidden1 = torch.nn.Linear(1024, 1024)
        self.batchnorm2 = torch.nn.BatchNorm1d(1024)

        self.hidden2 = torch.nn.Linear(1024, 512)
        self.batchnorm3 = torch.nn.BatchNorm1d(512)

        self.hidden3 = torch.nn.Linear(512, 256)
        self.batchnorm4 = torch.nn.BatchNorm1d(256)

        self.hidden4 = torch.nn.Linear(256, 128)
        self.batchnorm5 = torch.nn.BatchNorm1d(128)

        self.hidden5 = torch.nn.Linear(128, 64)
        self.batchnorm6 = torch.nn.BatchNorm1d(64)

        self.skip_proj = torch.nn.Linear(1024, 64)
        self.output_layer = torch.nn.Linear(64, 12)

        self.dropout = torch.nn.Dropout(0.01)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.batchnorm1(x)
        x = self.activation(x)

        res1 = self.skip_proj(x)  # Skip connection

        x = self.hidden1(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden3(x)
        x = self.batchnorm4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden4(x)
        x = self.batchnorm5(x)
        x = self.activation(x)

        x = self.hidden5(x)
        x = self.batchnorm6(x)
        x = self.activation(x)

        x = x + res1  # Skip connection

        x = self.output_layer(x)
        return x


# **Multi-Processing Training Function**
def train():
    # Define model
    model = CircleNet().to("cpu")
    model = DDP(model)

    # Use DistributedSampler for multi-process training
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=4, pin_memory=True, persistent_workers=True)

    # Define loss function (combining L1 + Huber)
    criterion1 = nn.L1Loss()
    criterion2 = nn.SmoothL1Loss()  # Huber Loss for robustness

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Learning rate scheduler: Cosine Annealing with Warmup
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])

    # Training loop
    for epoch in range(30):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, desc=f"Process {rank}, Epoch {epoch+1}/30", unit="batch") as t:
            for inputs, targets in t:
                optimizer.zero_grad()
                outputs = model(inputs)

                # Sort predicted outputs and targets by radius
                outputs = outputs.view(-1, 4, 3)
                sorted_indices_outputs = torch.argsort(outputs[:, :, 2], dim=1)
                outputs_sorted = torch.gather(outputs, dim=1, index=sorted_indices_outputs.unsqueeze(-1).expand(-1, -1, 3))
                targets_sorted = targets.view(-1, 4, 3)

                # Compute combined loss
                loss = criterion1(outputs_sorted, targets_sorted) #+ 0.1 * criterion2(outputs_sorted, targets_sorted)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                t.set_postfix(train_loss=train_loss / len(train_loader.dataset))

        scheduler.step()
        print(f"Process {rank}: Epoch {epoch+1} complete.")

    if rank == 0:
        torch.save(model.module.state_dict(), "circle_model_100M_multi_4.pth")
        print("Model saved successfully!")
    dist.destroy_process_group()


# Main Multi-Process Execution
@record
def main():
    train()


if __name__ == "__main__":
    main()