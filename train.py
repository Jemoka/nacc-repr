# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import pandas as pd

import gc

import wandb

import functools

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

# import tqdm auto
from tqdm.auto import tqdm
tqdm.pandas()

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import math
import random

# model
from model import NACCEmbedder
from utils import sample_alignment, sample_uniformity
from dataset import NACCDataset


CONFIG = {
    "fold": 0,
    "batch_size": 128,
    "lr": 0.0001,
    "epochs": 55,

    # "base": "efficient-dragon-18",
    "base": None,

    "nhead": 4,
    "nlayers": 3,

    "latent_size": 128,
}


ONLINE = False
# ONLINE = True

run = wandb.init(project="nacc-repr", entity="jemoka", config=CONFIG, mode=("online" if ONLINE else "disabled"))

config = run.config

BATCH_SIZE = config.batch_size
LR = config.lr
EPOCHS = config.epochs
MODEL = config.base
FOLD = config.fold

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

# initialize the training set and loader
dataset = NACCDataset("./data/investigator_nacc57.csv", f"./features/combined", fold=FOLD)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_set = TensorDataset(*dataset.val())

# create the model
if not MODEL:
    model = NACCEmbedder(dataset._num_features, latent=config.latent_size, nhead=config.nhead, nlayers=config.nlayers).to(DEVICE)
else:
    model = NACCEmbedder(dataset._num_features, latent=config.latent_size, nhead=config.nhead, nlayers=config.nlayers).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(f"./models/{MODEL}", "model.save"), map_location=DEVICE))

# and optimizer
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)


model.train()
for epoch in range(EPOCHS):
    print(f"Currently training epoch {epoch}...")

    for i, batch in tqdm(enumerate(iter(dataloader)), total=len(dataloader)):
        if i % 64 == 0 and i != 0:
            alignment = sample_alignment(model, validation_set)
            uniformity = sample_uniformity(model, validation_set)

            run.log({"alignment": alignment.detach().cpu().item(),
                     "uniformity": uniformity.detach().cpu().item()})
            

        batchp = batch
        # send batch to GPU if needed
        batch = [i.to(DEVICE) for i in batch]

        # we skip any batch of 1 element, because of BatchNorm
        if batch[0].shape[0] == 1:
            continue

        # run with actual backprop
        try:
            output = model(batch[0].float(), batch[1],
                           batch[2].float(), batch[3],
                           batch[4].float(), batch[5])
        except RuntimeError:
            optimizer.zero_grad()
            continue

        # backprop
        try:
            output["loss"].backward()
        except RuntimeError:
            breakpoint()
        optimizer.step()
        optimizer.zero_grad()

        # logging
        run.log({"loss": output["loss"].detach().cpu().item()})

        if i % 64 == 0:
            run.log({"repr": output["latent"][0].detach().cpu()})

# Saving
print("Saving model...")
os.mkdir(f"./models/{run.name}")
torch.save(model.state_dict(), f"./models/{run.name}/model.save")
torch.save(optimizer, f"./models/{run.name}/optimizer.save")
