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
    "batch_size": 64,
    "lr": 0.0001,
    "epochs": 128,

    # "base": "efficient-dragon-18",
    "base": None,

    "nhead": 4,
    "nlayers": 3,

    "latent_size": 256,

    # stop after number of epochs without improvement
    "pretrain__stopping_epochs": 3,
    # maximum epochs
    "pretrain__max_epochs": 50,

    # stop after number of epochs without improvement
    "stopping_epochs": 5,
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

PRETRAIN_EARLY = config.pretrain__stopping_epochs
PRETRAIN_MAX = config.pretrain__max_epochs
TRAIN_EARLY = config.stopping_epochs

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

# initialize the training set and loader
dataset = NACCDataset("./data/investigator_nacc57.csv", f"./features/combined", fold=FOLD)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_set = TensorDataset(*dataset.val())
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)


# create the model
if not MODEL:
    model = NACCEmbedder(dataset._num_features, latent=config.latent_size, nhead=config.nhead, nlayers=config.nlayers).to(DEVICE)
else:
    model = NACCEmbedder(dataset._num_features, latent=config.latent_size, nhead=config.nhead, nlayers=config.nlayers).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(f"./models/{MODEL}", "model.save"), map_location=DEVICE))

run.watch(model)

# and optimizer
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# if the model is pretraining
is_pretraining = True

# get the last running accuracy
last_running_acc = 0
last_running_au_sum = 0
no_improvement_counter = 0

model.train()
for epoch in range(EPOCHS):
    if is_pretraining:
        if no_improvement_counter >= PRETRAIN_EARLY:
            print(f"STOPPING pretraining! No improvement for {PRETRAIN_EARLY} steps! Moving on to representation learning.") 
            is_pretraining = False
        elif epoch >= PRETRAIN_MAX:
            print(f"STOPPING pretraining! Moving on after {epoch} epochs") 
            is_pretraining = False
    elif not is_pretraining and no_improvement_counter > TRAIN_EARLY:
        print(f"EARLY STOPPING training! No improvement for {TRAIN_EARLY} steps!") 
        break
        
    print(f"Currently training epoch {epoch}...")

    running_acc = []
    # alignment + uniformity
    running_a_u_sum = []

    for i, batch in tqdm(enumerate(iter(dataloader)), total=len(dataloader)):
        if i % 64 == 0:
            alignment = sample_alignment(model, validation_set)
            uniformity = sample_uniformity(model, validation_set)

            # create a single validation batch
            val_batch = next(iter(validation_loader))
            val_batch = [i.to(DEVICE) for i in val_batch]

            # perform inference
            with torch.inference_mode():
                model.eval()
                logits = model(val_batch[0], val_batch[1])["logits"]
                model.train()

            # and calculate accuracy via the labels
            label_indicies = torch.argmax(val_batch[-1], 1)
            accuracy = sum(label_indicies == torch.argmax(logits, axis=1))/len(label_indicies)
            accuracy = accuracy.detach().cpu().item()

            # detach other variables
            alignment = alignment.detach().cpu().item()
            uniformity = uniformity.detach().cpu().item()

            running_acc.append(accuracy)
            running_a_u_sum.append(alignment+uniformity)

            run.log({"alignment": alignment,
                     "uniformity": uniformity,
                     "accuracy": accuracy}, commit=False)
            

        batchp = batch
        # send batch to GPU if needed
        batch = [i.to(DEVICE) for i in batch]

        # we skip any batch of 1 element, because of BatchNorm
        if batch[0].shape[0] == 1:
            continue

        # run with actual backprop
        try:
            if is_pretraining:
                output = model(batch[0].float(), batch[1], pretrain_target=batch[-1])
            else:
                output = model(batch[0].float(), batch[1],
                               batch[2].float(), batch[3],
                               batch[4].float(), batch[5])
        except RuntimeError as e:
            print(e)
            optimizer.zero_grad()
            continue

        # backprop
        try:
            output["loss"].backward()
        except RuntimeError:
            breakpoint()
        optimizer.step()
        optimizer.zero_grad()

        if i % 64 == 0:
            run.log({"repr": output["latent"][0].detach().cpu()}, commit=False)

        # logging
        if is_pretraining:
            run.log({"pretrain_loss": output["loss"].detach().cpu().item()})
        else:
            run.log({"loss": output["loss"].detach().cpu().item()})

    print("Acc:", sum(running_acc)/len(running_acc), last_running_acc)
    print("A/U Sum:", sum(running_a_u_sum)/len(running_a_u_sum), last_running_au_sum)

    if ((is_pretraining and
         sum(running_acc)/len(running_acc) <= last_running_acc) or
        (is_pretraining == False and
         sum(running_a_u_sum)/len(running_a_u_sum) >= last_running_au_sum)):
        print("No Improvement!")
        no_improvement_counter += 1
    else:
        no_improvement_counter = 0
        print("Improvement! Saving model...")
        if not os.path.exists(f"./models/{run.name}"):
            os.mkdir(f"./models/{run.name}")
        torch.save(model.state_dict(), f"./models/{run.name}/model.save")
        torch.save(optimizer, f"./models/{run.name}/optimizer.save")

        last_running_acc = sum(running_acc)/len(running_acc)
        last_running_au_sum = sum(running_a_u_sum)/len(running_a_u_sum)

# Saving
if epoch == EPOCHS - 1:
    print("Saving model...")
    if not os.path.exists(f"./models/{run.name}"):
        os.mkdir(f"./models/{run.name}")
    torch.save(model.state_dict(), f"./models/{run.name}/model.save")
    torch.save(optimizer, f"./models/{run.name}/optimizer.save")
