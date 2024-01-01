# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import gc

import wandb

import functools

from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import math
import random
import pickle

tqdm.pandas()

bound=(1,3)

# a = pd.read_csv("../investigator_nacc57.csv")
# len(a[a.NACCETPR == 88])
# len(a[(a.NACCETPR == 1) & (a.DEMENTED == 1)])
# len(a[(a.NACCETPR == 1) & (a.NACCTMCI == 1)])
# len(a[(a.NACCETPR == 1) & (a.NACCTMCI == 2)])

# loading data
class NACCDataset(Dataset):

    def __init__(self, data_path, feature_path,
              # skipping 2 impaired because of labeling inconsistency
                 target_indicies=[1,3,4], fold=0):
        """The NeuralPsycology Dataset

        Arguments:

        data_path (str): path to the NACC csv
        feature_path (str): path to a text file with the input features to scean
        [target_indicies] ([int]): how to translate the output key values
                                   to the indicies of an array
        [fold] (int): the n-th fold to select
        
        """

        # initialize superclass
        super(NACCDataset, self).__init__()

        #### OPS ####
        # load the data
        data = pd.read_csv(data_path)

        # get the fature variables
        with open(feature_path, 'r') as f:
            lines = f.readlines()
            features = list(sorted(set([i.strip() for i in lines])))

        #### CURRENT PREDICTION TARGETS ####
        # construct the artificial target 
        # everything is to be ignored by default
        # this new target has: 0 - Control; 1 - MCI; 2 - Dementia
        data.loc[:, "current_target"] = -1

        # NACCETPR == 88; DEMENTED == 0 means Control
        data.loc[(data.NACCETPR == 88)&
                (data.DEMENTED == 0), "current_target"] = 0
        # NACCETPR == 1; DEMENTED == 1; means AD Dementia
        data.loc[(data.NACCETPR == 1)&
                (data.DEMENTED == 1), "current_target"] = 2
        # NACCETPR == 1; DEMENTED == 0; NACCTMCI = 1 or 2 means amnestic MCI
        data.loc[((data.NACCETPR == 1)&
                (data.DEMENTED == 0)&
                ((data.NACCTMCI == 1) |
                (data.NACCTMCI == 2))), "current_target"] = 1

        # drop the columns that are irrelavent to us (i.e. not the labels above)
        data = data[data.current_target != -1]
        # data.current_target.value_counts()

        #### TARGET BALANCING ####
        # crop the data to ensure balanced classes
        # TODO better dataaug that could exist?
        # we crop by future target, because that ensures
        # that results in more balanced classes for current
        # target even if we are nox explicitly balancing it
        min_class = min(data.current_target.value_counts())

        data = pd.concat([data[data.current_target==0].sample(n=min_class, random_state=7),
                          data[data.current_target==1].sample(n=min_class, random_state=7),
                          data[data.current_target==2].sample(n=min_class, random_state=7)]).sample(frac=1, random_state=7)


        #### TRAIN_VAL SPLIT ####
        kf = KFold(n_splits=10, shuffle=True, random_state=7)

        # split participants for indexing
        participants = list(sorted(set(data.NACCID.tolist())))
        splits = kf.split(participants)
        train_ids, test_ids = list(splits)[fold]
        train_participants = [participants[i] for i in train_ids]
        test_participants = [participants[i] for i in test_ids]

        # calculate number of features
        self._num_features = len(features) + 1
        # we add 1 for dummy variable used for fine tuning later

        data = data[features+["current_target", "NACCID"]]
        data = data.dropna()

        # crop the data for validatino
        self.val_data = data[data.NACCID.isin(test_participants)][features]
        self.val_targets = data[data.NACCID.isin(test_participants)].current_target

        self.data = data[data.NACCID.isin(train_participants)][features]
        self.targets = data[data.NACCID.isin(train_participants)].current_target

        self.features = features

    def __process(self, data, target, index=None):
        # as a test, we report results without masking
        # if a data entry is <0 or >80, it is "not found"
        # so, we encode those values as 0 in the FEATURE
        # column, and encode another feature of "not-found"ness
        data_found = (data > 80) | (data < 0)
        data[data_found] = 0
        # then, the found-ness becomes a mask
        data_found_mask = data_found

        # if it is a sample with no tangible data
        # well give up and get another sample:
        if sum(~data_found_mask) == 0:
            if not index:
                raise ValueError("All-Zero found in validation!")
            indx = random.randint(2,5)
            if index-indx <= 0:
                return self[index+indx]
            else:
                return self[index-indx]
        
        # seed the one-hot vector
        one_hot_target = [0 for _ in range(3)]
        # and set it
        one_hot_target[target] = 1

        return torch.tensor(data).float()/30, torch.tensor(data_found_mask).bool(), torch.tensor(one_hot_target).float()

    def __getitem__(self, index):
        # index the data
        data = self.data.iloc[index].copy()
        target = self.targets.iloc[index].copy()

        return self.__process(data, target, index)

    @functools.cache
    def val(self):
        """Return the validation set"""

        # collect dataset
        dataset = []

        print("Processing validation data...")

        # get it
        for index in tqdm(range(len(self.val_data))):
            try:
                dataset.append(self.__process(self.val_data.iloc[index].copy(),
                                            self.val_targets.iloc[index].copy()))
            except ValueError:
                continue # all zero ignore

        # return parts
        inp, mask, out = zip(*dataset)

        # process already divides by 30; don't do it twice
        return torch.stack(inp).float(), torch.stack(mask).bool(), torch.stack(out).float()

    def __len__(self):
        return len(self.data)

