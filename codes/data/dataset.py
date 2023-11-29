import os
import numpy as np 
import pandas as pd 
import torch
import torchaudio


from torch import nn 
from torch.utils.data import Dataset, DataLoader
from typing import List


class Dataset(Dataset):
    def __init__(self, data_param: dict):
        self.data_param = data_param
        self.data = pd.read_csv(
            self.data_param.get('path_to_csv'),
            header=None,
            names=['Path', 'Label'],
            delimiter=','
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.data['Path'][idx])
        label = self.data['Label'][idx]

        return((waveform, sample_rate), label)