import os
import numpy as np 
import pandas as pd 
import torch
import librosa
import codes.data.augmentation as augmentation


from torch import nn 
from torch.utils.data import Dataset, DataLoader
from typing import List
from codes.data.csv_reader import CsvReader
from codes.utils.utils import str_csv_to_dict

import matplotlib.pyplot as plt



class BaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, sample_rate = librosa.load(self.data['Path'][idx], sr=48_000)
        label = self.data['Label'][idx]
        
        name_aug = None if pd.isna(self.data['Name_Augmentation'][idx]) else self.data['Name_Augmentation'][idx]
        if name_aug:
            param = str_csv_to_dict(self.data['Param_Augmentation'][idx])
            if name_aug == "change_volume":
                waveform = augmentation.change_volume(waveform, **param)
            elif name_aug == "change_tonalities":
                waveform = augmentation.change_tonalities(waveform, sample_rate, **param)
            elif name_aug == "add_noise":
                waveform = augmentation.add_noise(
                    waveform,
                    **param
                )
        
        waveform_pad = np.zeros((48_000,))
        waveform_pad[:len(waveform)] = waveform

        spectrogram = librosa.stft(waveform_pad)
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))

        # plt.figure(figsize=(12, 8))
        # librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='log')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Spectrogram')
        # plt.show()


        return(torch.tensor(spectrogram_db, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.float32))
    

class BuildDataLoader:
    def __init__(self, paths: List[str], dataloader_params: dict):
        self.csv_reader = CsvReader(paths)
        self.dataloader = DataLoader(BaseDataset(self.csv_reader.df), **dataloader_params)