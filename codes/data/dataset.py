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
from codes.utils.utils import str_csv_to_dict, windowing, load_melspectrograms_dB_settings

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

        melspectrogram = librosa.feature.melspectrogram(y=waveform_pad, sr=sample_rate, n_mels=128, fmax=4096)
        melspectrogram_dB = librosa.power_to_db(melspectrogram, ref=np.max)
        
        return(torch.tensor(melspectrogram_dB, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.float32))
    

class PredictionDataset(Dataset):
    def __init__(self, file, file_settings="melspectrograms_dB_settings/1.npy"):
        self.melspectrograms_dB = load_melspectrograms_dB_settings(file_settings)
        self.melspectrograms_dB = np.append(self.melspectrograms_dB, windowing(file), axis=0)
        
    def __len__(self):
        return len(self.melspectrograms_dB)
    
    def __getitem__(self, idx):
        return torch.tensor(self.melspectrograms_dB[idx], dtype=torch.float32).unsqueeze(0)

    
class BuildDataLoader:
    def __init__(self, paths: List[str], dataloader_params: dict):
        self.csv_reader = CsvReader(paths)
        self.dataloader = DataLoader(BaseDataset(self.csv_reader.df), **dataloader_params)


class PredictionDataloader:
    def __init__(self, file) -> None:
        self.dataloader = DataLoader(PredictionDataset(file), batch_size=4, num_workers=2)