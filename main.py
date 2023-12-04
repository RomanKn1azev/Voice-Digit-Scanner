import argparse
import glob
import os
import warnings
import yaml
import random

import numpy as np
import torch
import torch.nn as nn

from codes.setup import Setup

from codes.data.dataset import PredictionDataloader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', type=str, required=True, help='Path to options file.')

    args = parser.parse_args()
    
    cfg_path = args.cfg

    with open(cfg_path, 'r') as file_option:
        cfg = yaml.safe_load(file_option)


    Setup(cfg).run_tasks()


if __name__ == "__main__":
    main()