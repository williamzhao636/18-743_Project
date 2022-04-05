import sys
sys.path[-1] = "/usr/local/lib/python3.9/site-packages"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

import csv

class AccelerometerDataset(Dataset):
    def __init__(self, values):
        super(Dataset, self).__init__()
        self.values = values
      
    def __len__(self):
        return 162000

    def __getitem__(self, index):
      return index, self.values[index]


def parse_group_csv(csv_file_name):
    group_map = {}
    with open(csv_file_name, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            group_map[row['index']] = (row['x'], row['y'], row['z'], row['label'])
    return group_map

data = parse_group_csv("data/1.csv")

data_train = AccelerometerDataset(data)
data_train_loader = DataLoader(data_train, batch_size=1, shuffle=False)
print(len(data_train))

