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

from lab1_sol import *

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

import csv

# median across all acceleration is 2023
class PosNeg(object):
    def __init__(self, median):
        self.median = median

    def __call__(self, tensor):
        # maxt                                    = torch.max(tensor)
        tensor[tensor >= self.median]           = float('Inf')
        tensor[tensor <  self.median]           = 0
        tensor_pos                              = tensor.clone()
        tensor_pos[tensor_pos == 0]             = 1
        tensor_pos[tensor_pos == float('Inf')]  = 0
        tensor_pos[tensor_pos == 1]             = float('Inf')
        out                    = torch.stack([tensor_pos,tensor])
        return out



class AccelerometerDataset(Dataset):
    def __init__(self, values):
        super(Dataset, self).__init__()
        self.values = values
      
    def __len__(self):
        return 162502

    def __getitem__(self, index):
        inter = PosNeg(2023)
        return inter(torch.tensor(self.values[index][:-1])), self.values[index][-1]


def parse_group_csv(csv_file_name):
    group_map = {}
    with open(csv_file_name, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            try:
                group_map[int(row['index'])] = [float(row['x']), float(row['y']), float(row['z']), int(row['label'])]
            except:
                ind = row['index'].split('e')
                ind = float(ind[0]) * (10 ** 5)
                ind = int(ind)
                group_map[ind] = [float(row['x']), float(row['y']), float(row['z']), int(row['label'])]
    return group_map

dataset_train = parse_group_csv("data/1.csv")
dataset_test = parse_group_csv("data/2.csv")

data_train = AccelerometerDataset(dataset_train)
data_test = AccelerometerDataset(dataset_test)

data_train_loader = DataLoader(data_train, batch_size=1, shuffle=False)
data_test_loader = DataLoader(data_test, batch_size=1, shuffle=False)
# print(len(data_train))

train_loader = data_train_loader
test_loader = data_test_loader



# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================

# Taken from Lab 1, change
weights_save = 1
ucapture  = 1/2
usearch   = 1/1024
ubackoff  = 1/2


### Column Layer Parameters ###
inputsize = 28
rfsize    = 3
stride    = 1
nprev     = 2
neurons   = 12
theta     = 4

### Voter Layer Parameters ###
rows_v    = 26
cols_v    = 26
nprev_v   = 12
classes_v = 10
thetav_lo = 1/32
thetav_hi = 15/32
tau_eff   = 2

### Enabling CUDA support for GPU ###
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Layer Initialization ###

# Test different theta for snl and rnl versions
clayer = TNNColumnLayer(1, 3, 1, 2, 12, 30, ntype="rnl", device=device)

### Training ###
print("Starting column training")
for epochs in range(1):
    start = time.time()

    for idx, (data,target) in enumerate(train_loader):
        # if idx == 162502:
        if idx >= 100000:
            break
        print("Sample: {0}\r".format(idx), end="")

        out1, layer_in1, layer_out1 = clayer(data[0].permute(1,0))
        clayer.weights = clayer.stdp(layer_in1, layer_out1, clayer.weights, ucapture, usearch, ubackoff)

        endt                   = time.time()
        print("                                 Time elapsed: {0}\r".format(endt-start), end="")

    end   = time.time()
    print("Column training done in ", end-start)
    # print("weights: ", clayer.weights)
    '''
    Here are the weights
    
    weights:  Parameter containing:
    tensor([[3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000],
        [3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000,
         3.5508, 7.0000, 7.0000, 3.9092, 7.0000, 7.0000, 3.5508, 7.0000, 7.0000]])
    '''


### Display and save weights as images ###
'''
if weights_save == 1:

    image_list = []
    for i in range(12):
        temp = clayer.weights[i].reshape(56,28)
        image_list.append(temp)

    out = torch.stack(image_list, dim=0).unsqueeze(1)
    save_image(out, 'column_visweights_snl.png', nrow=6)
'''

### Testing and computing metrics ###

table    = torch.zeros((12,10))
pred     = torch.zeros(10)
totals   = torch.zeros(10)

print("Starting testing")
start    = time.time()

for idx, (data,target) in enumerate(test_loader):
    if idx >= 100000:
    # if idx >= 1000:
        break
    print("Sample: {0}\r".format(idx), end="")

    if cuda:
        data                    = data.cuda()
        target                  = target.cuda()

    # out1, layer_in1, layer_out1 = clayer(data[0].permute(1,2,0))
    out1, layer_in1, layer_out1 = clayer(data[0].permute(1,0))
    # print(out1, layer_in1, layer_out1)
    out = torch.flatten(out1)

    arg = torch.nonzero(out != float('Inf'))

    # print(arg)
    # print(target)
    # print(table)

    if arg.shape[0] != 0:
        table[arg[0].long(), target[0]] += 1

    endt = time.time()
    print("                                 Time elapsed: {0}\r".format(endt-start), end="")

end = time.time()
print("Testing done in ", end-start)


'''
Each row is one neuron
Each column in each row represents the weight that a particular input sample corresponds to a label
Number in row 0, column 1 represents the weight that neuron 0 assigns to label 1
'''
print("Confusion Matrix:")
print(table)

maxval   = torch.max(table, 1)[0]
totals   = torch.sum(table, 1)
pred     = torch.sum(maxval)
covg_cnt = torch.sum(totals)

# Purity of 0.4941 with SNL and 0.6726 with RNL

print("Purity: ", pred/covg_cnt)
print("Coverage: ", covg_cnt/(idx+1))


