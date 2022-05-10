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

encoding = 1

# median across all acceleration is 2023
class PosNeg(object):
    def __init__(self, median):
        self.median = median

    def __call__(self, tensor):
        # maxt                                    = torch.max(tensor)
        # print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        # print(tensor)
        tensor[tensor >= self.median]           = float('Inf')
        tensor[tensor <  self.median]           = 0
        tensor_pos                              = tensor.clone()
        tensor_pos[tensor_pos == 0]             = 1
        tensor_pos[tensor_pos == float('Inf')]  = 0
        tensor_pos[tensor_pos == 1]             = float('Inf')
        out                    = torch.stack([tensor_pos,tensor])
        if encoding:
            out = poisson(tensor, 10)
        # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        return out

def poisson(
    datum: torch.Tensor,
    time: int,
    dt: float = 1.0,
    device="cpu",
    approx=False,
    **kwargs,
) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be
    non-negative, and give the firing rate in Hz. Inter-spike intervals (ISIs) for
    non-negative data incremented by one to avoid zero intervals while maintaining ISI
    distributions.
    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :param device: target destination of poisson spikes.
    :param approx: Bool: use alternate faster, less accurate computation.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    time = int(time / dt)

    if approx:
        # random normal power awful approximation
        x = torch.randn((time, size), device=device).abs()
        x = torch.pow(x, (datum * 0.11 + 5) / 50)
        y = torch.tensor(x < 0.6, dtype=torch.bool, device=device)

        return y.view(time, *shape).byte()
    else:
        # Compute firing rates in seconds as function of data intensity,
        # accounting for simulation time step.
        rate = torch.zeros(size, device=device)
        rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

        # Create Poisson distribution and sample inter-spike intervals
        # (incrementing by 1 to avoid zero intervals).
        dist = torch.distributions.Poisson(rate=rate, validate_args=False)
        intervals = dist.sample(sample_shape=torch.Size([time + 1]))
        intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

        # Calculate spike times by cumulatively summing over time dimension.
        times = torch.cumsum(intervals, dim=0).long()
        times[times >= time + 1] = 0

        # Create tensor of spikes.
        spikes = torch.zeros(time + 1, size, device=device)#.byte()
        spikes[times, torch.arange(size)] = 1
        spikes = spikes[1:]

        return spikes.view(time, *shape)

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
                # print(row)
                ind = row['index'].split('e')
                # print(ind)
                ind = float(ind[0]) * (10 ** 5)
                ind = int(ind)
                group_map[ind] = [float(row['x']), float(row['y']), float(row['z']), int(row['label'])]
    return group_map

train_data = parse_group_csv("data/01.csv")
test_data = parse_group_csv("data/09.csv")

data_train = AccelerometerDataset(train_data)
data_test = AccelerometerDataset(test_data)
data_train_loader = DataLoader(data_train, batch_size=1, shuffle=False)

train_loader = data_train_loader
test_loader = DataLoader(data_test, batch_size=1, shuffle=False)



# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================

# Taken from Lab 1, change
weights_save = 1
ucapture  = 1/2
usearch   = 1/2048
ubackoff  = 1/2

# Posneg

# SNL
# 1/2 1/1024 1/2 -> .4415
# 1/2 1/9162 1/2 -> .4943
# 1/8 1/1024 1/8 -> .7017
# 1/16 1/1024 1/16 -> .7017
# 1/16 1/9162 1/16 -> .7020
# 1/16 1/9162 1/2 -> .7020
# 1/2 1/9162 1/16 -> .6726
# 1/32 1/9162 1/2 -> .6798

# RNL
# 1/2 1/1024 1/2 -> .6726
# 1/4 1/1024 1/4 -> .6726
# 1/2 1/2048 1/2 -> .6726
# 1/8 1/2048 1/8 -> .6703
# 1/16 1/2048 1/16 -> .6703
# 1/32 1/2048 1/32 -> .6501
# 1/2 1/4096 1/2 -> .6726
# 1/2 1/9162 1/2 -> .7020
# 1/2 1/18324 1/2 -> .7020
# 1/2 1/1024 3/4 -> .6726
# 1/16 1/9162 1/16 -> .6726
# 1/32 1/9162 1/2 -> .7020

# Poisson

# SNL
# 1/2 1/1024 1/2 -> .6497
# 1/4 1/1024 1/4 -> .4689
# 1/2 1/1024 1/4 -> .4689
# 1/4 1/1024 1/2 -> .6497
# 1/2 1/512 1/2 -> .4690
# 1/2 1/2048 1/2 -> .6497
# 1/4 1/2048 1/2 -> .6497
# 1/8 1/2048 1/2 -> .4719
# 1/8 1/1024 1/2 -> .4719
# 1/2 1/1024 1/8 -> .4689
# 1/4 1/512 1/4 -> .4690
# 1/8 1/512 1/8 -> .4688
# 1/2 1/4096 1/2 -> .6497
# 1/2 1/9162 1/2 -> .4719
# 1/2 1/1024 2/3 -> .6497
# 1/2 1/1024 3/4 -> .6497

# RNL
# 1/2 1/1024 1/2 -> .4688
# 1/4 1/1024 1/4 -> .4688
# 1/2 1/1024 1/4 -> .4688
# 1/4 1/1024 1/2 -> .4688
# 1/2 1/512 1/2 -> .4688
# 1/2 1/2048 1/2 -> .4719
# 1/4 1/2048 1/2 -> .4719
# 1/8 1/2048 1/2 -> .4719
# 1/8 1/2048 1/4 -> .4719
# 1/8 1/2048 1/8 -> .4688
# 1/2 1/4096 1/2 -> .4719
# 1/2 1/512 1/4 -> .4688
# 1/4 1/512 1/4 -> .4688
# 1/2 1/9162 1/2 -> .4719
# 1/2 1/1024 2/3 -> .4688
# 1/2 1/2048 2/3 -> .4719
# 1/2 1/2048 3/4 -> .4719

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
classes_v = 7
thetav_lo = 1/32
thetav_hi = 15/32
tau_eff   = 2

### Enabling CUDA support for GPU ###
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Layer Initialization ###

# clayer = TNNColumnLayer(1, 3, 1, 2, 12, 5, ntype="snl", device=device)
clayer = TNNColumnLayer(1, 3, 1, 6, 12, 20, ntype="snl", device=device)
# snl, 30 -> .4444
# snl, 400 -> .4415
# snl, 20 -> .6479
# snl, 19 -> .4719
'''
Confusion Matrix:
tensor([[0.0000e+00, 2.6040e+03, 5.2300e+02, 2.3244e+04, 2.1032e+04, 3.8450e+03,
         2.8000e+03, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 4.1541e+04, 2.9670e+03, 2.3100e+02, 1.1420e+03, 6.5000e+01,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 5.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])
Purity:  tensor(0.6479)
Coverage:  tensor(1.0000)
'''
# rnl, 20 -> .4688
# rnl, 10 -> .4688
# snl, 10 -> .4415
# snl, 15 -> .4445
# snl, 25 -> .4719
# snl, 18 -> .4445
# snl, 22 -> .4445

# Posneg
# snl 5 .6796
# snl 10 .6270
# snl 15 .4415
# snl 20 .4415

# rnl 5 .6726
# rnl 10 .6726
# rnl 15 .6726
# rnl 20 .6726s
### Training ###

print("Starting column training")
for epochs in range(1):
    start = time.time()

    for idx, (data,target) in enumerate(train_loader):
        if idx >= 100000:
            break
        
        # Error because we are using the old lab solution with new data format
        # Need to change lab solution to accept a 2 x 1 x 3 dimension tensor
        if encoding:
            out1, layer_in1, layer_out1 = clayer(data[0].permute(1,0))
        else:
            out1, layer_in1, layer_out1 = clayer(data[0].permute(1,0))
        clayer.weights = clayer.stdp(layer_in1, layer_out1, clayer.weights, ucapture, usearch, ubackoff)

        endt                   = time.time()
        print("                                 Time elapsed: {0}\r".format(endt-start), end="")

    end   = time.time()
    print("Column training done in ", end-start)


### Display and save weights as images ###

# if weights_save == 1:

#     image_list = []
#     for i in range(12):
#         temp = clayer.weights[i].reshape(56,28)
#         image_list.append(temp)

#     out = torch.stack(image_list, dim=0).unsqueeze(1)
#     save_image(out, 'column_visweights_snl.png', nrow=6)


### Testing and computing metrics ###

table    = torch.zeros((12,10))
pred     = torch.zeros(10)
totals   = torch.zeros(10)

print("Starting testing")
start    = time.time()

for idx, (data,target) in enumerate(test_loader):
    if idx >= 99998:
        break
    print("Sample: {0}\r".format(idx), end="")

    if cuda:
        data                    = data.cuda()
        target                  = target.cuda()
        
    if encoding:
        out1, layer_in1, layer_out1 = clayer(data[0].permute(1,0))
    else:
        out1, layer_in1, layer_out1 = clayer(data[0].permute(1,0))
    out = torch.flatten(out1)

    arg = torch.nonzero(out != float('Inf'))

    if arg.shape[0] != 0:
        table[arg[0].long(), target[0]] += 1

    endt = time.time()
    print("                                 Time elapsed: {0}\r".format(endt-start), end="")

end = time.time()
print("Testing done in ", end-start)

print("Confusion Matrix:")
print(table)

maxval   = torch.max(table, 1)[0]
totals   = torch.sum(table, 1)
pred     = torch.sum(maxval)
covg_cnt = torch.sum(totals)

print("Purity: ", pred/covg_cnt)
print("Coverage: ", covg_cnt/(idx+1))


