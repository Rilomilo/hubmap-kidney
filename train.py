import gc
import time
import pathlib
import argparse
import sys

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from utils import set_seeds, rle_numba_encode, make_grid
set_seeds()
from models import get_model
from dataloader import get_dataset
from loss import loss_fn



@torch.no_grad()
def validation(model, loader, loss_fn, device):
    losses = []
    model.eval() # disable dropout
    for image, target in loader:
        image, target = image.to(device), target.float().to(device)
        output = model(image)['out']
        loss = loss_fn(output, target)
        losses.append(loss.item())
        
    return np.array(losses).mean()

def train(
    data_path,
    batch_size,
    device,
    window,
    min_overlap,
    new_size,
    epochs,
    num_workers,
):
    train_dataset, valid_dataset = get_dataset(data_path, window, min_overlap, new_size)

    trainloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)

    model = get_model()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    ### Table for results
    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Time, m
    '''
    #          Epoch         metrics            time
    raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'




    print(header)

    best_loss = 10
    
    for epoch in range(1, epochs+1):
        losses = []
        start_time = time.time()
        model.train()
        for image, target in trainloader:
            image, target = image.to(device), target.float().to(device)
            optimizer.zero_grad()
            output = model(image)['out']
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        vloss = validation(model, validloader, loss_fn, device)
        print(raw_line.format(epoch, np.array(losses).mean(), vloss,(time.time()-start_time)/60**1))
        losses = []
        
        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), 'model_best.pth')


if __name__=="__main__":
    opts = {
        'data_path': 'data',
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'window': 1024,
        'min_overlap': 32,
        'new_size': 256,
        'epochs': 10,
        'num_workers': 0,
    }
    train(**opts)
