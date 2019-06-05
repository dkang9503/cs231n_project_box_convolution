# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:49:12 2019

@author: dkang
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from box_convolution import BoxConv2d
import pickle
import visdom
import time
import random
from tiny_imagenet_loader import data_loader
from torch.utils.data.sampler import SubsetRandomSampler

def main():
    train_set = torchvision.datasets.ImageFolder(
            root = './data/ImageNet/tiny-imagenet-200/train',
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean =[.4802486, .44807222, .39754647],
                                                                 std = [.2769859, .26906505, .2820814])])
    )
    
    val_set = torchvision.datasets.ImageFolder(
            root = './data/ImageNet/tiny-imagenet-200/val',
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean =[.4802486, .44807222, .39754647],
                                                                 std = [.2769859, .26906505, .2820814])])
    )
    
    test_set = torchvision.datasets.ImageFolder(
            root = './data/ImageNet/tiny-imagenet-200/test',
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean =[.4802486, .44807222, .39754647],
                                                                 std = [.2769859, .26906505, .2820814])])
    )
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 256)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 256)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 256)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    res_net = nn.Sequential( 
            BoxConv2d(3, 32, 64, 64)
    )
    
    sample_x, sample_y = next(iter(train_loader))
    
    ex = res_net(sample_x)