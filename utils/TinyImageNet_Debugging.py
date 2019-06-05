# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:43:57 2019

@author: dkang
"""
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib import pyplot as plt


train_set = torchvision.datasets.ImageFolder(root = '../Code/data/ImageNet/tiny-imagenet-200/train', transform=
                                                 transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean =[.4802486, .44807222, .39754647],
                                                                             std = [.2769859, .26906505, .2820814])]))
    
test_set = torchvision.datasets.ImageFolder(root = '../data/ImageNet/tiny-imagenet-200/val/images', transform=
                                             transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean =[.4802486, .44807222, .39754647],
                                                                         std = [.2769859, .26906505, .2820814])]))

train_indices = np.empty(90000, dtype = int)
val_indices = np.empty(10000, dtype = int)

for i in range(200):
    temp_list = np.array(range(500*i, 500+500*i))
    rand_samp_idx = random.sample(range(500), 50)
    mask = np.array([False]*500)
    mask[np.array(rand_samp_idx)] = True
    val_indices[(50*i):(50*i+50)] = temp_list[mask]
    train_indices[(450*i):(450*i+450)] = temp_list[mask == False]

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, 
                                           sampler = SubsetRandomSampler(train_indices))

val_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, 
                                           sampler = SubsetRandomSampler(val_indices))

test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128)

plt.imshow(test_set[0][0].numpy().transpose(), interpolation = 'nearest')
plt.show