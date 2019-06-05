# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:23:17 2019

@author: dkang
"""

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle
import visdom
import time
import random
from torch.utils.data import Dataset
import os

class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    """
    def __init__(self, root, transform = None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train_dir = os.path.join(root, 'train')
        self.val_dir = os.path.join(root, 'val')
        self.labels = {}
        self.images = []
        self.samples =[]
        self.data = []
        self.num_images = 0
        self.train_set = []
        
        self.train_set = torchvision.datasets.ImageFolder(root = os.path.join(root, 'train'), transform = transform)
    
        self.labels = self.train_set.class_to_idx
        
        val_text_file = open( os.path.join(self.val_dir, 'val_annotations.txt') , 'r')
        val_lines = val_text_file.readlines()
        
        val_new_lines = []
        for line in val_lines:            
            cur_line = line.split('\t')
            holder = cur_line[1]
            cur_line[1] = self.labels[cur_line[1]]
            val_new_lines.append(cur_line)
            self.samples.append(cur_line[0] + ' ' + holder + ' '+ str(cur_line[1]))
            
        val_set = torchvision.datasets.ImageFolder(root = os.path.join(root, 'val'), transform = transform)
        
        for i,_ in enumerate(val_set):
            self.data.append((val_set[i][0], val_new_lines[i][1]))
        
        self.num_images = len(val_set)
        
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, index):
        return self.data[index] 
    
    def train_data(self):
        return self.train_set
    
    def samples(self):
        return self.samples

def data_loader(rootdir, normalized = True):
    if normalized:
        transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean =[.4802486, .44807222, .39754647],
                                                                 std = [.2769859, .26906505, .2820814])])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        
    val_set = TinyImageNet(rootdir, transform)
    
    train_set = val_set.train_data()
        
    return train_set, val_set