# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:47:54 2019

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
from torch.utils.data.sampler import SubsetRandomSampler
import utils.transforms as extended_transforms
from models.PSP_ResNetModel import Resnet50_8s_psp
from models.PSP_BoxNetModel import BoxResnet50_8s_psp
from dataset.voc import VOCSegmentation
from sklearn.metrics import confusion_matrix
import sklearn as sk


def main():
    mean_std = ([0.45679754, 0.44313163, 0.4082983], [0.23698017, 0.23328756, 0.23898676])    
    val_x_transform = transforms.Compose([
                transforms.CenterCrop((250, 400)),
                transforms.ToTensor(),
                transforms.Normalize(*mean_std)
                ])
        
    val_y_transform = transforms.Compose([
            transforms.CenterCrop((250, 400)),
            extended_transforms.MaskToTensor()
            ])
    
    
    train_data = torchvision.datasets.SBDataset(
        root = '../data',
        image_set = 'train',
        mode = 'segmentation',
        transforms = extended_transforms.JointTransform()
    )
    
    
    val_data = torchvision.datasets.SBDataset(
        root = '../data',
        image_set = 'val',
        mode = 'segmentation',
        transforms = extended_transforms.JointTransformTuning()
    )
    
    
    tuning = False
    save_idx = False
    
    #Hyperparameters
    lr = .001
    weight_decay = 1e-4
    momentum = .9
    epochs = 200
    print_every = 100
    batch_size = 16

    mean_std = ([0.45679754, 0.44313163, 0.4082983], [0.23698017, 0.23328756, 0.23898676])    
    val_x_transform = transforms.Compose([
                transforms.CenterCrop((250, 400)),
                transforms.ToTensor(),
                transforms.Normalize(*mean_std)
                ])
        
    val_y_transform = transforms.Compose([
            transforms.CenterCrop((250, 400)),
            extended_transforms.MaskToTensor()
            ])

    if tuning:
        train_transform = extended_transforms.JointTransformTuning()
    else:
        train_transform = extended_transforms.JointTransform() #Horizontal flip, cropping, to tensor, normalization
    
    train_set = VOCSegmentation(
            root = '../data',
            year = '2012',
            image_set = 'train',            
            transforms = train_transform            
    )

    val_set = VOCSegmentation(
            root = '../data',
            year = '2012',
            image_set = 'val',
            transform = val_x_transform,
            target_transform = val_y_transform
    )
    
    val_set2 = VOCSegmentation(
            root = '../data',
            year = '2012',
            image_set = 'val',
            transforms = extended_transforms.JointTransformImageOnly()
    )        
        
    if tuning:
        rand_idx = random.choices(range(len(train_set)), k = 80)
        rand_idx2 = random.choices(range(len(val_set)), k = 32)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, 
                                                   sampler = SubsetRandomSampler(rand_idx))
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, 
                                                 sampler = SubsetRandomSampler(rand_idx2))
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True)
    
    '''    
    #calculating mean/sd of pixels to normalize 
    mean_pixel = []
    sd_pixel = []
    for data,_ in train_loader:
        mean_pixel.append(np.mean(data.numpy(), axis = (0, 2, 3)))
        sd_pixel.append(np.sqrt(np.var(data.numpy(), axis = (0, 2, 3))))
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 21
    loss_fcn = torch.nn.CrossEntropyLoss(ignore_index = 255)
    blah = BoxResnet50_8s_psp(num_classes = n_classes)
    x, y = next(iter(train_loader))
    ex = blah(x)
    loss = loss_fcn(input = ex, target = y)
    
    with torch.no_grad():    
        pred = ex.max(1)[1]        
                
        pred2 = pred[y != 255]
        y2 = y[y != 255]        
        
        conf_mat2 = torch.zeros(n_classes, n_classes)
        
        lin_index = pred2 * n_classes + y2
        blah = conf_mat2.put_(lin_index, torch.tensor(1.).expand_as(lin_index), accumulate = True) + 1e-6
        blah.diag().sum()/blah.sum()
        
        true_positive = blah.diag()
        false_positive = blah.sum(0) - true_positive
        false_negative = blah.sum(1) - true_positive
        iou = true_positive/(true_positive + false_positive + false_negative)
        
        (blah.diag()/(blah.sum(1)+blah.sum(0) - blah.diag())).mean()
        
        
        #torch.sum(pred == y).item()/pred.numel()
        ####
        intersection = (pred2 & y2).float().sum((1,2))
        union = (pred2 | y2).float().sum((1,2))
        iou = (intersection + 1e-6)/(union + 1e-6)
        ####
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.long)
        n_class_array = np.arange(21)
        confusion_matrix += sk.metrics.confusion_matrix(pred2.view(-1), y2.view(-1), n_class_array)        
        #sk.metrics.confusion_matrix(pred.view(-1), y.view(-1), n_class_array)
        np.diag(confusion_matrix).sum()/confusion_matrix.sum()
        
        confusion_matrix = torch.zeros((n_classes, n_classes), device = device, dtype=torch.long)
        output = pred.view(-1)
        target = y.view(-1)
        
        output_2 = output[target != 255].to(confusion_matrix)
        target_2 = target[target != 255].to(output)
        
        conf_mat_up = torch.bincount(target_2 * (n_classes+1) + output_2, minlength = (n_classes + 1)**2)
        conf_mat_test = conf_mat_up.view(n_classes+1, n_classes+1)[1:, 1:].to(confusion_matrix)
        
        confusion_matrix = torch.zeros((n_classes, n_classes), device = device, dtype=torch.long)
        for t, p in zip(target, output):
            if t != 255:
                confusion_matrix[t.long(), p.long()] +=1
        
        
        conf_diag = conf_mat_test.diag().type(torch.float) + 1e-6
        conf_sums = conf_mat_test.sum(1).type(torch.float) + 1e-6
        print((conf_diag/conf_sums).numpy())
    
    ####
    label = pred.numpy().reshape(-1)
    target = y.numpy().reshape(-1) 
    
    jsc(target, label)
    
    
if __name__ == '__main__':
    main()