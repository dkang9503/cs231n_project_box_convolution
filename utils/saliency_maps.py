# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:05:06 2019

@author: dkang
"""
import visdom
import torch
import pickle
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from models.ModResNet50 import BoxResNet502, ModResNet502

def preprocess(img, size=64):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.4802486, .44807222, .39754647],
                    std=[.2769859, .26906505, .2820814]),
        transforms.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def compute_saliency_maps(X, y, model):
    model.eval()
    X.requires_grad_()
    saliency = None
    output = model(X)
    output = output.gather(1, y.view(-1, 1)).squeeze()
    loss = torch.sum(output)
    loss.backward()
    saliency = X.grad.data
    saliency = saliency.abs()
    saliency, idx = saliency.max(dim=1)
    
    return saliency

def show_saliency_maps(X, y, model):
    X_tensor = torch.cat([preprocess(F.to_pil_image(x)) for x in X ], dim =0)
    y_tensor = torch.LongTensor(y)
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i+1)
        plt.imshow(X[i].transpose(0, 2).transpose(0,1))
        plt.axis('off')
        plt.subplot(2,N,N+i+1)
        plt.imshow(saliency[i], cmap = plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12,5)
    plt.show
        
    
test_set = torchvision.datasets.ImageFolder(root = '../data/ImageNet/tiny-imagenet-200/val/images',
                                            transform = transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(test_set, batch_size = 6, shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

hb_dict = torch.load('20modelstateHB.pth', map_location = 'cpu')
rn_dict = torch.load('40modelstateRN.pth', map_location = 'cpu')

from collections import OrderedDict
new_hb_state_dict = OrderedDict()

for k, v in hb_dict['state_dict'].items():
    name = 'module.' + k
    new_hb_state_dict[name] = v

hb_model = BoxResNet502(num_classes = 200)
hb_model = torch.nn.DataParallel(hb_model)
hb_model.load_state_dict(new_hb_state_dict, strict = False)
hb_model.eval()

rn_model = ModResNet502(num_classes = 200)
rn_model.load_state_dict(rn_dict['state_dict'], strict = False)
rn_model.eval()

X, y = next(iter(test_loader))
show_saliency_maps(X,y, hb_model)
show_saliency_maps(X,y, rn_model)