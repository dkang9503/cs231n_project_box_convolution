# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:32:37 2019

@author: dkang
"""

import visdom
import torch
import pickle
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from models.ModResNet50 import BoxResNet502, ModResNet502


#importing visualization    
viz = visdom.Visdom()
viz.close()

def viz_tracker(plot, value, num):
    viz.line(X = num, Y = value, win = plot, update = 'append')

loss_plot = viz.line(Y= torch.tensor([[0,0]]).zero_(), opts = dict(title = 'Loss Tracker', 
                     legend = ['Training Loss', 'Validation Loss'],
                     xlabels = 'Epochs',
                     ylabels = 'Loss',
                     show_legend = True))

acc_plot = viz.line(Y= torch.tensor([[0,0]]).zero_(), opts = dict(title = 'Accuracy Tracker', 
                     legend = ['Training Accuracy', 'Validation Accuracy'],
                     xlabels = 'Epochs',
                     ylabels = 'Accuracy',
                     show_legend = True))

epoch_time_plot = viz.line(Y = torch.tensor([0]).zero_(), opts = dict(title = 'Epoch Time Tracker',
                           xlabels = 'Epochs',
                           ylabels = 'Time'))


####LOADING DATA####
with open('BXepoch_time.pkl', 'rb') as handle:
    BX_epoch_time = pickle.load(handle)

with open('BXtrain_acc1.pkl', 'rb') as handle:
    BX_train_acc1 = pickle.load(handle)

with open('BXtrain_acc5.pkl', 'rb') as handle:
    BX_train_acc5 = pickle.load(handle)

with open('BXvalid_acc1.pkl', 'rb') as handle:
    BX_valid_acc1 = pickle.load(handle)

with open('BXvalid_acc5.pkl', 'rb') as handle:
    BX_valid_acc5 = pickle.load(handle)
    
with open('BXtrain_loss.pkl', 'rb') as handle:
    BX_train_loss = pickle.load(handle)

with open('BXvalid_loss.pkl', 'rb') as handle:
    BX_valid_loss = pickle.load(handle)
    
with open('RN_epoch_time.pkl', 'rb') as handle:
    RN_epoch_time = pickle.load(handle)

with open('RN_train_acc1.pkl', 'rb') as handle:
    RN_train_acc1 = pickle.load(handle)

with open('RN_train_acc5.pkl', 'rb') as handle:
    RN_train_acc5 = pickle.load(handle)

with open('RN_valid_acc1.pkl', 'rb') as handle:
    RN_valid_acc1 = pickle.load(handle)

with open('RN_valid_acc5.pkl', 'rb') as handle:
    RN_valid_acc5 = pickle.load(handle)
    
with open('RN_train_loss.pkl', 'rb') as handle:
    RN_train_loss = pickle.load(handle)

with open('RN_valid_loss.pkl', 'rb') as handle:
    RN_valid_loss = pickle.load(handle)
    
##########################

#######PLOTS#############
acc_plot = viz.line(Y=np.column_stack((BX_train_acc1,RN_train_acc1, BX_valid_acc1, RN_valid_acc1, BX_train_acc5, RN_train_acc5, BX_valid_acc5, RN_valid_acc5 )), 
                    X= np.column_stack((list(range(50)), list(range(50)), list(range(50)), list(range(50)),\
                                        list(range(50)), list(range(50)), list(range(50)), list(range(50)))),
                    opts = dict(title = 'Accuracy',
                                legend = ['Box Train Top1', 'RN Train Top1', 'Box Valid Top1', 'RN Valid Top1',\
                                          'Box Train Top5', 'RN Train Top5', 'Box Valid Top5', 'RN Valid Top5'],
                                xlabels = 'Epochs',
                                ylabels = 'Percentage',
                                show_legend = True))

loss_plot = viz.line(Y=np.column_stack((BX_train_loss, RN_train_loss, BX_valid_loss, RN_valid_loss)), 
                     X= np.column_stack((list(range(50)), list(range(50)), list(range(50)), list(range(50)))),
                    opts = dict(title = 'Loss',
                                legend = ['Box Train Loss', 'RN Train Loss', 'Box Valid Loss', 'RN Valid Loss'],
                                xlabels = 'Epochs',
                                ylabels = 'Loss',
                                show_legend = True))

epoch_plot = viz.line(Y = np.column_stack(( BX_epoch_time, RN_epoch_time )), X= np.column_stack((list(range(50)), list(range(50)))),
                    opts = dict(title = 'Epoch Time',
                                legend = ['Box ResNet', 'Regular ResNet'],
                                xlabels = 'Epochs',
                                ylabels = 'Seconds per Epoch',
                                show_legend = True))

np.mean(BX_epoch_time)
np.mean(RN_epoch_time)

#########################
    
    
#####PREDICTION###########
test_set = torchvision.datasets.ImageFolder(root = '../data/ImageNet/tiny-imagenet-200/val/images', transform=
                                             transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean =[.4802486, .44807222, .39754647],
                                                                         std = [.2769859, .26906505, .2820814])]))

test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = True)

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

###### Saliency Maps
x, y = next(iter(test_loader))
x.requires_grad_()
hb_model.eval()
hb_output = hb_model(x)
hb_output1 = hb_output.gather(1, y.view(-1, 1)).squeeze()
loss = torch.sum(hb_output1)
loss.backward()
hb_sal = x.grad.data
hb_sal, idx = hb_sal.max(dim=1)

hb_sal = hb_sal.numpy()
N = x.shape[0]
x = torch.cat([])

for i in range(N):
    plt.subplot(2, N, i+1)
    plt.imshow(x[i])
    plt.axis('off')
    plt.subplot(2, N, N+i+1)    
    plt.imshow(hb_sal[i], cmap = plt.cm.hot)
    plt.axis('off')
    plt.gcf().set_size_inches(12,5)

plt.show
######



loss_fcn = torch.nn.CrossEntropyLoss()

hb_test1 = []
hb_test5 = []
rn_test1 = []
rn_test5 = []

for t, (x, y) in enumerate(test_loader):
    x = x.to(device = device)
    y = y.to(device = device)

    hb_scores = hb_model(x)        
    rn_scores = rn_model(x)
    hb_loss = loss_fcn(hb_scores, y)
    rn_loss = loss_fcn(rn_scores, y)

    hb_acc1, hb_acc5 = accuracy(hb_scores, y, topk=(1,5))  
    rn_acc1, rn_acc5 = accuracy(rn_scores, y, topk=(1,5))  
    
    hb_test1.append(hb_acc1.data.cpu().numpy()[0])
    hb_test5.append(hb_acc5.data.cpu().numpy()[0])
    rn_test1.append(rn_acc1.data.cpu().numpy()[0])
    rn_test5.append(rn_acc5.data.cpu().numpy()[0])

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
###########################