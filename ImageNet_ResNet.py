# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:22:30 2019

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
from tiny_imagenet_loader import data_loader

def main():
    
    train_set, val_set = data_loader(rootdir = '../Code/data/ImageNet/tiny-imagenet-200/', normalized =True)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 128)
    
    '''
    #Calculating mean/sd of the pixel channels
    mean_pixel = []
    sd_pixel = []
    for data,_ in train_loader:
        mean_pixel.append(np.mean(data.numpy(), axis = (0, 2, 3)))
        sd_pixel.append(np.sqrt(np.var(data.numpy(), axis = (0, 2, 3))))
    '''
    
    ####Subset of data#####
    #choice of random index
    
#     rand_idx = random.choices(range(100000), k = 320)
#     rand_idx2 = random.choices(range(10000), k = 320)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, 
#                                                sampler = torch.utils.data.sampler.SubsetRandomSampler(rand_idx))
#     val_loader = torch.utils.data.DataLoader(val_set, batch_size = 32, 
#                                                sampler = torch.utils.data.sampler.SubsetRandomSampler(rand_idx2))
    
    #######################
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    res_net = models.resnet18(num_classes = 200)
    res_net.to(device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(res_net.parameters(), lr = 3e-4, weight_decay = 0.0001)
    
    epochs = 5
    print_every = 1
    
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    
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
    
    #####debugging#######
    train_plot = viz.line(Y = torch.tensor([0]).zero_(), opts = dict(title = 'Training Loss Tracker',
                               xlabels = 'Iteration',
                               ylabels = 'Time'))
    
    #####################
    
    epoch_time = []
    for e in range(epochs):
        print('Epoch ', e)
        epoch_start = time.time()
        #Training
        num_correct = 0
        num_samples = 0
        
        iter_train_loss = []
        iter_train_acc = []

        #Training
        for t, (x,y) in enumerate(train_loader):
            num_correct = 0
            num_samples = 0
            res_net.train()
            x = x.to(device = device)
            y = y.to(device = device)
            
            scores = res_net(x)
            loss = loss_fcn(scores, y)
            iter_train_loss.append(loss.item()) #to keep track of loss
            
            loss.backward()
            optimizer.step()    
            optimizer.zero_grad()
            
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            
            iter_train_acc.append(float(num_correct)/num_samples)            
            viz_tracker(train_plot, torch.tensor([iter_train_loss[t]]), torch.tensor([t]))
            
            if (t % print_every) == 0:
                print('Training: Iteration %d, loss = %.3f' % (t, loss.item()))    
                print('Training: Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100*float(num_correct)/num_samples))
        
        train_loss.append(np.mean(iter_train_loss))
        train_acc.append(np.mean(iter_train_acc))
        
        #Validation
        print('Checking accuracy on validation set')
        num_correct = 0
        num_samples = 0
        res_net.eval()
        with torch.no_grad():
            iter_valid_loss = []
            iter_valid_acc = []
            for t, (x, y) in enumerate(val_loader):               
                x = x.to(device = device)
                y = y.to(device = device)
                
                scores = res_net(x)
                _, preds = scores.max(1)
                loss = loss_fcn(scores, y)
                iter_valid_loss.append(loss.item())
                
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                
                iter_valid_acc.append(float(num_correct)/num_samples)                
                
            valid_loss.append(np.mean(iter_valid_loss))
            valid_acc.append(np.mean(iter_valid_acc))
            print('Validation: Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100*float(num_correct)/num_samples))
    
        #Epoch time
        epoch_end = time.time()
        epoch_time.append(epoch_end- epoch_start)
        
        #Update plots
        viz_tracker(epoch_time_plot, torch.tensor([epoch_time[e]]), torch.tensor([e]) )
        viz_tracker(loss_plot, torch.tensor([[train_loss[e], valid_loss[e]]]), torch.tensor([[e,e]]))
        viz_tracker(acc_plot, torch.tensor([[train_acc[e], valid_acc[e]]]), torch.tensor([[e,e]]))
        
        #Save resulting arrays so far every 10 or so epochs
        if((e+1) % 10 == 0):
            with open('train_loss.pkl', 'wb') as handle:
                pickle.dump(train_loss, handle, protocol = pickle.HIGHEST_PROTOCOL)
            with open('valid_loss.pkl', 'wb') as handle:
                pickle.dump(valid_loss, handle, protocol = pickle.HIGHEST_PROTOCOL)
            with open('train_acc.pkl', 'wb') as handle:
                pickle.dump(train_acc, handle, protocol = pickle.HIGHEST_PROTOCOL)
            with open('valid_acc.pkl', 'wb') as handle:
                pickle.dump(valid_acc, handle, protocol = pickle.HIGHEST_PROTOCOL)
            with open('epoch_time.pkl', 'wb') as handle:
                pickle.dump(epoch_time, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    #calculate test loss
#     num_correct = 0
#     num_samples = 0
#     res_net.eval()
#     for x, y in test_loader:
#         x = x.to(device = device)
#         y = y.to(device = device)
        
#         scores = res_net(x)
#         _, preds = scores.max(1)
#         loss = loss_fcn(scores, y)
        
#         num_correct += (preds == y).sum()
#         num_samples += preds.size(0)

#     print(float(num_correct)/num_samples)
    
#     with open('test.pkl', 'wb') as handle:
#         pickle.dump([loss.item(), float(num_correct)/num_samples], handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    
if __name__ == '__main__':
    main()