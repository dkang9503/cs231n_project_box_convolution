# -*- coding: utf-8 -*-
"""
Created on Mon May 13 08:49:42 2019

@author: dkang
"""
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import pickle
import visdom
import time
import random
from models.HalfBoxResNet import resnetHalfBox
from torch.utils.data.sampler import SubsetRandomSampler

def main():
    
    train_set = torchvision.datasets.ImageFolder(root = './data/ImageNet/tiny-imagenet-200/train', transform=
                                                 transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean =[.4802486, .44807222, .39754647],
                                                                             std = [.2769859, .26906505, .2820814])]))
    
    test_set = torchvision.datasets.ImageFolder(root = './data/ImageNet/tiny-imagenet-200/test', transform=
                                                 transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean =[.4802486, .44807222, .39754647],
                                                                             std = [.2769859, .26906505, .2820814])]))
    
    train_indices = np.empty(90000)
    val_indices = np.empty(10000)
    
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
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = True)
    
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
    '''
    rand_idx = random.choices(range(100000), k = 192)
    rand_idx2 = random.choices(range(10000), k = 320)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, 
                                               sampler = torch.utils.data.sampler.SubsetRandomSampler(rand_idx))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 32, 
                                               sampler = torch.utils.data.sampler.SubsetRandomSampler(rand_idx2))
    '''
    #######################
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    res_net = resnetHalfBox(num_classes = 200)
    res_net.to(device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(res_net.parameters(), lr = 3e-4, weight_decay = 0)
    
    epochs = 100
    print_every = 100
    
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
    
    
    
    epoch_time = []
    for e in range(epochs):        
        #####debugging#######
        train_plot = viz.line(Y = torch.tensor([0]).zero_(), opts = dict(title = 'Training Loss Tracker',
                                   xlabels = 'Iteration',
                                   ylabels = 'Time'))
    
        #####################
        
        epoch_start = time.time()
        #Training
        res_net.train()
        iter_train_loss = []
        iter_train_acc = []

        for t, (x,y) in enumerate(train_loader):
            num_correct = 0
            num_samples = 0
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
            
            if t % print_every == 0:
                print('Iteration %d, loss = %.3f' % (t, loss.item()))    
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
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100*float(num_correct)/num_samples))
    
        #Epoch time
        epoch_end = time.time()
        epoch_time.append(epoch_end- epoch_start)
        
        #Update plots
        viz_tracker(epoch_time_plot, torch.tensor([epoch_time[e]]), torch.tensor([e]) )
        viz_tracker(loss_plot, torch.tensor([[train_loss[e], valid_loss[e]]]), torch.tensor([[e,e]]))
        viz_tracker(acc_plot, torch.tensor([[train_acc[e], valid_acc[e]]]), torch.tensor([[e,e]]))
        viz.close(win = train_plot)
        
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
    num_correct = 0
    num_samples = 0
    res_net.eval()
    for x, y in test_loader:
        x = x.to(device = device)
        y = y.to(device = device)

        scores = res_net(x)
        _, preds = scores.max(1)
        loss = loss_fcn(scores, y)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    
    with open('test_pred.pkl', 'wb') as handle:
        pickle.dump(preds, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open('test_scores.pkl', 'wb') as handle:
        pickle.dump(scores, handle, protocol = pickle.HIGHEST_PROTOCOL)    
    
    with open('test_acc.pkl', 'wb') as handle:
        pickle.dump(float(num_correct)/num_samples, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    main()