# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:47:54 2019

@author: dkang
"""
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms
import pickle
import visdom
import time
import random
import os
from torch.utils.data.sampler import SubsetRandomSampler
import sys
sys.path.insert(0, '../')
import utils.transforms as extended_transforms
from utils.schedulers import PolynomialLR
from models.PSP_ResNetModel import Resnet50_8s_psp
from dataset.voc import VOCSegmentation


def main():
    print("Updated 6/3 5:39PM")
    
    tuning = False
    save_idx = True
    
    #Hyperparameters
    lr = 3e-5
    weight_decay = 1e-3
    #momentum = .9
    epochs = 3001
    print_every = 100
    batch_size = 16
        
    folder_dir = './../../Results/PSP_' + 'lr' + str(lr) + '_wd' + str(weight_decay) \
    + '_epochs' + str(epochs) + 'Adam'
    
    ############DATA LOADING AND TRANSFORMS##############    
    if tuning:
        train_transform = extended_transforms.JointTransformTuning()
    else:
        train_transform = extended_transforms.JointTransform() #Horizontal flip, cropping, to tensor, normalization
    
    train_set = torchvision.datasets.SBDataset(
        root = '../data',
        image_set = 'train',
        mode = 'segmentation',
        transforms = train_transform
    )

    val_set = torchvision.datasets.SBDataset(
        root = '../data',
        image_set = 'val',
        mode = 'segmentation',
        transforms = extended_transforms.JointTransformTuning()
    )
    
    #Define test set
    np.random.seed(1)
    indices = list(range(len(val_set)))
    np.random.shuffle(indices)
    
    val_indices = indices[0:(len(val_set)//2)]
    test_indices = indices[(len(val_set)//2): len(val_set)]
    
    val_set = torch.utils.data.Subset(val_set, val_indices)
    test_set = torch.utils.data.Subset(val_set, test_indices)
    
    if tuning:
        rand_idx = random.choices(range(len(train_set)), k = 80)
        rand_idx2 = random.choices(val_indices, k = 32)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, 
                                                   sampler = SubsetRandomSampler(rand_idx))
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, 
                                                 sampler = SubsetRandomSampler(rand_idx2))
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = 4)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    
    #Parameters
    n_classes = 21
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_iter = len(train_loader)*epochs
    power = .9
    
    #CREATING DIRECTORY TO SAVE ALL FILES and loading models
    if not tuning and save_idx:    
        resume_train = os.path.exists(folder_dir)
        if not resume_train:
            os.makedirs(folder_dir)
            print("Directory " , folder_dir ,  " Created ")
            
            model = Resnet50_8s_psp(num_classes = n_classes)
            model.to(device)
            loss_fcn = torch.nn.CrossEntropyLoss(ignore_index = 255)
            loss_fcn.to(device)
            
            #Set optimizers/scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)    
            #scheduler = PolynomialLR(optimizer = optimizer, max_iter = max_iter, gamma = power)
            e = 0
                                    
        else:    
            print("Directory " , folder_dir ,  " already exists. Resuming training")   
            model = Resnet50_8s_psp(num_classes = n_classes)
            model.to(device)
            loss_fcn = torch.nn.CrossEntropyLoss(ignore_index = 255)
            loss_fcn.to(device)
            
            #Set optimizers/scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)    
            #scheduler = PolynomialLR(optimizer = optimizer, max_iter = max_iter, gamma = power)
            
            #Load state
            model_dict = torch.load(folder_dir + '/modelstate.pth')
            model.load_state_dict(model_dict['state_dict'])
            optimizer.load_state_dict(model_dict['optimizer'])
            #scheduler.load_state(model_dict['scheduler'])     
            e = model_dict['epoch']
    
    #Initialize vizdom
    viz2 = visdom.Visdom()
    viz2.close()
    
    #Setup data storage/loading
    if not resume_train:
        train_loss = []
        valid_loss = []
        train_class_iou = []
        valid_class_iou = []    
        train_total_iou = []
        valid_total_iou = []
        train_total_acc = []
        valid_total_acc = []
        epoch_time = []
        
        loss_plot = viz2.line(Y= torch.tensor([[0,0]]).zero_(), opts = dict(title = 'Loss Tracker', 
                             legend = ['Training Loss', 'Validation Loss'],
                             xlabels = 'Epochs',
                             ylabels = 'Loss',
                             show_legend = True))

        iou_plot = viz2.line(Y= torch.tensor([[0,0]]).zero_(), opts = dict(title = 'Mean IoU Tracker', 
                             legend = ['Training Mean IoU', 'Validation Mean IoU'],
                             xlabels = 'Epochs',
                             ylabels = 'IoU',
                             show_legend = True))

        acc_plot = viz2.line(Y= torch.tensor([[0,0]]).zero_(), opts = dict(title = 'Accuracy Tracker', 
                             legend = ['Training Pixel Accuracy', 'Validation Pixel Accuracy'],
                             xlabels = 'Epochs',
                             ylabels = 'IoU',
                             show_legend = True))

        epoch_time_plot = viz2.line(Y = torch.tensor([0]).zero_(), opts = dict(title = 'Epoch Time Tracker',
                                   xlabels = 'Epochs',
                                   ylabels = 'Time'))
    else:
        with open(folder_dir +'/RPSPtrain_loss.pkl', 'wb') as handle:
            train_loss = pickle.load(handle)
        with open(folder_dir +'/RPSPvalid_loss.pkl', 'wb') as handle:
            valid_loss = pickle.load(handle)
        with open(folder_dir +'/RPSPtrain_class_iou.pkl', 'wb') as handle:
            train_class_iou = pickle.load(handle)
        with open(folder_dir +'/RPSPtrain_total_iou.pkl', 'wb') as handle:
            train_total_iou = pickle.load(handle)
        with open(folder_dir +'/RPSPvalid_class_iou.pkl', 'wb') as handle:
            valid_class_iou = pickle.load(handle)
        with open(folder_dir +'/RPSPvalid_total_iou.pkl', 'wb') as handle:
            valid_total_iou = pickle.load(handle)
        with open(folder_dir +'/RPSPepoch_time.pkl', 'wb') as handle:
            epoch_time = pickle.load(handle)
        with open(folder_dir +'/RPSPtrain_total_acc.pkl', 'wb') as handle:
            train_total_acc = pickle.load(handle)
        with open(folder_dir +'/RPSPtrain_total_acc.pkl', 'wb') as handle:
            train_total_acc = pickle.load(handle)
        
        loss_plot = viz2.line(Y= torch.tensor([train_loss, valid_loss]), opts = dict(title = 'Loss Tracker', 
                             legend = ['Training Loss', 'Validation Loss'],
                             xlabels = 'Epochs',
                             ylabels = 'Loss',
                             show_legend = True))

        iou_plot = viz2.line(Y= torch.tensor([train_total_iou, valid_total_iou]), \
                             opts = dict(title = 'Mean IoU Tracker', 
                             legend = ['Training Mean IoU', 'Validation Mean IoU'],
                             xlabels = 'Epochs',
                             ylabels = 'IoU',
                             show_legend = True))

        acc_plot = viz2.line(Y= torch.tensor([train_total_acc, valid_total_acc]),\
                             opts = dict(title = 'Accuracy Tracker', 
                             legend = ['Training Pixel Accuracy', 'Validation Pixel Accuracy'],
                             xlabels = 'Epochs',
                             ylabels = 'IoU',
                             show_legend = True))

        epoch_time_plot = viz2.line(Y = torch.tensor([0]).zero_(), opts = dict(title = 'Epoch Time Tracker',
                                   xlabels = 'Epochs',
                                   ylabels = 'Time'))
                    
    #Begin iteration
    while e < epochs:        
        #####debugging#######        
        train_plot = viz2.line(Y = torch.tensor([0]).zero_(), opts = dict(title = 'Training Loss Tracker',
                              xlabels = 'Iteration',
                              ylabels = 'Time'))
        #####################
        print('Epoch ', e)
                
        #Training        
        iter_train_loss = []           
        
        #Initiate confusion matrix to calculate accuracy
        confusion_matrix = torch.zeros((n_classes, n_classes), device = device, dtype = torch.float)
        
        epoch_start = time.time()
        model.train()
        for t ,(x,y) in enumerate(train_loader):            
            #scheduler.step()
            
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            loss = loss_fcn(scores, y)
            iter_train_loss.append(loss.item())
            #add to vizdom
            viz_tracker(viz2, train_plot, torch.tensor([iter_train_loss[t]]), torch.tensor([t]))
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                
            
            with torch.no_grad():                
                pred = scores.max(1)[1]                
                
                #Update confusion matrix. First ignore all the void values                                
                lin_index = pred[y != 255] * n_classes + y[y != 255]
                confusion_matrix += torch.zeros(n_classes,n_classes, device = device).put_(lin_index,\
                                               torch.tensor(1., device = device).expand_as(lin_index), accumulate = True)
                
            if (t % print_every) == 0:
                print('Training: Iteration %d, loss = %.3f' % (t, loss.item()))                    
                        
        train_loss.append(np.mean(iter_train_loss))
        print('Epoch loss is: ' + str(train_loss[e]) )
        #Append accuracy        
        #First convert into float
        confusion_matrix = confusion_matrix.type(torch.float) + 1e-6
        #Add per class IoU
        train_class_iou.append((confusion_matrix.diag()/(confusion_matrix.sum(1) + confusion_matrix.sum(0) -\
                                                      confusion_matrix.diag()) ).cpu().numpy())
        #Add mean IoU
        train_total_iou.append( (confusion_matrix.diag()/(confusion_matrix.sum(1) + confusion_matrix.sum(0) \
                                                       - confusion_matrix.diag() )).mean().item())
        #Add total pixel accuracy
        train_total_acc.append( (confusion_matrix.diag().sum()/confusion_matrix.sum()).item())
        ############CHECK#############
        
        #EVAL
        print('Checking accuracy on validation set')        
        model.eval()
        with torch.no_grad():
            iter_valid_loss = []
            
            #Initiate confusion matrix to calculate accuracy
            confusion_matrix = torch.zeros((n_classes, n_classes), device = device, dtype = torch.float)
            
            for t, (x,y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                
                scores = model(x)
                loss = loss_fcn(scores, y)
                iter_valid_loss.append(loss.item())
                
                pred = scores.max(1)[1]                
                
                #Update confusion matrix. First ignore all the void values                                
                lin_index = pred[y != 255] * n_classes + y[y != 255]
                confusion_matrix += torch.zeros(n_classes,n_classes, device = device).put_(lin_index,\
                                               torch.tensor(1., device = device).expand_as(lin_index), accumulate = True)
                
                if (t % print_every) == 0:
                    print('Validation loss is ' + str(loss.item()))          
            
            valid_loss.append(np.mean(iter_valid_loss))
            #Append accuracy        
            #First convert into float, add a little eps to make sure it doesn't become undefined
            confusion_matrix = confusion_matrix.type(torch.float) + 1e-6
            #Add per class accuracy
            valid_class_iou.append((confusion_matrix.diag()/(confusion_matrix.sum(1) + confusion_matrix.sum(0) -\
                                                      confusion_matrix.diag()) ).cpu().numpy())
            #Add mean IoU
            valid_total_iou.append((confusion_matrix.diag()/(confusion_matrix.sum(1) + confusion_matrix.sum(0) \
                                                       - confusion_matrix.diag() )).mean().item())   
            #Add total accuracy
            valid_total_acc.append((confusion_matrix.diag().sum()/confusion_matrix.sum()).item())                     
        
        epoch_end = time.time()
        epoch_time.append(epoch_end - epoch_start)                
        if not tuning and save_idx:
            if e % 50 == 0 and e > 0:
                state = {
                    'epoch': e,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()#,
                    #'scheduler': scheduler.state_dict()
                }
                
                torch.save(state, folder_dir + '/modelstate.pth')
        
        #Add to metrics to plot
        viz_tracker(viz2, epoch_time_plot, torch.tensor([epoch_time[e]]), torch.tensor([e]) )
        viz_tracker(viz2, loss_plot, torch.tensor([[train_loss[e], valid_loss[e]]]), torch.tensor([[e,e]]))
        viz_tracker(viz2, iou_plot, torch.tensor([[train_total_iou[e], valid_total_iou[e]]]), torch.tensor([[e,e]]))
        viz_tracker(viz2, acc_plot, torch.tensor([[train_total_acc[e], valid_total_acc[e]]]), torch.tensor([[e,e]]))
        viz2.close(win = train_plot)
        
        #Save resulting arrays so far every 10 or so epochs
        if not tuning and save_idx:
            if((e+1) % 10 == 0):
                with open(folder_dir +'/RPSPtrain_loss.pkl', 'wb') as handle:
                    pickle.dump(train_loss, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open(folder_dir +'/RPSPvalid_loss.pkl', 'wb') as handle:
                    pickle.dump(valid_loss, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open(folder_dir +'/RPSPtrain_class_iou.pkl', 'wb') as handle:
                    pickle.dump(train_class_iou, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open(folder_dir +'/RPSPtrain_total_iou.pkl', 'wb') as handle:
                    pickle.dump(train_total_iou, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open(folder_dir +'/RPSPtrain_total_acc.pkl', 'wb') as handle:
                    pickle.dump(train_total_acc, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open(folder_dir +'/RPSPvalid_class_iou.pkl', 'wb') as handle:
                    pickle.dump(valid_class_iou, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open(folder_dir +'/RPSPvalid_total_iou.pkl', 'wb') as handle:
                    pickle.dump(valid_total_iou, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open(folder_dir +'/RPSPvalid_total_acc.pkl', 'wb') as handle:
                    pickle.dump(valid_total_acc, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open(folder_dir +'/RPSPepoch_time.pkl', 'wb') as handle:
                    pickle.dump(epoch_time, handle, protocol = pickle.HIGHEST_PROTOCOL)  
        e += 1

def viz_tracker(viz, plot, value, num):
    viz.line(X = num, Y = value, win = plot, update = 'append')    

if __name__ == '__main__':
    main()