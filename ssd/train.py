'''
Adapted from https://github.com/amdegroot/ssd.pytorch by Harry Emeric
CS231N Final Project: Exploring Box Convolutional Layers

File: train.py

------------------------------------------------------------------------

This file contains the main script for training the architectures for object
detection.
'''

import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import visdom
import pickle

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from box_ssd2 import build_ssd_box2
from ssd_full_box import build_ssd_full_box
import os
import sys
import time
import torch
from torch.utils.data import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from utils import metrics


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--subset_size', default=1,type=float,
                    help='Subset of dataset to use')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ssd',
                help='model architecture: ' +
                    ' (default: ssd)')
args = parser.parse_args()

if args.visdom:
    viz = visdom.Visdom()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset_train = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
#         if args.dataset_root == COCO_ROOT:
#             parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset_train = VOCDetection(root=args.dataset_root,
                               image_sets = [('2012', 'trainval')],
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
        dataset_val = VOCDetection(root=args.dataset_root,
                               image_sets = [('2007', 'val')],
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

        dataset_test = VOCDetection(root=args.dataset_root,
                               image_sets = [('2007', 'test')],
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
        
    # Take subsample of the datasets
    frac = args.subset_size
    data_sizes = np.array([len(dataset_train), len(dataset_val), len(dataset_test)])
    data_sizes_sub = (data_sizes*frac).astype(int)
    np.random.seed(10)
    data_indices = [np.random.choice(data_sizes[i], data_sizes_sub[i]) for i in range(3)]
    
    
    print("Train/Val/Test split " + str(data_sizes_sub[0]) + ':' + str(data_sizes_sub[1]) + ':' + str(data_sizes_sub[2]))

    if args.arch == 'ssd':
        ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net
    elif args.arch == 'box_ssd':
        ssd_net = build_ssd_box2('train', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net
    elif args.arch == 'full_box_ssd':
        ssd_net = build_ssd_full_box('train', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net
    else:
        raise("Incorrect Architecture chosen")

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

#     if args.resume:
#         print('Resuming training, loading {}...'.format(args.resume))
#         ssd_net.load_weights(args.resume)
#     else:
#         args.basenet == False
#         pass
#         vgg_weights = torch.load(args.save_folder + args.basenet)
#         print('Loading base network...')
#         ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        device = torch.device('cuda')
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    
    train_loss = []
    valid_loss = []
    train_loss_iter = []
    epoch_time = []

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    loss_val = 0
    print('Loading the dataset...')

    epoch_size = len(dataset_train) // (args.batch_size / frac)
    print('Training ' + args.arch + ' on:', dataset_train.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = args.arch+' on ' + dataset_train.name + ' LR=' \
                    + str(args.lr) + ' WD=' + str(args.weight_decay)
        vis_legend_train = ['Loc Loss Train', 'Conf Loss Train', 'Total Loss Train']
        vis_legend_trainval = ['Train Loss', 'Val Loss']
        
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend_train)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend_train)
        
        loss_plot = viz.line(Y= torch.tensor([[0,0]]).zero_(), opts = dict(title = 'Loss Tracker', 
                         legend = ['Training Loss', 'Validation Loss'],
                         xlabel = 'Iteration',
                         ylabel = 'Loss',
                         show_legend = True))

    data_loader_train = data.DataLoader(dataset_train, args.batch_size,
                                  num_workers=args.num_workers,
                                  sampler=SubsetRandomSampler(data_indices[0]),
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
    data_loader_val = data.DataLoader(dataset_val, args.batch_size,
                                  num_workers=args.num_workers,
                                  sampler=SubsetRandomSampler(data_indices[1]),
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
#     data_loader_test = data.DataLoader(dataset_test, args.batch_size,
#                                   num_workers=args.num_workers,
#                                   sampler=SubsetRandomSampler(data_indices[2]),
#                                   shuffle=False, collate_fn=detection_collate,
#                                   pin_memory=True)
    
#     import pdb; pdb.set_trace()
#     mean = 0
#     count = 0
#     for t, (img,y) in enumerate(data_loader_train):
#             mean += img.mean([0,2,3])
#             print(img.mean([0,2,3]))
#             count += 1
#             if t % 10 == 0:
#                 print(mean/count)
#     mean = mean/count
#     print(mean)
#     breakpoint()
   
    
    # create batch iterator
    batch_iterator = iter(data_loader_train)
    for iteration in range(args.start_iter, cfg['max_iter']//10): # //10 here to end early
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1
            print("Training Epoch number " + str(epoch))
            epoch_start = time.time()
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)

            
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader_train)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]
#             images = Variable(images.cuda())
#             with torch.no_grad():
#                 targets = [Variable(ann.cuda()) for ann in targets]
#         else:
#             images = Variable(images)
#             with torch.no_grad():
#                 targets = [Variable(ann) for ann in targets]
        # forward
        t0 = time.time()
        
        out = net(images)
        # backprop
        optimizer.zero_grad()
        
        
        
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        
        train_loss_iter.append(loss.item())

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Training Loss: %.4f ||' % (loss.item()), end=' ')

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')

#         if iteration != 0 and iteration % 5000 == 0:
#             print('Saving state, iter:', iteration)
#             torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
#                        repr(iteration) + '.pth')

    
    # calculate val loss
        #import pdb; pdb.set_trace()
        
        
        if iteration != 0 and (iteration % epoch_size == 0):
            net.eval()
            loss_val = 0
            with torch.no_grad():
                for t, (images, targets) in enumerate(data_loader_val):
                    if args.cuda:
                        images = images.to(device)
                        targets = [ann.to(device) for ann in targets]
#                         images = images.cuda()
                    
#                         targets = [Variable(ann.cuda()) for ann in targets]
                
                    out = net(images)
                    loss_l, loss_c = criterion(out, targets)
                    loss_val += loss_l + loss_c
            loss_val /= len(data_loader_val)
        
#             loc_loss += loss_l.item()
#             conf_loss += loss_c.item()
            
            print('iter ' + repr(iteration) + ' || Val Loss: %.4f ||' % (loss_val.item()), end=' ')
            viz_tracker(loss_plot, torch.tensor([[loss.item(), loss_val.item()]]), torch.tensor([[iteration-1,iteration-1]]))
                #epoch += 1

                # reset epoch loss counters
            loss_l = 0
            loss_c = 0
            
            train_loss.append(loss.item())
            valid_loss.append(loss_val.item())
            
           #Epoch time
            epoch_end = time.time()
            epoch_time.append(epoch_end - epoch_start)
            print("Epoch " + str(epoch) + " took " + str(int(epoch_end - epoch_start)) + "secs to train")
            
            suffix = args.arch + '_lr'+ str(args.lr) + '_wd' + str(args.weight_decay) \
                          + '_sub'+ str(args.subset_size)
            #Save resulting arrays so far every 10 epochs
            if((epoch) % 1 == 0):
                with open('pkl_files/'+ suffix + 'train_loss.pkl', 'wb') as handle:
                    pickle.dump(train_loss, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open('pkl_files/'+ suffix +'valid_loss.pkl', 'wb') as handle:
                    pickle.dump(valid_loss, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open('pkl_files/'+ suffix + 'epoch_time.pkl', 'wb') as handle:
                    pickle.dump(epoch_time, handle, protocol = pickle.HIGHEST_PROTOCOL)
                with open('pkl_files/'+ suffix +'train_loss_iter.pkl', 'wb') as handle:
                    pickle.dump(train_loss_iter, handle, protocol = pickle.HIGHEST_PROTOCOL)
                

                state = {
                    'epoch': epoch,
                    'state_dict': ssd_net.state_dict(),
                    'optimizer': optimizer.state_dict()
                        }        
    
                torch.save(state, args.save_folder + '' + str(args.subset_size) +args.dataset + suffix + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )
        
def viz_tracker(plot, value, num):
        viz.line(X = num, Y = value, win = plot, update = 'append')        

if __name__ == '__main__':
    train()
