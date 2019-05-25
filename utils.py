'''
Author: Harry Emeric
CS231N Final Project: Exploring Box Convolutional Layers

File: utils.py

------------------------------------------------------------------------

This file contains various utility functions useful for this project.

Reference https://github.com/pytorch/examples/tree/master/imagenet

'''

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count

'''
To manage arguments off the main training scripts.
'''
def getParser():
    import argparse # Bad style - how to avoid this?
    
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', 
                    default='../Code/data/ImageNet/tiny-imagenet-200/train',
                help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                help='model architecture: ' +
                    ' (default: resnet18)')
#     parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                 help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                help='number of total epochs to run')
#     parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                 help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                metavar='N',
                help='mini-batch size (default: 256), this is the total '
                     'batch size of all GPUs on the current node when '
                     'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                metavar='W', help='weight decay (default: 1e-4)',
                dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                metavar='N', help='print frequency (default: 100)')
#     parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                 help='path to latest checkpoint (default: none)')
#     parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                 help='evaluate model on validation set')
#     parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                 help='use pre-trained model')
#     parser.add_argument('--world-size', default=-1, type=int,
#                 help='number of nodes for distributed training')
#     parser.add_argument('--rank', default=-1, type=int,
#                 help='node rank for distributed training')
#     parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                 help='url used to set up distributed training')
#     parser.add_argument('--dist-backend', default='nccl', type=str,
#                 help='distributed backend')
#     parser.add_argument('--seed', default=None, type=int,
#                 help='seed for initializing training. ')
#     parser.add_argument('--gpu', default=None, type=int,
#                 help='GPU id to use.')
#     parser.add_argument('--multiprocessing-distributed', action='store_true',
#                 help='Use multi-processing distributed training to launch '
#                      'N processes per node, which has N GPUs. This is the '
#                      'fastest way to use PyTorch for either single node or '
#                      'multi node data parallel training')
    return parser


'''
Script to move ImageNet tiny into correct directories for each class

https://github.com/DennisHanyuanXu/Tiny-ImageNet/blob/master/src/data_prep.py
'''

def create_val_img_folder():
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    dataset_dir = '/home/dkang/Project/Code/data/ImageNet/tiny-imagenet-200/' 
            #os.path.join(args.data_dir, args.dataset)
        
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


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
