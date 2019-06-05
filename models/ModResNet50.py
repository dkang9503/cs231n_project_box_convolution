# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:19:43 2019

@author: dkang
"""

import torch.nn as nn
from box_convolution import BoxConv2d


class Bottleneck2(nn.Module):
    def __init__(self, input_dim, conv1_numf, conv2_numf, conv3_numf, conv2_stride, downsample = None):
        super(Bottleneck2, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dim, conv1_numf, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(conv1_numf)
        self.conv2 = nn.Conv2d(conv1_numf, conv2_numf, kernel_size = 3, stride = conv2_stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(conv2_numf)
        self.conv3 = nn.Conv2d(conv2_numf, conv3_numf, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn3 = nn.BatchNorm2d(conv3_numf)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x) #1x1 convolution with stride 1 with output channels width (64)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) #3x3 with stride 1
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) #1x1 with output channels planes* self.expansion
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return(out)
        

class BoxBottleneck2(nn.Module):
    def __init__(self, input_dim, conv1_numf, conv2_numb, conv3_numf, conv2_stride, max_w, max_h, downsample = None):
        super(BoxBottleneck2, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dim, conv1_numf, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(conv1_numf)
        self.conv2 = BoxConv2d(conv1_numf, conv2_numb, max_w, max_h, 1.5625)
        self.bn2 = nn.BatchNorm2d(conv2_numb * conv1_numf)
        self.conv3 = nn.Conv2d(conv2_numb * conv1_numf, conv3_numf, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn3 = nn.BatchNorm2d(conv3_numf)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x) #1x1 convolution with stride 1 with output channels width (64)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) #3x3 with stride 1
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) #1x1 with output channels planes* self.expansion
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return(out)

class ModResNet50(nn.Module):
    def __init__(self, input_dim = 64, num_classes = 200):
        super(ModResNet50, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        
        self.inplanes = input_dim #number of filters
        self.dilation = 1
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.downsample1 = nn.Sequential(
                nn.Conv2d(self.inplanes, 256, kernel_size = 1, stride = 1, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(256),
            )
        
        self.downsample2 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(512),
            )
        
        self.downsample3 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(1024),
            )
        
        self.downsample4 = nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(2048),
            )
        
        self.layer1 = nn.Sequential(
                Bottleneck2(input_dim = 64, conv1_numf = 64, conv2_numf = 64, conv3_numf = 256, conv2_stride = 1, downsample = self.downsample1),
                Bottleneck2(input_dim = 256, conv1_numf = 64, conv2_numf = 64, conv3_numf = 256, conv2_stride = 1),
                Bottleneck2(input_dim = 256, conv1_numf = 64, conv2_numf = 64, conv3_numf = 256, conv2_stride = 1)
                )
        
        self.layer2 = nn.Sequential(
                Bottleneck2(input_dim = 256, conv1_numf = 64, conv2_numf = 64, conv3_numf = 512, conv2_stride = 2, downsample = self.downsample2),
                Bottleneck2(input_dim = 512, conv1_numf = 64, conv2_numf = 64, conv3_numf = 512, conv2_stride = 1),
                Bottleneck2(input_dim = 512, conv1_numf = 64, conv2_numf = 64, conv3_numf = 512, conv2_stride = 1),
                Bottleneck2(input_dim = 512, conv1_numf = 64, conv2_numf = 64, conv3_numf = 512, conv2_stride = 1)
                )
        
        self.layer3 = nn.Sequential(
                Bottleneck2(input_dim = 512, conv1_numf = 64, conv2_numf = 64, conv3_numf = 1024, conv2_stride = 2, downsample = self.downsample3),
                Bottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numf = 64, conv3_numf = 1024, conv2_stride = 1),
                Bottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numf = 64, conv3_numf = 1024, conv2_stride = 1),
                Bottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numf = 64, conv3_numf = 1024, conv2_stride = 1),
                Bottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numf = 64, conv3_numf = 1024, conv2_stride = 1),
                Bottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numf = 64, conv3_numf = 1024, conv2_stride = 1)
                )
        
        self.layer4 = nn.Sequential(
                Bottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numf = 64, conv3_numf = 2048, conv2_stride = 2, downsample = self.downsample4),
                Bottleneck2(input_dim = 2048, conv1_numf = 64, conv2_numf = 64, conv3_numf = 2048, conv2_stride = 1),
                Bottleneck2(input_dim = 2048, conv1_numf = 64, conv2_numf = 64, conv3_numf = 2048, conv2_stride = 1)                
                )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        
        return x
    
class BoxResNet50(nn.Module):
    def __init__(self, input_dim = 64, num_classes = 200):
        super(BoxResNet50, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        
        self.inplanes = input_dim #number of filters
        self.dilation = 1
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.downsample1 = nn.Sequential(
                nn.Conv2d(self.inplanes, 256, kernel_size = 1, stride = 1, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(256),
            )
        
        self.downsample2 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(512),
            )
        
        self.downsample3 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(1024),
            )
        
        self.downsample4 = nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(2048),
            )
        
        self.layer1 = nn.Sequential(
                Bottleneck2(input_dim = 64, conv1_numf = 64, conv2_numf = 64, conv3_numf = 256, conv2_stride = 1, downsample = self.downsample1),
                BoxBottleneck2(input_dim = 256, conv1_numf = 51, conv2_numb = 5, conv3_numf = 256, conv2_stride = 1, max_w = 16, max_h = 16),
                BoxBottleneck2(input_dim = 256, conv1_numf = 51, conv2_numb = 5, conv3_numf = 256, conv2_stride = 1, max_w = 16, max_h = 16)
                )
        
        self.layer2 = nn.Sequential(
                Bottleneck2(input_dim = 256, conv1_numf = 64, conv2_numf = 64, conv3_numf = 512, conv2_stride = 2, downsample = self.downsample2),
                BoxBottleneck2(input_dim = 512, conv1_numf = 51, conv2_numb = 5, conv3_numf = 512, conv2_stride = 1, max_w = 8, max_h = 8),
                BoxBottleneck2(input_dim = 512, conv1_numf = 51, conv2_numb = 5, conv3_numf = 512, conv2_stride = 1, max_w = 8, max_h = 8),
                BoxBottleneck2(input_dim = 512, conv1_numf = 51, conv2_numb = 5, conv3_numf = 512, conv2_stride = 1, max_w = 8, max_h = 8)
                )
        
        self.layer3 = nn.Sequential(
                Bottleneck2(input_dim = 512, conv1_numf = 64, conv2_numf = 64, conv3_numf = 1024, conv2_stride = 2, downsample = self.downsample3),
                BoxBottleneck2(input_dim = 1024, conv1_numf = 51, conv2_numb = 5, conv3_numf = 1024, conv2_stride = 1, max_w = 4, max_h = 4),
                BoxBottleneck2(input_dim = 1024, conv1_numf = 51, conv2_numb = 5, conv3_numf = 1024, conv2_stride = 1, max_w = 4, max_h = 4),
                BoxBottleneck2(input_dim = 1024, conv1_numf = 51, conv2_numb = 5, conv3_numf = 1024, conv2_stride = 1, max_w = 4, max_h = 4),
                BoxBottleneck2(input_dim = 1024, conv1_numf = 51, conv2_numb = 5, conv3_numf = 1024, conv2_stride = 1, max_w = 4, max_h = 4),
                BoxBottleneck2(input_dim = 1024, conv1_numf = 51, conv2_numb = 5, conv3_numf = 1024, conv2_stride = 1, max_w = 4, max_h = 4)
                )
        
        self.layer4 = nn.Sequential(
                Bottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numf = 64, conv3_numf = 2048, conv2_stride = 2, downsample = self.downsample4),
                BoxBottleneck2(input_dim = 2048, conv1_numf = 51, conv2_numb = 5, conv3_numf = 2048, conv2_stride = 1, max_w = 2, max_h = 2),
                BoxBottleneck2(input_dim = 2048, conv1_numf = 51, conv2_numb = 5, conv3_numf = 2048, conv2_stride = 1, max_w = 2, max_h = 2)
                )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        
        return x
    
class ModResNet502(nn.Module):
    def __init__(self, input_dim = 64, num_classes = 200):
        super(ModResNet502, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        
        self.inplanes = input_dim #number of filters
        self.dilation = 1
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.downsample1 = nn.Sequential(
                nn.Conv2d(self.inplanes, 256, kernel_size = 1, stride = 1, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(256),
            )
        
        self.downsample2 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(512),
            )
        
        self.downsample3 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(1024),
            )
        
        self.downsample4 = nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(2048),
            )
        
        self.layer1 = nn.Sequential(
                Bottleneck2(input_dim = 64, conv1_numf = 64, conv2_numf = 64, conv3_numf = 256, conv2_stride = 1, downsample = self.downsample1),
                Bottleneck2(input_dim = 256, conv1_numf = 64, conv2_numf = 64, conv3_numf = 256, conv2_stride = 1),
                Bottleneck2(input_dim = 256, conv1_numf = 64, conv2_numf = 64, conv3_numf = 256, conv2_stride = 1)
                )
        
        self.layer2 = nn.Sequential(
                Bottleneck2(input_dim = 256, conv1_numf = 128, conv2_numf = 128, conv3_numf = 512, conv2_stride = 2, downsample = self.downsample2),
                Bottleneck2(input_dim = 512, conv1_numf = 128, conv2_numf = 128, conv3_numf = 512, conv2_stride = 1),
                Bottleneck2(input_dim = 512, conv1_numf = 128, conv2_numf = 128, conv3_numf = 512, conv2_stride = 1),
                Bottleneck2(input_dim = 512, conv1_numf = 128, conv2_numf = 128, conv3_numf = 512, conv2_stride = 1)
                )
        
        self.layer3 = nn.Sequential(
                Bottleneck2(input_dim = 512, conv1_numf = 256, conv2_numf = 256, conv3_numf = 1024, conv2_stride = 2, downsample = self.downsample3),
                Bottleneck2(input_dim = 1024, conv1_numf = 256, conv2_numf = 256, conv3_numf = 1024, conv2_stride = 1),
                Bottleneck2(input_dim = 1024, conv1_numf = 256, conv2_numf = 256, conv3_numf = 1024, conv2_stride = 1),
                Bottleneck2(input_dim = 1024, conv1_numf = 256, conv2_numf = 256, conv3_numf = 1024, conv2_stride = 1),
                Bottleneck2(input_dim = 1024, conv1_numf = 256, conv2_numf = 256, conv3_numf = 1024, conv2_stride = 1),
                Bottleneck2(input_dim = 1024, conv1_numf = 256, conv2_numf = 256, conv3_numf = 1024, conv2_stride = 1)
                )
        
        self.layer4 = nn.Sequential(
                Bottleneck2(input_dim = 1024, conv1_numf = 512, conv2_numf = 512, conv3_numf = 2048, conv2_stride = 2, downsample = self.downsample4),
                Bottleneck2(input_dim = 2048, conv1_numf = 512, conv2_numf = 512, conv3_numf = 2048, conv2_stride = 1),
                Bottleneck2(input_dim = 2048, conv1_numf = 512, conv2_numf = 512, conv3_numf = 2048, conv2_stride = 1)                
                )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        
        return x
    
class BoxResNet502(nn.Module):
    def __init__(self, input_dim = 64, num_classes = 200):
        super(BoxResNet502, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        
        self.inplanes = input_dim #number of filters
        self.dilation = 1
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.downsample1 = nn.Sequential(
                nn.Conv2d(self.inplanes, 256, kernel_size = 1, stride = 1, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(256),
            )
        
        self.downsample2 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(512),
            )
        
        self.downsample3 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(1024),
            )
        
        self.downsample4 = nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size = 1, stride = 2, padding = 0, bias = False), # receives 64, makes output 64*4, stride = 1
                nn.BatchNorm2d(2048),
            )
        
        self.layer1 = nn.Sequential(
                Bottleneck2(input_dim = 64, conv1_numf = 64, conv2_numf = 64, conv3_numf = 256, conv2_stride = 1, downsample = self.downsample1),
                BoxBottleneck2(input_dim = 256, conv1_numf = 16, conv2_numb = 4, conv3_numf = 256, conv2_stride = 1, max_w = 16, max_h = 16),
                BoxBottleneck2(input_dim = 256, conv1_numf = 16, conv2_numb = 4, conv3_numf = 256, conv2_stride = 1, max_w = 16, max_h = 16)
                )
        
        self.layer2 = nn.Sequential(
                Bottleneck2(input_dim = 256, conv1_numf = 128, conv2_numf = 128, conv3_numf = 512, conv2_stride = 2, downsample = self.downsample2),
                BoxBottleneck2(input_dim = 512, conv1_numf = 32, conv2_numb = 4, conv3_numf = 512, conv2_stride = 1, max_w = 8, max_h = 8),
                BoxBottleneck2(input_dim = 512, conv1_numf = 32, conv2_numb = 4, conv3_numf = 512, conv2_stride = 1, max_w = 8, max_h = 8),
                BoxBottleneck2(input_dim = 512, conv1_numf = 32, conv2_numb = 4, conv3_numf = 512, conv2_stride = 1, max_w = 8, max_h = 8)
                )
        
        self.layer3 = nn.Sequential(
                Bottleneck2(input_dim = 512, conv1_numf = 256, conv2_numf = 256, conv3_numf = 1024, conv2_stride = 2, downsample = self.downsample3),
                BoxBottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numb = 4, conv3_numf = 1024, conv2_stride = 1, max_w = 4, max_h = 4),
                BoxBottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numb = 4, conv3_numf = 1024, conv2_stride = 1, max_w = 4, max_h = 4),
                BoxBottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numb = 4, conv3_numf = 1024, conv2_stride = 1, max_w = 4, max_h = 4),
                BoxBottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numb = 4, conv3_numf = 1024, conv2_stride = 1, max_w = 4, max_h = 4),
                BoxBottleneck2(input_dim = 1024, conv1_numf = 64, conv2_numb = 4, conv3_numf = 1024, conv2_stride = 1, max_w = 4, max_h = 4)
                )
        
        self.layer4 = nn.Sequential(
                Bottleneck2(input_dim = 1024, conv1_numf = 512, conv2_numf = 512, conv3_numf = 2048, conv2_stride = 2, downsample = self.downsample4),
                BoxBottleneck2(input_dim = 2048, conv1_numf = 128, conv2_numb = 4, conv3_numf = 2048, conv2_stride = 1, max_w = 2, max_h = 2),
                BoxBottleneck2(input_dim = 2048, conv1_numf = 128, conv2_numb = 4, conv3_numf = 2048, conv2_stride = 1, max_w = 2, max_h = 2)
                )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        
        return x