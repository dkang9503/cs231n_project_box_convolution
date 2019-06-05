# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:39:54 2019

@author: dkang
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:46:07 2019

@author: dkang
"""

import torch.nn as nn
from box_convolution import BoxConv2d

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding =1"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BoxBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, num_boxes, max_input_h, max_input_w, downsample = None,
                 groups =1, base_width = 64, dilation =1, norm_layer = None):
        super(BoxBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
    
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width//4)
        self.bn1 = norm_layer(width//4)
        self.conv2 = BoxConv2d(width//4, num_boxes, max_input_h, max_input_w, reparametrization_factor = 1.5625)
        self.bn2 = nn.BatchNorm2d(width//4 * num_boxes)
        self.conv3 = conv1x1(width//4 * num_boxes, planes * self.expansion )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample        

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        return out

class BoxResNet(nn.Module):
    def __init__(self, layers, num_classes, width, height, zero_init_residual = False,
                 groups = 1, width_per_group = 64, replace_stride_with_dilation = None,
                 norm_layer = None):
        super(BoxResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 64 #number of filters
        self.width = width
        self.height = height #width of picture
        self.dilation = 1 #convolutions with dilation
        
        if replace_stride_with_dilation is None:            
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        #input 3x64x64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding =3,
                                bias = False)
        #(64 - 7 + 6)/ 2 + 1 = 32.5 = 32
        #Output is 64x32x32
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #(32 - 3 + 2)/2 + 1 = 16.5 = 16
        #Output is 64x16x16
        self.block = BoxBottleneck
        
        #_make_layer(self, BottleNeck, 64, 3, stride=1, dilate=False):
        self.layer1 = self._make_layer(64, layers[0]) #3
        self.layer2 = self._make_layer(128, layers[1], stride=2, #4
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, layers[2], stride=2, #6
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, layers[3], stride=2, #3
                                       dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        
        #Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                    
    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * 4: #64 \neq 64* 4, so TRUE
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * 4, stride), # receives 64, makes output 64*4, stride = 1
                norm_layer(planes * 4),
            )            
    
        if stride != 1:
            self.height = self.width //2
            self.width = self.width //2
    
        layers = []
        temp = Bottleneck
        #Bottleneck(64, 64, stide = 1, downsample,  )
        layers.append(temp(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * 4 #now self.inplanes = 256
        
        #For the rest of the blocks, blocks = 3

        for _ in range(1, blocks):
            #(self, inplanes, planes, num_boxes, max_input_h, max_input_w, downsample = None,
             #groups =1, base_width = 64, dilation =1, norm_layer = None):
            layers.append(self.block(self.inplanes, planes, num_boxes = 4, 
                                        max_input_h = self.height //2, max_input_w = self.width //2,
                                        norm_layer = norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #Input dimension is 3x64x64
        x = self.conv1(x) #64 output filters, filter size is 7, stride is 2, padding is 3
        #output is (64-(7-1)+(2*3)- 1)/2 +1 = 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

def boxresnet50(num_classes = 200, width = 64, height = 64, **kwargs):
    model = BoxResNet([3, 4, 6, 3], num_classes, width, height, **kwargs)
    return model
