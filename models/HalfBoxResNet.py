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
import torch.nn.functional as F


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding =1"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BottleneckBoxConv(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, num_boxes, max_input_h, max_input_w, dropout_prob=0.0):
        super().__init__()
        assert in_channels % num_boxes == 0
        bt_channels = in_channels // num_boxes # bottleneck channels

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1,1), bias=False),
            nn.BatchNorm2d(bt_channels),
            nn.ReLU(True),
            
            # BEHOLD:
            BoxConv2d(
                bt_channels, num_boxes, max_input_h, max_input_w,
                reparametrization_factor=1.5625),

            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout_prob))

    def forward(self, x):
        return F.relu(x + self.main_branch(x), inplace=True)


class BasicBlock(nn.Module):
    expansion = 1
    
    #Inplanes is the input size of the image
    #Planes is the output size of both convolution layers

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #x.shape = 3x64x64
        identity = x

        out = self.conv1(x) # 3x3 convolution with stride = stride (default = 1), padding = 1
        #(64-3 + 2)/1 + 1 = 64
        #Output size is planes x 64 x 64
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3 convolution with stride = 1
        #(64-3 + 2)/1 +1 = 64
        #Output size is planes x 64 x 64
        out = self.bn2(out)

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


class HalfBoxResNet(nn.Module):
    def __init__(self, layers, num_classes = 200, zero_init_residual = False,
                 groups = 1, width_per_group = 64, replace_stride_with_dilation = None,
                 norm_layer = None):
        super(HalfBoxResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 64 #number of filters
        self.dilation = 1 #convolutions with dilation
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
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
        
        #_make_layer(self, BottleNeck, 64, 3, stride=1, dilate=False):
        self.layer1 = self._make_layer2(64, layers[0]) #3
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

        layers = []
        #Bottleneck(64, 64, stide = 1, downsample,  )
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * 4 #now self.inplanes = 256
        
        #For the rest of the blocks, blocks = 3
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    #self.inplanes = 64
    #_make_layer(self, BottleNeck, 64, 3, stride=1, dilate=False):
    def _make_layer2(self, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        max_input_h = 64
        max_input_w = 64
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * BottleneckBoxConv.expansion: #64 \neq 64* 4, so TRUE
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BottleneckBoxConv.expansion, stride), # receives 64, makes output 64*4, stride = 1
                norm_layer(planes * BottleneckBoxConv.expansion),
        )

        layers = []
        #Bottleneck(64, 64, stide = 1, downsample,  )
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * Bottleneck.expansion #now self.inplanes = 256
        
        #For the rest of the blocks, blocks = 3
        for i in range(1, blocks):
            if ((i+1) % 2) == 0:
                layers.append(BottleneckBoxConv(self.inplanes, planes, max_input_h, max_input_w, dropout_prob = 0.15 ))
            else:
                layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)
        #result is 
        #self.layer1 = [
        #   Bottleneck(self.inplanes = 64, planes = 64, stride = 1, downsample)    
        #   Bottleneck(self.inplanes = 256, planes = 64, stride = 1)
        #   Bottleneck(self.inplanes = 256, planes = 64, stride = 1)
        #]
        #self.layer2 = [
        #   Bottleneck(self.inplanes = 256, planes = 128, stride = 2, downsample)    256 because we changed self.inplanes, plane sfrom self.layer2
        #   Bottleneck(self.inplanes = 512, planes = 128, stride = 1, downsample)    
        #   Bottleneck(self.inplanes = 512, planes = 128, stride = 1, downsample)    
        #   Bottleneck(self.inplanes = 512, planes = 128, stride = 1, downsample)    
        #]            
    def _make_layer3(self, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        max_input_h = 64
        max_input_w = 64
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * BottleneckBoxConv.expansion: #64 \neq 64* 4, so TRUE
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BottleneckBoxConv.expansion, stride), # receives 64, makes output 64*4, stride = 1
                norm_layer(planes * BottleneckBoxConv.expansion),
        )

        layers = []
        #Bottleneck(64, 64, stide = 1, downsample,  )
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * Bottleneck.expansion #now self.inplanes = 256
        
        #For the rest of the blocks, blocks = 3
        for i in range(1, blocks):
            if (i == 1):
                layers.append(BottleneckBoxConv(self.inplanes, planes, max_input_h, max_input_w, dropout_prob = 0.15 ))
            else:
                layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

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

class ResNet(nn.Module):
    #_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress)
    # model = ResNet(inplanes, planes, **kwargs)
    # layers = Bottleneck, layers = [3, 4, 6, 3]
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64 #number of filters
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        #input 3x64x64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #(64 - 7 + 6)/ 2 + 1 = 32.5 = 32
        #Output is 64x32x32
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #(32 - 3 + 2)/2 + 1 = 16.5 = 16
        #Output is 64x16x16
        
        #_make_layer(self, BottleNeck, 64, 3, stride=1, dilate=False):
        self.layer1 = self._make_layer(block, 64, layers[0]) #3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, #4
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, #6
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, #3
                                       dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    #self.inplanes = 64
    #_make_layer(self, BottleNeck, 64, 3, stride=1, dilate=False):
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: #64 \neq 64* 4, so TRUE
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), # receives 64, makes output 64*4, stride = 1
                norm_layer(planes * block.expansion),
            )

        layers = []
        #Bottleneck(64, 64, stide = 1, downsample,  )
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion #now self.inplanes = 256
        
        #For the rest of the blocks, blocks = 3
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
        #result is 
        #self.layer1 = [
        #   Bottleneck(self.inplanes = 64, planes = 64 stride = 1, downsample)    
        #   Bottleneck(self.inplanes = 256, planes = 64, stride = 1)
        #   Bottleneck(self.inplanes = 256, planes = 64, stride = 1)
        #]

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


def _resnet(arch, inplanes, planes, pretrained, progress, **kwargs):
    model = ResNet(inplanes, planes, **kwargs)
    return model

def _halfboxresnet(planes, pretrained, progress, **kwargs):
    model = HalfBoxResNet(planes, **kwargs)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

def resnetHalfBox(pretrained = False, progress = True, **kwargs):
    return _halfboxresnet([3, 4, 6, 3], pretrained, progress, **kwargs)

def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained=False, progress=True, **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained=False, progress=True, **kwargs)