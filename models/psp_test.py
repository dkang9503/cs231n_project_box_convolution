# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:36:11 2019

@author: dkang
"""

import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch
import math


class PSP_head(nn.Module):
    
    def __init__(self, in_channels):
        
        super(PSP_head, self).__init__()
        
        out_channels = int( in_channels / 4 )
#                                   nn.BatchNorm2d(out_channels),        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.ReLU(True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.fusion_bottleneck = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(out_channels),
                                               nn.ReLU(True),
                                               nn.Dropout2d(0.1, False))
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):

        fcn_features_spatial_dim = x.size()[2:]

        pooled_1 = nn.functional.adaptive_avg_pool2d(x, 1)
        pooled_1 = self.conv1(pooled_1)
        pooled_1 = nn.functional.upsample_bilinear(pooled_1, size=fcn_features_spatial_dim)

        pooled_2 = nn.functional.adaptive_avg_pool2d(x, 2)
        pooled_2 = self.conv2(pooled_2)
        pooled_2 = nn.functional.upsample_bilinear(pooled_2, size=fcn_features_spatial_dim)

        pooled_3 = nn.functional.adaptive_avg_pool2d(x, 3)
        pooled_3 = self.conv3(pooled_3)
        pooled_3 = nn.functional.upsample_bilinear(pooled_3, size=fcn_features_spatial_dim)

        pooled_4 = nn.functional.adaptive_avg_pool2d(x, 6)
        pooled_4 = self.conv4(pooled_4)
        pooled_4 = nn.functional.upsample_bilinear(pooled_4, size=fcn_features_spatial_dim)

        x = torch.cat([x, pooled_1, pooled_2, pooled_3, pooled_4],
                       dim=1)

        x = self.fusion_bottleneck(x)

        return x

        
class Resnet50_8s_psp(nn.Module):
    
    def __init__(self, num_classes=21):
        
        super(Resnet50_8s_psp, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet50_8s = resnet50(fully_conv=True,
                                      pretrained=False,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        
        self.psp_head = PSP_head(resnet50_8s.inplanes)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes // 4, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s.conv1(x)
        x = self.resnet50_8s.bn1(x)
        x = self.resnet50_8s.relu(x)
        x = self.resnet50_8s.maxpool(x)

        x = self.resnet50_8s.layer1(x)
        x = self.resnet50_8s.layer2(x)
        x = self.resnet50_8s.layer3(x)
        x = self.resnet50_8s.layer4(x)
        
        x = self.psp_head(x)
        
        x = self.resnet50_8s.fc(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    
    kernel_size = np.asarray((3, 3))
    
    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    
    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2
    
    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 fully_conv=False,
                 remove_avg_pool_layer=False,
                 output_stride=32,
                 additional_blocks=0,
                 multi_grid=(1,1,1) ):
        
        # Add additional variables to track
        # output stride. Necessary to achieve
        # specified output stride.
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1
        
        self.remove_avg_pool_layer = remove_avg_pool_layer
        
        self.inplanes = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
        
        self.additional_blocks = additional_blocks
        
        if additional_blocks == 1:
            
            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
        
        if additional_blocks == 2:
            
            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer6 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
        
        if additional_blocks == 3:
            
            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer6 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer7 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
        
        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
            # In the latest unstable torch 4.0 the tensor.copy_
            # method was changed and doesn't work as it used to be
            #self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    multi_grid=None):
        
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            
            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:
                
                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                
                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride
                
            
            # We don't dilate 1x1 convolution.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        
        dilation = multi_grid[0] * self.current_dilation if multi_grid else self.current_dilation
            
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
            
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            
            dilation = multi_grid[i] * self.current_dilation if multi_grid else self.current_dilation
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.additional_blocks == 1:
            
            x = self.layer5(x)
        
        if self.additional_blocks == 2:
            
            x = self.layer5(x)
            x = self.layer6(x)
        
        if self.additional_blocks == 3:
            
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
        
        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)
        
        if not self.fully_conv:
            x = x.view(x.size(0), -1)
            
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    
    if pretrained:
        
        if model.additional_blocks:
            
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
            
            return model
           
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        
        if model.additional_blocks:
            
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
            
            return model
           
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        
        if model.additional_blocks:
            
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
            
            return model
           
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    
   
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    
    
    if pretrained:
        
        if model.additional_blocks:
            
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
            
            return model
           
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    
   
    return model
    


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model