# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:13:09 2019

@author: dkang
"""

import random

import numpy as np
import torch
from torchvision.transforms import functional as F

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class JointTransform(object):
    def __call__(self, img, target):        
        hor_trans = random.choice((True, False))
        
        return( F.normalize(F.to_tensor(F.center_crop(ImageTransform(img, hor_trans),(300, 500))), \
                            [0.45679754, 0.44313163, 0.4082983], [0.23698017, 0.23328756, 0.23898676], False), \
                torch.from_numpy(np.array(F.center_crop(horiz_flip(target, hor_trans), (300,500)), dtype = np.int32 )).long())
                
        
class JointTransformTuning(object):
    def __call__(self, img, target):
        return (F.normalize(F.to_tensor(F.center_crop(img, (300, 500) )), [0.45679754, 0.44313163, 0.4082983], [0.23698017, 0.23328756, 0.23898676], False), \
                        torch.from_numpy(np.array(F.center_crop(target, (300, 500)), dtype = np.int32 )).long())

class JointTransformImageOnly(object):
    def __call__(self, img, target):        
        hor_trans = random.choice((True, False))
        
        return( F.center_crop(ImageTransform(img, hor_trans), (300, 500)), \
                F.center_crop(horiz_flip(target, hor_trans), (300,500)))    
    
def ImageTransform(img, hor_trans):
    brightness = random.gauss(1, .15)
    contrast = random.uniform(.8, 1.3)
    gamma = random.uniform(.7, 1.3)
    saturation = random.uniform(.7, 1.3)
    
    if hor_trans:
        return F.adjust_saturation(F.adjust_gamma(F.adjust_contrast(F.adjust_brightness(F.hflip(img), brightness), contrast), gamma), saturation)
    else:
        return F.adjust_saturation(F.adjust_gamma(F.adjust_contrast(F.adjust_brightness(img, brightness), contrast), gamma), saturation)    

def horiz_flip(img, hor_trans):
    if hor_trans:
        return F.hflip(img)
    else:
        return img