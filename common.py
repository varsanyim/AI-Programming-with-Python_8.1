import argparse 
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import os
import datetime

torch.backends.cudnn.benchmark = True

print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 

TRAIN = "train"
TEST  = "test"
VALID = "valid"

BATCH_SIZE=32
WORKERS=2

SIZE=224 # default: 224, small: 128, verysmall: 64 
t_mean = [0.485, 0.456, 0.406]
t_std = [0.229, 0.224, 0.225]

def getModel(arch):
    if arch == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model


def save_checkpoint(model, class_to_idx, filename, arch='vgg16_bn'):
    print(">> Save checkpoint")    
    dir=os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    model.class_to_idx = class_to_idx
    checkpoint = {
        'arch': arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, filename)
