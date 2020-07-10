#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

#from imutils import paths
from pathlib import Path
import os, sys
import time
import copy

import pandas as pd
import matplotlib.pylab as plt
import numpy as np

from itertools import product
from PIL import Image

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo.util import get_model_layers
from lucent.misc.io import show
from lucent.misc.channel_reducer import ChannelReducer
from lucent.misc.io import show

from captum.insights import AttributionVisualizer, Batch
from captum.insights.features import ImageFeature

# Local modules
from cub_tools.train import train_model
from cub_tools.visualize import imshow, visualize_model
from cub_tools.utils import unpickle, save_pickle
from cub_tools.transforms import makeAggresiveTransforms, makeDefaultTransforms, resizeCropTransforms


def baseline_func(input):
    return input * 0

def formatted_data_iter(dataloader):
    dataloader = iter(dataloader)
    while True:
        images, labels = next(dataloader)
        yield Batch(inputs=images, labels=labels)

# Script runtime options
model = 'inceptionv4'
root_dir = '..'
data_root_dir = os.path.join(root_dir, 'data')
model_root_dir = os.path.join(root_dir, 'models')
stages = ['test']


# Paths setup
data_dir = os.path.join(data_root_dir,'images')
output_dir = os.path.join(model_root_dir,'classification/{}'.format(model))
model_history = os.path.join(output_dir,'model_history.pkl')
model_file = os.path.join(output_dir, 'caltech_birds_{}_full.pth'.format(model))


# Get image transforms
# Get data transforms
data_transforms = resizeCropTransforms(img_crop_size=224, img_resize=256)

# Setup data loaders with augmentation transforms
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in stages}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=False, num_workers=2)
              for x in stages}
dataset_sizes = {x: len(image_datasets[x]) for x in stages}
class_names = image_datasets[stages[0]].classes


# Setup the device to run the computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device::', device)


# Load the best model from file
model_ = torch.load(model_file)
_ = model_.to(device).eval()


visualizer = AttributionVisualizer(
    models=[model_],
    score_func=lambda o: torch.nn.functional.softmax(o, 1),
    classes=class_names,
    features=[
        ImageFeature(
            "Photo",
            baseline_transforms=[baseline_func],
            input_transforms=[
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    ],
    dataset=formatted_data_iter(dataloaders['test']),
)


visualizer.serve(port=8600)



