#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

from imutils import paths
from pathlib import Path
import os
import time
import copy
import pickle

import pandas as pd
import matplotlib.pylab as plt
import numpy as np

# Local modules
from cub_tools.train import train_model
from cub_tools.visualize import imshow, visualize_model
from cub_tools.utils import save_model_dict, save_model_full
from cub_tools.transforms import makeDefaultTransforms

#################################################
# Script runtime options
model_name = 'resnet152'
model_func = models.resnet152
base_dir = '..'
root_dir = os.path.join(base_dir,'data')
data_dir = os.path.join(root_dir,'images')
working_dir = os.path.join(base_dir, 'models/classification', model_name)
batch_size = 8
num_workers = 4
num_epochs = 40
#################################################

os.makedirs(working_dir, exist_ok=True)

# Get data transforms
data_transforms = makeDefaultTransforms()

# Setup data loaders with augmentation transforms
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

print('Number of data')
print('========================================')
for dataset in dataset_sizes.keys():
    print(dataset,' size:: ', dataset_sizes[dataset],' images')

print('')
print('Number of classes:: ', len(class_names))
print('========================================')
for i_class, class_name in enumerate(class_names):
    print(i_class,':: ',class_name)


# Setup the device to run the computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device::', device)


# Setup the model and optimiser

model_ft = model_func(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1.0e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Decay LR by a factor of 0.1 every 7 epochs
lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, 
                                              patience=5, threshold=0.0001, threshold_mode='rel', 
                                              cooldown=0, min_lr=0, eps=1e-08, verbose=False)

# Train the model
model_ft, history = train_model(model=model_ft, criterion=criterion, optimizer=optimizer_ft, scheduler=lr_scheduler, 
                                device=device, dataloaders=dataloaders, dataset_sizes=dataset_sizes, num_epochs=num_epochs,
                                return_history=True, log_history=True, working_dir=working_dir, 
                                log_history_fname='caltech_birds_{}_history'.format(model_name+'_adam')
                               )

# Save out the best model and finish
save_model_full(model=model_ft, PATH=os.path.join(working_dir,'caltech_birds_{}_full.pth'.format(model_name+'_adam')))
save_model_dict(model=model_ft, PATH=os.path.join(working_dir,'caltech_birds_{}_dict.pth'.format(model_name+'_adam')))