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

import pandas as pd
import matplotlib.pylab as plt
import numpy as np

def make_predictions(model, dataloaders, device):
    '''
    Function to take the test validation set of images, and produce predictions.
    
    Returns two numpy arrays of truth labels and prediction labels.
    '''
    was_training = model.training
    model.eval()

    print('Commencing predictions minibatch..', end='')
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            if i % 25 == 0:
                print(i,'..', end='')

            inputs = inputs.to(device)
            labels = labels.to(device)
            

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if i == 0:
                labels_truth = labels.cpu().numpy()
                labels_pred = preds.cpu().numpy()
            else:
                labels_truth = np.concatenate((labels_truth,labels.cpu().numpy()))
                labels_pred = np.concatenate((labels_pred,preds.cpu().numpy()))

    print('Complete.')

    return labels_truth, labels_pred