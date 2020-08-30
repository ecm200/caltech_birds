from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

import pandas as pd
import matplotlib.pylab as plt
import numpy as np

import math

def lr_rate_inc_scheduler(optimizer, lr_batches, lr_epochs=2, lr_range=(1.0E-05, 1.0)):
    '''
    Function produces PyTorch Learning Rate Scheduler to increase the learning rate with mini-batch.
    
    This is used to find optimal learning rate for the provided network to be trained.
    
    '''
    
    lr_lambda = lambda x: math.exp(x * math.log(lr_range[1] / lr_range[0]) / (lr_epochs * lr_batches))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    
def learning_rate_optimizer(model, criterion, optimizer, device, dataloaders, 
                            lr_range=(1.0E-05, 1.0), num_epochs=2, smoothing=0.05, 
                            verbose=False, print_interval=50):
    '''
    Function to perform optimal learning rate investigation for a given network.
    
    The function increases the learning rate after each mini-batch over a given range and computes the loss.
    
    The function then returns the loss with the specified learning rate.
    
    '''
    
    lr_loss = []
    lr = []
    
    # Setup the increasing learning rate with batch scheduler
    scheduler = lr_rate_inc_scheduler(optimizer, len(dataloaders['train']), num_epochs, lr_range)

    iter = 0
    
    # Outer loop over epochs
    for i in range(num_epochs):
        print("epoch {}".format(i))
        
        # Inner loop over mini-batches
        for inputs, labels in dataloaders["train"]:

            # Send to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Training mode and zero gradients
            model.train()
            optimizer.zero_grad()

            # Get outputs to calc loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update LR
            scheduler.step()
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lr.append(lr_step)

            # smooth the loss
            if smoothing > 0.0:
                if iter==0:
                    lr_loss.append(loss)
                else:
                    loss = smoothing  * loss + (1 - smoothing) * lr_loss[-1]
                    lr_loss.append(loss)
            # Don't smooth the output function
            else:
                lr_loss.append(loss)
            
            if verbose:
                if iter % print_interval == 0:
                    print('[ITER] Epoch: {} :: Batch: {} :: LR: {} :: Loss {}'.format(i, iter, lr_step, loss))

            iter += 1
    
    return lr, lr_loss