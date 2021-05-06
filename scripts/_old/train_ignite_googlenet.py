#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import models

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix, Recall, Fbeta, Precision, TopKCategoricalAccuracy
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine, WeightsHistHandler, GradsHistHandler
from ignite.contrib.engines import common

import timm

import os, shutil

import pandas as pd
import matplotlib.pylab as plt
import numpy as np

# Local modules
from cub_tools.transforms import makeDefaultTransforms
from cub_tools.data import create_dataloaders

#################################################
## Script runtime options

# Model settings
model_name = 'googlenet'
model_args = {'pretrained' : True}
model_func = models.googlenet

# Directory settings
root_dir = '/home/edmorris/projects/image_classification/caltech_birds'
data_dir = os.path.join(root_dir,'data/images')
working_dir = os.path.join(root_dir,'models/classification', 'ignite_'+model_name)
clean_up = True

# Training parameters
criterion = None # default is nn.CrossEntropyLoss
optimizer = None # default is optim.SGD
scheduler = None # default is StepLR
batch_size = 64
num_workers = 4
num_epochs = 40
early_stopping_patience = 5

# Image parameters
img_crop_size=224
img_resize=256
#################################################

def initialize(model_func, model_args, criterion=None, optimizer=None, scheduler=None, is_torchvision=True, num_classes=200):
    
     # Setup the model object
    model = model_func(**model_args)
    if is_torchvision:
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Setup loss criterion and optimizer
    if (optimizer == None):
        optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    if criterion == None:
        criterion = nn.CrossEntropyLoss()

    # Setup learning rate scheduler
    if scheduler == None:
        scheduler = StepLR(optimizer=optimizer,step_size=7, gamma=0.1)

    return model, optimizer, criterion, scheduler


def create_trainer(model, optimizer, criterion, lr_scheduler):

     # Define any training logic for iteration update
    def train_step(engine, batch):

        x, y = batch[0].to(device), batch[1].to(device)

        model.train()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        return loss.item()

    # Define trainer engine
    trainer = Engine(train_step)

    return trainer

def create_evaluator(model, metrics, tag='val'):

    # Evaluation step function
    @torch.no_grad()
    def evaluate_step(engine: Engine, batch):
        model.eval()
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        return y_pred, y

    # Create the evaluator object
    evaluator = Engine(evaluate_step)

    # Attach the metrics
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

def evaluate_model():
    epoch = trainer.state.epoch
    train_state = train_evaluator.run(train_loader)
    tr_accuracy = train_state.metrics['accuracy']
    tr_precision = train_state.metrics['precision']
    tr_recall = train_state.metrics['recall']
    tr_f1 = train_state.metrics['f1']
    tr_topKCatAcc = train_state.metrics['topKCatAcc']
    tr_loss = train_state.metrics['loss']
    print("Train - Epoch: {:0>4}  Accuracy: {:.2f} Precision: {:.2f} Recall: {:.2f} F1-score: {:.2f} TopKCatAcc: {:.2f} Loss: {:.2f}"
          .format(epoch, tr_accuracy, tr_precision, tr_recall, tr_f1, tr_topKCatAcc, tr_loss))

    val_state = evaluator.run(val_loader)
    val_accuracy = val_state.metrics['accuracy']
    val_precision = val_state.metrics['precision']
    val_recall = val_state.metrics['recall']
    val_f1 = val_state.metrics['f1']
    val_topKCatAcc = val_state.metrics['topKCatAcc']
    val_loss = val_state.metrics['loss']
    print("Valid - Epoch: {:0>4}  Accuracy: {:.2f} Precision: {:.2f} Recall: {:.2f} F1-score: {:.2f} TopKCatAcc: {:.2f} Loss: {:.2f}"
          .format(epoch, val_accuracy, val_precision, val_recall, val_f1, val_topKCatAcc, val_loss))


print('')
print('***********************************************')
print('**                                           **')
print('**         CUB 200 DATASET TRAINING          **')
print('**        --------------------------         **')
print('**                                           **')
print('**    Image Classification of 200 North      **')
print('**          American Bird Species            **')
print('**                                           **')
print('**   PyTorch Ignite Based Training Script    **')
print('**                                           **')
print('**            Ed Morris (c) 2021             **')
print('**                                           **')
print('***********************************************')
print('')
print('[INFO] Model and Directories')
print('[PARAMS] Model Name:: {}'.format(model_name))
print('[PARAMS] Model Dir:: {}'.format(working_dir))
print('[PARAMS] Data Dir:: {}'.format(data_dir))
print('')
print('[INFO] Training Parameters')
print('[PARAMS] Batch Size:: {}'.format(batch_size))
print('[PARAMS] Number of image processing CPUs:: {}'.format(num_workers))
print('[PARAMS] Number of epochs to train:: {}'.format(num_epochs))
print('')
print('[INFO] Image Settings')
print('[PARAMS] Image Size:: {}'.format(img_crop_size))
print('')

## SETUP DIRS
# Clean up the output directory by removing it if desired
if clean_up:
    shutil.rmtree(working_dir)
# Create the output directory for results
os.makedirs(working_dir, exist_ok=True)

## SETUP DEVICE
# Setup the device to run the computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## SETUP DATALOADERS
print('[INFO] Begining setting up training of model {}'.format(model_name))
print('[INFO] Setting up dataloaders for train and test sets.')
# Get data transforms
data_transforms = makeDefaultTransforms(img_crop_size=img_crop_size, img_resize=img_resize)
# Make train and test loaders
train_loader, val_loader = create_dataloaders(data_transforms=data_transforms, data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)

## SETUP MODEL OBJECTS
# Get the model, optimizer, loss criterion and learning rate scheduler objects
print('[INFO] Getting model {} from library and setting up loss criterion, optimizer and learning rate scheduler...'.format(model_name), end='')
model, optimizer, criterion, lr_scheduler = initialize(
    model_func=model_func, 
    model_args=model_args, 
    criterion=criterion, 
    optimizer=optimizer, 
    scheduler=scheduler, 
    is_torchvision=True, 
    num_classes=200
    )
print('done')
# send model to the device for training
print('[INFO] Sending model {} to device {}...'.format(model_name, device), end='')
model = model.to(device)
print('done')

## SETUP TRAINER AND EVALUATOR
# Setup model trainer and evaluator
print('[INFO] Creating Ignite training, evaluation objects and logging...', end='')
trainer = create_trainer(model=model, optimizer=optimizer, criterion=criterion, lr_scheduler=lr_scheduler)
metrics = {
    'accuracy':Accuracy(),
    'recall':Recall(average=True),
    'precision':Precision(average=True),
    'f1':Fbeta(beta=1),
    'topKCatAcc':TopKCategoricalAccuracy(k=5),
    'loss':Loss(criterion)
}
evaluator = create_evaluator(model, metrics=metrics)
train_evaluator = create_evaluator(model, metrics=metrics, tag='train')
# Add validation logging
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), evaluate_model)
# Add TB logging
evaluators = {"training": train_evaluator, "validation": evaluator}
tb_logger = common.setup_tb_logging(
    output_path=os.path.join(working_dir,'tb_logs'), 
    trainer=trainer,
    optimizers=optimizer,
    evaluators=evaluators
    )
print('done')

## TRAIN
# Train the model
print('[INFO] Executing model training...')
trainer.run(train_loader, max_epochs=num_epochs)
print('[INFO] Model training is complete.')