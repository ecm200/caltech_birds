#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, division
from ignite.distributed import auto

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast

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
model_name = 'resnext101_32x8d'
model_func = models.resnext101_32x8d
model_args = {'pretrained' : True}
is_torchvision = True
with_amp = True
with_grad_scale = False

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

def initialize(device, model_func, model_args, criterion=None, optimizer=None, scheduler=None, is_torchvision=True, num_classes=200):
    
     # Setup the model object
    model = model_func(**model_args)
    if is_torchvision:
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

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

        # Get the images and labels for this batch
        x, y = batch[0].to(device), batch[1].to(device)

        # Set the model into training mode
        model.train()

        # Zero paramter gradients
        optimizer.zero_grad()

        # Update the model
        if with_grad_scale:
            with autocast(enabled=with_amp):
                y_pred = model(x)
                loss = criterion(y_pred, y)
            scaler = GradScaler(enabled=with_amp)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.set_grad_enabled(True):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

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
        if with_grad_scale:
            with autocast(enabled=with_amp):
                y_pred = model(x)
        else:
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
    # Training Metrics
    train_state = train_evaluator.run(train_loader)
    tr_accuracy = train_state.metrics['accuracy']
    tr_precision = train_state.metrics['precision']
    tr_recall = train_state.metrics['recall']
    tr_f1 = train_state.metrics['f1']
    tr_topKCatAcc = train_state.metrics['topKCatAcc']
    tr_loss = train_state.metrics['loss']
    # Validation Metrics
    val_state = evaluator.run(val_loader)
    val_accuracy = val_state.metrics['accuracy']
    val_precision = val_state.metrics['precision']
    val_recall = val_state.metrics['recall']
    val_f1 = val_state.metrics['f1']
    val_topKCatAcc = val_state.metrics['topKCatAcc']
    val_loss = val_state.metrics['loss']
    print("Epoch: {:0>4}  TrAcc: {:.3f} ValAcc: {:.3f} TrPrec: {:.3f} ValPrec: {:.3f} TrRec: {:.3f} ValRec: {:.3f} TrF1: {:.3f} ValF1: {:.3f} TrTopK: {:.3f} ValTopK: {:.3f} TrLoss: {:.3f} ValLoss: {:.3f}"
          .format(epoch, tr_accuracy, val_accuracy, tr_precision, val_precision, tr_recall, val_recall, tr_f1, val_f1, tr_topKCatAcc, val_topKCatAcc, tr_loss, val_loss))

def score_function_loss(engine):
    '''
    Scoring function on loss for the early termination handler.
    As an increase in a metric is deemed to be a positive improvement, then negative of the loss needs to be returned.
    '''
    val_loss = engine.state.metrics['loss']
    return -val_loss

def score_function_acc(engine):
    return engine.state.metrics["accuracy"]

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
print('[')
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
if clean_up and (os.path.exists(working_dir)):
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
print('[INFO] Getting model {} from library and sending to device, setting up loss criterion, optimizer and learning rate scheduler...'.format(model_name), end='')
model, optimizer, criterion, lr_scheduler = initialize(
    device=device,
    model_func=model_func, 
    model_args=model_args, 
    criterion=criterion, 
    optimizer=optimizer, 
    scheduler=scheduler, 
    is_torchvision=is_torchvision, 
    num_classes=200
    )
print('done')

## SETUP TRAINER AND EVALUATOR
# Setup model trainer and evaluator
print('[INFO] Creating Ignite training, evaluation objects and logging...', end='')
trainer = create_trainer(model=model, optimizer=optimizer, criterion=criterion, lr_scheduler=lr_scheduler)
# Metrics - running average
RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
# Metrics - epochs
metrics = {
    'accuracy':Accuracy(),
    'recall':Recall(average=True),
    'precision':Precision(average=True),
    'f1':Fbeta(beta=1),
    'topKCatAcc':TopKCategoricalAccuracy(k=5),
    'loss':Loss(criterion)
}

# Create evaluators
evaluator = create_evaluator(model, metrics=metrics)
train_evaluator = create_evaluator(model, metrics=metrics, tag='train')

# Add validation logging
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), evaluate_model)

# Add step length update at the end of each epoch
trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: lr_scheduler.step())

# Add TensorBoard logging
tb_logger = TensorboardLogger(log_dir=os.path.join(working_dir,'tb_logs'))
# Logging iteration loss
tb_logger.attach_output_handler(
    engine=trainer, 
    event_name=Events.ITERATION_COMPLETED, 
    tag='training', 
    output_transform=lambda loss: {"batch loss": loss}
    )
# Logging epoch training metrics
tb_logger.attach_output_handler(
    engine=train_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names=["loss", "accuracy", "precision", "recall", "f1", "topKCatAcc"],
    global_step_transform=global_step_from_engine(trainer),
)
# Logging epoch validation metrics
tb_logger.attach_output_handler(
    engine=evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="validation",
    metric_names=["loss", "accuracy", "precision", "recall", "f1", "topKCatAcc"],
    global_step_transform=global_step_from_engine(trainer),
)
# Attach the logger to the trainer to log model's weights as a histogram after each epoch
tb_logger.attach(
    trainer,
    event_name=Events.EPOCH_COMPLETED,
    log_handler=WeightsHistHandler(model)
)
# Attach the logger to the trainer to log model's gradients as a histogram after each epoch
tb_logger.attach(
    trainer,
    event_name=Events.EPOCH_COMPLETED,
    log_handler=GradsHistHandler(model)
)
print('Tensorboard Logging...', end='')
print('done')

## SETUP CALLBACKS
print('[INFO] Creating callback functions for training loop...', end='')
# Early Stopping - stops training if the validation loss does not decrease after 5 epochs
handler = EarlyStopping(patience=early_stopping_patience, score_function=score_function_loss, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)
print('Early Stopping ({} epochs)...'.format(early_stopping_patience), end='')

# Checkpoint the model
# iteration checkpointer
checkpointer = ModelCheckpoint(
    dirname=working_dir, 
    filename_prefix='caltech_birds_ignite', 
    n_saved=2, 
    create_dir=True, 
    save_as_state_dict=True, 
    require_empty=False
    )
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {model_name: model})
# best model checkpointer, based on validation accuracy.
val_checkpointer = ModelCheckpoint(
    dirname=working_dir, 
    filename_prefix='caltech_birds_ignite_best', 
    score_function=score_function_acc,
    score_name='val_acc',
    n_saved=2, 
    create_dir=True, 
    save_as_state_dict=True, 
    require_empty=False,
    global_step_transform=global_step_from_engine(trainer)
    )
evaluator.add_event_handler(Events.COMPLETED, val_checkpointer, {model_name: model})
print('Model Checkpointing...', end='')
print('Done')


## TRAIN
# Train the model
print('[INFO] Executing model training...')
trainer.run(train_loader, max_epochs=num_epochs)
print('[INFO] Model training is complete.')