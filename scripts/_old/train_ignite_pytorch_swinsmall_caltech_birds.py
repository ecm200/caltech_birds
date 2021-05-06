#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import datasets

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix, Recall, Fbeta, Precision, TopKCategoricalAccuracy
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine, WeightsHistHandler, GradsHistHandler

import timm

import os

import pandas as pd
import matplotlib.pylab as plt
import numpy as np

# Local modules
from cub_tools.transforms import makeDefaultTransforms

'''
Training script for image classification models of the CUB 200 dataset.

This version uses PyTorch Ignite to control the training and validation phases.
Ed Morris (c) 2021
'''
#################################################
## Script runtime options

# Model settings
model_name = 'swin_small_patch4_window7_224'
model_func = timm.create_model

# Directory settings
root_dir = '/home/edmorris/projects/image_classification/caltech_birds'
data_dir = os.path.join(root_dir,'data/images')
working_dir = os.path.join(root_dir,'models/classification', 'ignite_'+model_name)

# Training parameters
batch_size = 16
num_workers = 4
num_epochs = 40
early_stopping_patience = 5

# Image parameters
img_crop_size=224
img_resize=256
#################################################

def score_function(engine):
    val_loss = engine.state.metrics['nll']
    return -val_loss

def log_training_results(trainer):
    train_evaluator.run(dataloaders['train'])
    metrics = train_evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    precision = metrics['precision']*100
    recall = metrics['recall']*100
    f1 = metrics['f1']*100
    topKCatAcc = metrics['topKCatAcc']*100
    loss = metrics['nll']
    last_epoch.append(0)
    training_history['accuracy'].append(accuracy)
    training_history['precision'].append(precision)
    training_history['recall'].append(recall)
    training_history['f1'].append(f1)
    training_history['topKCatAcc'].append(topKCatAcc)
    training_history['loss'].append(loss)
    print("Train - Epoch: {:0>4}  Accuracy: {:.2f} Precision: {:.2f} Recall: {:.2f} F1-score: {:.2f} TopKCatAcc: {:.2f} Loss: {:.2f}"
          .format(trainer.state.epoch, accuracy, precision, recall, f1, topKCatAcc, loss))

def log_validation_results(trainer):
    val_evaluator.run(dataloaders['test'])
    metrics = val_evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    precision = metrics['precision']*100
    recall = metrics['recall']*100
    f1 = metrics['f1']*100
    topKCatAcc = metrics['topKCatAcc']*100
    loss = metrics['nll']
    validation_history['accuracy'].append(accuracy)
    validation_history['precision'].append(precision)
    validation_history['recall'].append(recall)
    validation_history['f1'].append(f1)
    validation_history['topKCatAcc'].append(topKCatAcc)
    validation_history['loss'].append(loss)
    print("Valid - Epoch: {:0>4}  Accuracy: {:.2f} Precision: {:.2f} Recall: {:.2f} F1-score: {:.2f} TopKCatAcc: {:.2f} Loss: {:.2f}"
          .format(trainer.state.epoch, accuracy, precision, recall, f1, topKCatAcc, loss))

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

os.makedirs(working_dir, exist_ok=True)

## SETUP DATALOADERS
#####################

print('[INFO] Begining setting up training of model {}'.format(model_name))
print('[INFO] Setting up dataloaders for train and test sets.')
# Get data transforms
data_transforms = makeDefaultTransforms(img_crop_size=img_crop_size, img_resize=img_resize)

# Setup data loaders with augmentation transforms
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

print('***********************************************')
print('**            DATASET SUMMARY                **')
print('***********************************************')
for dataset in dataset_sizes.keys():
    print(dataset,' size:: ', dataset_sizes[dataset],' images')
print('Number of classes:: ', len(class_names))
print('***********************************************')
print('[INFO] Created data loaders.')

## SETUP MODEL AND OPTIMIZER
#############################

# Setup the device to run the computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup the model and send to the device
print('[INFO] Getting model {} from library...'.format(model_name), end='')
model_ft = model_func(model_name, pretrained=True, num_classes=len(class_names))
print('done')
print('[INFO] Sending model {} to device {}...'.format(model_name, device), end='')
model_ft = model_ft.to(device)
print('done')


print('[INFO] Setting up optimiser and loss functions...', end='')
# Set up the loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
print('done')

## SETUP TRAINING AND EVALUATION
#################################

print('[INFO] Creating Ignite training object and metrics...', end='')
# Create a trainer using ignite
trainer = create_supervised_trainer(model=model_ft, optimizer=optimizer_ft, loss_fn=criterion, device=device)

# Specify metrics
metrics = {
    'accuracy':Accuracy(),
    'recall':Recall(average=True),
    'precision':Precision(average=True),
    'f1':Fbeta(beta=1),
    'topKCatAcc':TopKCategoricalAccuracy(k=5),
    'nll':Loss(criterion)
}

# Create evaluators
train_evaluator = create_supervised_evaluator(model=model_ft, metrics=metrics, device=device)
val_evaluator = create_supervised_evaluator(model=model_ft, metrics=metrics, device=device)

# Create records for history
training_history = {'accuracy':[],'precision':[],'recall':[],'f1':[],'topKCatAcc':[],'loss':[]}
validation_history = {'accuracy':[],'precision':[],'recall':[],'f1':[],'topKCatAcc':[],'loss':[]}
last_epoch=[]

# Metrics - running average
RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
print('done')

print('[INFO] Creating callback functions for training loop...', end='')
# Early Stopping - stops training if the validation loss does not decrease after 5 epochs
handler = EarlyStopping(patience=early_stopping_patience, score_function=score_function, trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, handler)
print('Early Stopping ({} epochs)...'.format(early_stopping_patience), end='')

# Report training metrics to terminal
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
print('Metrics logging...', end='')

# Learning Rate Scheduler
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
scheduler = LRScheduler(exp_lr_scheduler)
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
print('Learning Rate Schedule...', end='')

# Checkpoint the model
checkpointer = ModelCheckpoint(dirname=working_dir, filename_prefix='caltech_birds_ignite', n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {model_name: model_ft})
print('Model Checkpointing...', end='')

# Tensorboard logger
tb_logger = TensorboardLogger(log_dir=os.path.join(working_dir,'tb_logs'))
# Logging iteration loss
tb_logger.attach_output_handler(
    engine=trainer, 
    event_name=Events.ITERATION_COMPLETED, 
    tag='training', 
    output_transform=lambda loss: {"loss": loss}
    )
# Logging epoch training metrics
tb_logger.attach_output_handler(
    engine=train_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names=["nll", "accuracy", "precision", "recall", "f1", "topKCatAcc"],
    global_step_transform=global_step_from_engine(trainer),
)
# Logging epoch validation metrics
tb_logger.attach_output_handler(
    engine=val_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="validation",
    metric_names=["nll", "accuracy", "precision", "recall", "f1", "topKCatAcc"],
    global_step_transform=global_step_from_engine(trainer),
)
# Attach the logger to the trainer to log model's weights as a histogram after each epoch
tb_logger.attach(
    trainer,
    event_name=Events.EPOCH_COMPLETED,
    log_handler=WeightsHistHandler(model_ft)
)
# Attach the logger to the trainer to log model's gradients as a histogram after each epoch
tb_logger.attach(
    trainer,
    event_name=Events.EPOCH_COMPLETED,
    log_handler=GradsHistHandler(model_ft)
)
print('Tensorboard Logging...', end='')
print('done')


print('[INFO] Executing model training...')
# Train the model
trainer.run(dataloaders['train'], max_epochs=num_epochs)

print('[INFO] Model training is complete.')