#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets
from torchvision import models

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
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
model_name = 'googlenet'
model_func = models.googlenet

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

def score_function_loss(engine):
    '''
    Scoring function on loss for the early termination handler.
    As an increase in a metric is deemed to be a positive improvement, then negative of the loss needs to be returned.
    '''
    val_loss = engine.state.metrics['loss']
    return -val_loss

def score_function_acc(engine):
    return engine.state.metrics["accuracy"]

def process_function(engine,batch):
    model.train()
    optimizer.zero_grad()
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    labels_pred = model(inputs)
    loss = criterion(labels_pred,labels)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    return loss.item()

def thresholded_output_transformation(inputs, labels, labels_pred, loss):
    loss = criterion(labels_pred,labels)
    return loss.item()

def thresholded_output_transformation_evaluation(inputs, labels, labels_pred):
    labels_pred = torch.max(labels_pred)
    return labels_pred, labels

def eval_function(engine, batch):
    model.eval()
    with torch.no_grad():
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_pred = model(inputs)
        return labels_pred, labels


def log_training_results(trainer):
    train_evaluator.run(dataloaders['train'])
    metrics = train_evaluator.state.metrics
    accuracy = metrics['accuracy']*100.0
    precision = metrics['precision']*100.0
    recall = metrics['recall']*100.0
    f1 = metrics['f1']*100.0
    topKCatAcc = metrics['topKCatAcc']*100.0
    loss = metrics['loss']
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
    accuracy = metrics['accuracy']*100.0
    precision = metrics['precision']*100.0
    recall = metrics['recall']*100.0
    f1 = metrics['f1']*100.0
    topKCatAcc = metrics['topKCatAcc']*100.0
    loss = metrics['loss']
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
model = model_func(pretrained=True)
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, len(class_names))
print('done')
print('[INFO] Sending model {} to device {}...'.format(model_name, device), end='')
model = model.to(device)
print('done')


print('[INFO] Setting up optimiser and loss functions...', end='')
# Set up the loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print('done')

## SETUP TRAINING AND EVALUATION
#################################

print('[INFO] Creating Ignite training object and metrics...', end='')
# Create a trainer using ignite
#trainer = create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=criterion, device=device, output_transform=thresholded_output_transformation)
trainer = Engine(process_function)

# Specify metrics
metrics = {
    'accuracy':Accuracy(),
    'recall':Recall(average=True),
    'precision':Precision(average=True),
    'f1':Fbeta(beta=1),
    'topKCatAcc':TopKCategoricalAccuracy(k=5),
    'loss':Loss(criterion)
}

# Create evaluators
#train_evaluator = create_supervised_evaluator(model=model, metrics=metrics, device=device, output_transform=thresholded_output_transformation_evaluation)
#val_evaluator = create_supervised_evaluator(model=model, metrics=metrics, device=device, output_transform=thresholded_output_transformation_evaluation)
train_evaluator = Engine(eval_function)
metrics['accuracy'].attach(train_evaluator, 'accuracy')
metrics['recall'].attach(train_evaluator, 'recall')
metrics['precision'].attach(train_evaluator, 'precision')
metrics['f1'].attach(train_evaluator, 'f1')
metrics['topKCatAcc'].attach(train_evaluator, 'topKCatAcc')
metrics['loss'].attach(train_evaluator, 'loss')
val_evaluator = Engine(eval_function)
metrics['accuracy'].attach(val_evaluator, 'accuracy')
metrics['recall'].attach(val_evaluator, 'recall')
metrics['precision'].attach(val_evaluator, 'precision')
metrics['f1'].attach(val_evaluator, 'f1')
metrics['topKCatAcc'].attach(val_evaluator, 'topKCatAcc')
metrics['loss'].attach(val_evaluator, 'loss')

# Create records for history
training_history = {'accuracy':[],'precision':[],'recall':[],'f1':[],'topKCatAcc':[],'loss':[]}
validation_history = {'accuracy':[],'precision':[],'recall':[],'f1':[],'topKCatAcc':[],'loss':[]}
last_epoch=[]

# Metrics - running average
RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
print('done')

print('[INFO] Creating callback functions for training loop...', end='')
# Early Stopping - stops training if the validation loss does not decrease after 5 epochs
handler = EarlyStopping(patience=early_stopping_patience, score_function=score_function_loss, trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, handler)
print('Early Stopping ({} epochs)...'.format(early_stopping_patience), end='')

# Report training metrics to terminal
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
print('Metrics logging...', end='')

# Learning Rate Scheduler
# Decay LR by a factor of 0.1 every 7 epochs
lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
scheduler = LRScheduler(lr_scheduler)
# NOTE: Does this have to be a function with scheduler.step() in it, instead of just the function being passed?
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
print('Learning Rate Schedule...', end='')

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
val_evaluator.add_event_handler(Events.COMPLETED, val_checkpointer, {model_name: model})
print('Model Checkpointing...', end='')

# Tensorboard logger
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
    engine=val_evaluator,
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


print('[INFO] Executing model training...')
# Train the model
trainer.run(dataloaders['train'], max_epochs=num_epochs)

print('[INFO] Model training is complete.')