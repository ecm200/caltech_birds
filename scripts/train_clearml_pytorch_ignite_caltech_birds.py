#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

# Clear ML experiment
from clearml import Task, StorageManager, Dataset


# Local modules
from cub_tools.trainer import Ignite_Trainer
from cub_tools.args import get_parser

# Get the arguments from the command line, including configuration file and any overrides.
parser = get_parser()
parser.print_help()
args = parser.parse_args()

#print('[INFO] Optional Arguments from CLI:: {}'.format(args.opts))
#if args.opts == '[]':
#    args.opts = list()
#    print('[INFO] Setting empty CLI args to an explicit empty list')

## CLEAR ML
# Connecting with the ClearML process
task = Task.init(project_name='Caltech Birds', task_name='Train network on CUB200')
# Add the local python package as a requirement
task.add_requirements('./cub_tools')
task.add_requirements('git+https://github.com/rwightman/pytorch-image-models.git')
# Setup ability to add configuration parameters control.
params = {'TRAIN.NUM_EPOCHS': 20, 'TRAIN.BATCH_SIZE': 32, 'TRAIN.OPTIMIZER.PARAMS.lr': 0.001, 'TRAIN.OPTIMIZER.PARAMS.momentum': 0.9}
params = task.connect(params)  # enabling configuration override by clearml
print(params)  # printing actual configuration (after override in remote mode)
# Convert Params dictionary into a set of key value pairs in a list
params_list = []
for key in params:
    params_list.extend([key,params[key]])

# Check if the task is running locally.
# If not then, get the datasets from the server.
if not task.running_locally:
    print('[INFO] Getting a local copy of the CUB200 birds dataset')
    train_dataset = Dataset.get(dataset_project='Caltech Birds', dataset_name='cub200_2011_train_dataset')
    train_dataset.get_mutable_local_copy(target_folder='./data/images/train')
    test_dataset = Dataset.get(dataset_project='Caltech Birds', dataset_name='cub200_2011_test_dataset')
    train_dataset.get_mutable_local_copy(target_folder='./data/images/train')

# Create the trainer object
trainer = Ignite_Trainer(config=args.config, cmd_args=params_list) # NOTE: disabled cmd line argument passing but using it to pass ClearML configs.

# Setup the data transformers
print('[INFO] Creating data transforms...')
trainer.create_datatransforms()

# Setup the dataloaders
print('[INFO] Creating data loaders...')
trainer.create_dataloaders()

# Setup the model
print('[INFO] Creating the model...')
trainer.create_model()

# Setup the optimizer
print('[INFO] Creating optimizer...')
trainer.create_optimizer()

# Setup the scheduler
trainer.create_scheduler()

# Train the model
trainer.run()

## Save the best model
#trainer.save_best_model()