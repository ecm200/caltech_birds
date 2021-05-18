#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

# Clear ML experiment
from clearml import Task, StorageManager


# Local modules
from cub_tools.trainer import Ignite_Trainer
from cub_tools.args import get_parser

# Get the arguments from the command line, including configuration file and any overrides.
parser = get_parser()
parser.print_help()
args = parser.parse_args()

# Connecting with the ClearML process
task = Task.init(project_name='Caltech Birds', task_name='Train network on CUB200')
params = {'TRAIN.NUM_EPOCHS': 20, 'TRAIN.BATCH_SIZE': 32, 'TRAIN.OPTIMIZER.PARAMS.lr': 0.001, 'TRAIN.OPTIMIZER.PARAMS.momentum': 0.9}
params = task.connect(params)  # enabling configuration override by clearml
print(params)  # printing actual configuration (after override in remote mode)


# Create the trainer object
trainer = Ignite_Trainer(config=args.config, cmd_args=args.opts)

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
#trainer.run()

## Save the best model
#trainer.save_best_model()