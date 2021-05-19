#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division
import os, pathlib

# Clear ML experiment
from clearml import Task, StorageManager, Dataset

# Local modules
from cub_tools.trainer import Trainer
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
task = Task.init(project_name='Caltech Birds', task_name='Train PyTorch CNN on CUB200', task_type=Task.TaskTypes.training)
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

# Execute task remotely
task.execute_remotely()


# Get the dataset from the clearml-server and cache locally.
print('[INFO] Getting a local copy of the CUB200 birds datasets')
# Train
train_dataset = Dataset.get(dataset_project='Caltech Birds', dataset_name='cub200_2011_train_dataset')
#train_dataset.get_mutable_local_copy(target_folder='./data/images/train')
print('[INFO] Default location of training dataset:: {}'.format(train_dataset.get_default_storage()))
train_dataset_base = train_dataset.get_local_copy()
print('[INFO] Default location of training dataset:: {}'.format(train_dataset_base))

# Test
test_dataset = Dataset.get(dataset_project='Caltech Birds', dataset_name='cub200_2011_test_dataset')
#train_dataset.get_mutable_local_copy(target_folder='./data/images/train')
print('[INFO] Default location of testing dataset:: {}'.format(test_dataset.get_default_storage()))
test_dataset_base = test_dataset.get_local_copy()
print('[INFO] Default location of testing dataset:: {}'.format(test_dataset_base))

# Amend the input data directories and output directories for remote execution
# Modify experiment root dir
params_list = params_list + ['DIRS.ROOT_DIR', '']
# Add data root dir
params_list = params_list + ['DATA.DATA_DIR', str(pathlib.PurePath(train_dataset_base).parent)]
# Add data train dir
params_list = params_list + ['DATA.TRAIN_DIR', str(pathlib.PurePath(train_dataset_base).name)]
# Add data test dir
params_list = params_list + ['DATA.TEST_DIR', str(pathlib.PurePath(test_dataset_base).name)]
# Add working dir
params_list = params_list + ['DIRS.WORKING_DIR', str(task.cache_dir)]
print('[INFO] Task output destination:: {}'.format(task.get_output_destination()))

print('[INFO] Final parameter list passed to Trainer object:: {}'.format(params_list))

trainer = Trainer(config=args.config, cmd_args=params_list)

# Setup the data transformers
trainer.create_datatransforms()

# Setup the dataloaders
trainer.create_dataloaders()

# Setup the model
trainer.create_model()

# Setup the optimizer
trainer.create_optimizer()

# Setup the scheduler
trainer.create_scheduler()

# Train the model
trainer.run()

# Save the best model
trainer.save_best_model()