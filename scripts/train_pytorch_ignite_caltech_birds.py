#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

import argparse

# Local modules
from cub_tools.trainer import Ignite_Trainer
from cub_tools.args import get_parser

'''
Ignite Training Script

Train a PyTorch model using the Ignite framework.
Ed Morris (c) 2021.
'''

## PROGRAM START

# Get the arguments from the command line, including configuration file and any overrides.
parser = get_parser()
parser.print_help()
args = parser.parse_args()

trainer = Ignite_Trainer(config=args.config)

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
