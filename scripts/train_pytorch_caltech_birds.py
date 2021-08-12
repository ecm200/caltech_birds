#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

# Local modules
from cub_tools.trainer import Trainer
from cub_tools.args import get_parser

'''
Basic Training Script

Train a PyTorch model using a custom, basic training function.

Ed Morris (c) 2021.
'''

## PROGRAM START

# Get the arguments from the command line, including configuration file and any overrides.
parser = get_parser()
parser.print_help()
args = parser.parse_args()

trainer = Trainer(config=args.config)

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
