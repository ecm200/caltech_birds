#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

import argparse

# Local modules
from cub_tools.trainer import Trainer

parser = argparse.ArgumentParser(description='PyTorch Image Classification Trainer - Ed Morris (c) 2021')
parser.add_argument('--config', metavar="FILE", help='Path and name of configuration file for training. Should be a .yaml file.', required=False, default='scripts/configs/pytorchcv/efficientnet_b0_config.yaml')
parser.print_help()
args = parser.parse_args()
#config = 'configs/googlenet_config.yaml'

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