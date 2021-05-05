#!/bin/bash

source /opt/conda/bin/activate py38_pytorch181_cu111_timm

which python

python train_pytorch_swinbase_caltech_birds.py

python train_pytorch_swinlarge_caltech_birds.py


