#!/bin/bash

source /opt/conda/bin/activate py38_pytorch181_cu111_timm

which python

python train_pytorch_ignite_caltech_birds.py --config configs/timm/pnasnetlarge_config.yaml

python train_pytorch_ignite_caltech_birds.py --config configs/timm/resnet152_config.yaml

python train_pytorch_ignite_caltech_birds.py --config configs/timm/resnet152d_config.yaml

python train_pytorch_ignite_caltech_birds.py --config configs/timm/resnext101_64x4d_config.yaml

python train_pytorch_ignite_caltech_birds.py --config configs/torchvision/googlenet_config.yaml

python train_pytorch_ignite_caltech_birds.py --config configs/timm/vitbase_config.yaml

python train_pytorch_ignite_caltech_birds.py --config configs/timm/swinsmall_config.yaml

python train_pytorch_ignite_caltech_birds.py --config configs/timm/swinbase_config.yaml

#python train_ignite_pytorch_googlenet_caltech_birds.py

#python train_ignite_pytorch_resnext101_caltech_birds.py

#python train_ignite_pytorch_swinsmall_caltech_birds.py


