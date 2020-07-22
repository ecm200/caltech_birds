# Introduction

This is the Caltech Birds respository using PyTorch CNNs for image classification (CUB-200-2011).

This is a collection of notebooks and tools designed to show how to setup, build, train and evaluate Convolutional Neural Network architectures using PyTorch, Torchvision and other 3rd party packages, to generate state-of-the-art classification results on a fine-grained, long-tailed distribution classification problem. The set of example notebooks will cover the following workflow:

![workflow](docs/birds_roadmap.png)

The repository includes a set of example notebooks which walks the user through all the processes required to train and evaluate a network, as well as interrogate what and how the network is making it's decisions through the use of neuron, layer and spatial activations interpretation ([Lucent](https://github.com/greentfrapp/lucent)) and image feature attributions ([Captum](https://captum.ai/)).


# The data

The data can be downloaded from the CalTech USCD Birds website as a zip file. 



# Installation

The repository should be cloned into a local directory.

Additional steps are required to make a suitable python environment for running.


## Requirements

All model training has been performed on **PyTorch >=v1.4**.
All other packages are standard data science tools, for a full list see the **requirement.txt** file

## CUB_TOOLS package installation

To install the cub_tools set of modules into your PyTorch environment, do the following:

  *cd cub_tools*
  *pip install .*
  
 This should create an installed package in your python environment called **cub_tools**.



# Additional files

