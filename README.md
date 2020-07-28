![Title_image](docs/birds_classification_title_image.png)

# Image Classification of birds

**A whistle stop tour of how to use the latest image classification techniques to build and understand a deep neural network bird classifier**


This is an investigation using PyTorch CNNs of deep image classificaton to solve a bird species classification problem with the [Caltech UCSD Birds dataset (CUB-200-2011)](http://www.vision.caltech.edu/visipedia/CUB-200.html).

This is a collection of notebooks and tools designed to show how to setup, build, train and evaluate Convolutional Neural Network architectures using PyTorch, Torchvision and other 3rd party packages, to generate state-of-the-art classification results on a fine-grained, long-tailed distribution classification problem. The set of example notebooks will cover the following workflow:

![workflow](docs/birds_roadmap.png)

The repository includes a set of example notebooks which walks the user through all the processes required to train and evaluate a network, as well as interrogate what and how the network is making it's decisions through the use of neuron, layer and spatial activations interpretation ([Lucent](https://github.com/greentfrapp/lucent)) and image feature attributions ([Captum](https://captum.ai/)).

We round off the work flow exploration by showing how to convert the PyTorch trained model into an [ONNX (Open Neural Network eXchange)](https://onnx.ai/) format, which enables the bird species classifier to be deployed on vast array of platforms, either in the cloud or on personal devices. We test the ONNX version of the model using the ***ONNX Runtime*** and compare these predictions to the PyTorch version of the model, to show that the inference performance is the same.


# Dataset details and data

## Dataset

Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations. For detailed information about the dataset, please see the technical report linked below.



Number of categories: 200

Number of images: 11,788

Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box

Some related datasets are Caltech-256, the Oxford Flower Dataset, and Animals with Attributes. More datasets are available at the Caltech Vision Dataset Archive.

## Data files

To download the data click on the following links:

   1. Images and annotations [CalTech Visipedia Website - CUB_200_2011.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   
   **The image tarball should be downloaded to repository root, and extracted into the data/images_orig sub-directory, as shown below.***
               
    
   2. Segmentations (optional, not need for this work) [CalTech Visipedia Website - segmentations.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz)
    
Place the files into the root of the cloned caltech_birds repo file structure.

Unzip the dowloaded zip files into cloned repository structure that replicates the project structure:

    caltech_birds-|
        cub_tools-|
        data-|   ** SEPARATE DOWNLOAD FROM CALTECH VISIPEDIA WEBSITE **
            attributes-|
            images_orig-|   ** RENAME IMAGES FOLDER TO IMAGE_ORIG **
               ##class_name##-|
                  class001-image001.jpg
                  class001-image002.jpg
            images-|   ** NEW FOLDER CREATED BY FIRST WORKFLOW PROCESS **
               train-|
                  ##class_name##-|
                     class###-image001.jpg
               test-|
                  ##class_name##-|
                     class###-image035.jpg
            parts-|
            attributes.txt
            bounding_boxes.txt
            classes.txt
            image_class_labels.txt
            images.txt
            README
            train_test_split.txt
        example-notebooks-|   ** WORKSHOP MATERIAL IS IN HERE **
        models-|   ** SEPARATE DOWNLOAD FROM RELEASES **
            classification-|
                #modelname1#-|
                #modelname2#-|
        notebooks-|      ** INVESTIGATIONS OF ALL TYPES OF CNN ARCHITECTURES ON THE BIRD CLASSIFICATION PROBLEM **
        scripts-|        ** TRAINING SCRIPTS FOR PYTORCH CNN MODELS **



# Workshop walk-through notebooks series

**example_notebooks directory contains the walk-through workshop notebooks**, for following the full workflow of producing a bird classifier using deep neural networks, using a ResNet152 deep neural network architecture.

See below on the requirements for installing and using the notebooks, as  well details of the additional folders including more CNN architecture investigations using the CUB-200-2011 dataset.


# Installation

The repository should be cloned into a local directory.

Additional steps are required to make a suitable python environment for running.

## Requirements

All models have been produced using **Python v3.7**.
All model training has been performed on **PyTorch == v1.4.0**.
All other packages are standard data science tools, for a full list see the **requirement.txt** file

## CUB_TOOLS package installation

To install the cub_tools set of modules into your PyTorch environment, do the following:

  *cd cub_tools*
  
  *pip install .*
  
 This should create an installed package in your python environment called **cub_tools**.

# Additional files
        
**notebooks** directory contain the Jupyter notebooks where the majority of the visualisation and high level code will be maintained.

**scripts** directory contains the computationally intensive training scripts for various deep CNN architectures, which take longer to run (even on GPU) and are better executed in a python script. I suggest this be done using some form of terminal persistance method (if running in the cloud) to keep the terminal session open whilst the model is training, allowing you to log off the remote host without killing the process. These can typically take a few hours, to a day or so to complete the prescribed epochs. I prefer to use [TMUX](https://github.com/tmux/tmux/wiki/Getting-Started), which is a Linux utility that maintains separate terminal sessions that can be attached and detached to any linux terminal session once opened. This way, you can execute long running python training scripts inside a TMUX session, detach it, close down the terminal session and let the process run. You can then start a new terminal session, and then attach the running TMUX session to view the progress of the executed script, including all its output to terminal history.

**cub_tools** directory contains all the utility functions that have been developed to process, visualise, train and evaluate CNN models, as well results post processing have been contained. It has been converted into a python package that can be installed in the local environment by running in the **cub_tools** directory, *pip install -e .*. The functions can then be accessed using the cub_tools module import as ***import cub_tools***.

**models** directory contains the results from the model training processes, and also any other outputs from the evaluation processes including model predictions, network feature maps etc. **All model outputs used by the example notebooks can be downloaded from the release folder of the Github repo.** The models zip should be placed in the root of the repo directory structure and unziped to create a models directory with the ResNet152 results contained within. Other models are also available including PNASNET, Inception V3 and V4, GoogLenet and ResNeXt variants.
