from __future__ import print_function, division

import os
import shutil
import numpy as np

from torchsummary import summary

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler as torch_scheduler

from cub_tools.data import create_dataloaders
from cub_tools.config import get_cfg_defaults
from cub_tools.transforms import makeDefaultTransforms, makeAggresiveTransforms
from cub_tools.train import train_model
from cub_tools.utils import save_model_dict, save_model_full
from cub_tools.schedulers import ReduceLROnPlateauScheduler


class Trainer():
    def __init__(self, config=None, cmd_args=None, framework=None, model=None, device=None, optimizer=None, scheduler=None, criterion=None, train_loader=None, val_loader=None, data_transforms=None):

        #assert ((config is not None) and (cmd_args is not None)), '[ERROR] Configuration not found. You must specify at least a configuration [config] or a param value pair list [cmd_args], or both and have them merged.'
        # Create status check dictionary, model will only execute training if all True.
        self.trainer_status = {
            'model' : False,
            'optimizer' : False,
            'criterion' : False,
            'scheduler' : False,
            'train_loader' : False,
            'val_loader' : False,
            'data_transforms' : False
        }
        print('[INFO] Parameters Override:: {}'.format(cmd_args))

        # LOAD CONFIGURATION
        # Configuration can be provided by YAML file or key value pair list, or both.
        # NOTE: key value pair list take precedent over YAML file.
        # Load the default model configuration.
        self.config = get_cfg_defaults()
        # Load configuration from YAML file if it is provided and overide defaults.
        if (config is not None):
            self.config.merge_from_file(config)
        # Override config from command line arguments
        if (cmd_args is not None): #or (cmd_args == '[]') or (not cmd_args):
            self.config.merge_from_list(cmd_args)
        # Creating correct working directories
        if framework is not None:
            self.config.DIRS.WORKING_DIR = os.path.join(self.config.DIRS.ROOT_DIR, self.config.DIRS.WORKING_DIR, framework+'_'+self.config.MODEL.MODEL_NAME)
        else:
            self.config.DIRS.WORKING_DIR = os.path.join(self.config.DIRS.ROOT_DIR, self.config.DIRS.WORKING_DIR, self.config.MODEL.MODEL_NAME)
        self.config.DATA.DATA_DIR = os.path.join(self.config.DIRS.ROOT_DIR, self.config.DATA.DATA_DIR)
        # Freeze the configuration
        self.config.freeze()
        print(self.config)

        # Check the model output directories are there and if not make them.
        # Clean up the output directory by removing it if desired
        if self.config.DIRS.CLEAN_UP and (os.path.exists(self.config.DIRS.WORKING_DIR)):
            shutil.rmtree(self.config.DIRS.WORKING_DIR)
        # Create the output directory for results
        os.makedirs(self.config.DIRS.WORKING_DIR, exist_ok=True)

        # Check for specified device, if none take GPU by default.
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Check for provided model object, if none, don't load one and user needs to call get_model method
        if model is None:
            # Get the model loading function and arguments from the configuration file
            if self.config.MODEL.MODEL_LIBRARY == 'torchvision':
                self._select_torchvision_model()
                
            elif self.config.MODEL.MODEL_LIBRARY == 'timm':
                import timm
                self.model_func = timm.create_model
                self.model_args = {
                        'model_name' : self.config.MODEL.MODEL_NAME,
                        'pretrained' : self.config.MODEL.PRETRAINED,
                        'num_classes' : self.config.DATA.NUM_CLASSES
                        }
            
            elif self.config.MODEL.MODEL_LIBRARY == 'pytorchcv':
                from pytorchcv.model_provider import get_model as ptcv_get_model
                self.model_func = ptcv_get_model
                self.model_args = {
                        'name' : self.config.MODEL.MODEL_NAME,
                        'pretrained' : self.config.MODEL.PRETRAINED,
                        #'num_classes' : self.config.DATA.NUM_CLASSES
                        }
            
            else:
                print('Choose a proper library name.') # TODO: Raise an exception here
        else:
            self.model = model
            print('[WARNING] No model has been specified. Please either pass with class init, or use get_model method.')

        self.dataset_sizes = {}
        if train_loader is None:
            self.train_loader = None
        else:
            self.train_loader = train_loader
            self.dataset_sizes['train'] = len(self.train_loader.dataset.imgs)
            self.trainer_status['train_loader'] = True
        
        if val_loader is None:
            self.val_loader = None
        else:
            self.val_loader = val_loader
            self.dataset_sizes['test'] = len(self.val_loader.dataset.imgs)
            self.trainer_status['val_loader'] = True

        if data_transforms is None:
            if self.config.DATA.TRANSFORMS.TYPE == 'default':
                self.data_transforms = makeDefaultTransforms
                self.data_transforms_args = dict(self.config.DATA.TRANSFORMS.PARAMS.DEFAULT)
            elif self.config.DATA.TRANSFORMS.TYPE == 'aggresive':
                self.data_transforms = makeAggresiveTransforms
                self.data_transforms_args = dict(self.config.DATA.TRANSFORMS.PARAMS.DEFAULT).update(dict(self.config.DATA.TRANSFORMS.PARAMS.AGGRESIVE))

        else:
            self.data_transforms = data_transforms
            self.trainer_status['data_transforms'] = True

        if optimizer is None:
            self.optimizer_args = dict(self.config.TRAIN.OPTIMIZER.PARAMS)
            if self.config.TRAIN.OPTIMIZER.TYPE == 'SGD':
                self.optimizer = optim.SGD
        else:
            self.optimizer = optimizer
            self.trainer_status['optimizer'] = True

        
        if criterion is None:
            if self.config.TRAIN.LOSS.CRITERION == 'CrossEntropy':
                self.criterion = nn.CrossEntropyLoss()
                self.trainer_status['criterion'] = True
        else:
            self.criterion = criterion
            self.trainer_status['criterion'] = True
        
        if scheduler is None:
            #self.scheduler_args = dict(self.config.TRAIN.SCHEDULER.PARAMS)

            # Take key var paired list of scheduler parameters and turn into dictionary
            self.scheduler_args = {}
            for i in np.arange(0,len(self.config.TRAIN.SCHEDULER.PARAMS),2):
                self.scheduler_args[self.config.TRAIN.SCHEDULER.PARAMS[i]] = self.config.TRAIN.SCHEDULER.PARAMS[i+1]

            # Get the right scheduler based on the config argument
            if self.config.TRAIN.SCHEDULER.TYPE == 'StepLR':
                self.scheduler = torch_scheduler.StepLR
            elif self.config.TRAIN.SCHEDULER.TYPE == 'MultiStepLR':
                self.scheduler = torch_scheduler.MultiStepLR
            elif self.config.TRAIN.SCHEDULER.TYPE == 'CyclicLR':
                self.scheduler = torch_scheduler.CyclicLR
            elif self.config.TRAIN.SCHEDULER.TYPE == 'ExponentialLR':
                self.scheduler = torch_scheduler.ExponentialLR
            elif self.config.TRAIN.SCHEDULER.TYPE == 'CosineAnnealingLR':
                self.scheduler = torch_scheduler.CosineAnnealingLR
            elif self.config.TRAIN.SCHEDULER.TYPE == 'ReduceLROnPlateau':
                self.scheduler = ReduceLROnPlateauScheduler # special case because the step call requires a validation loss.
            elif self.config.TRAIN.SCHEDULER.TYPE == 'CosineAnnealingWarmRestarts':
                self.scheduler = torch_scheduler.CosineAnnealingWarmRestarts
            else:
                raise Exception('[ERROR] You must specify one of the PyTorch standard learning rate schedulers: StepLR, MultiStepLR, CyclicLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts')
        else:
            self.scheduler = scheduler
            self.trainer_status['scheduler'] = True

        # Init other attributes for later use
        self.history = None
        self.best_model = None

    def create_datatransforms(self):

        self.data_transforms = self.data_transforms(**self.data_transforms_args)
        self.trainer_status['data_transforms'] = True

    # Default dataloader creation
    def create_dataloaders(self, shuffle=None):

        assert self.trainer_status['data_transforms'], '[ERROR] You need to specify data transformers to create data loaders.'
        
        self.train_loader, self.val_loader = create_dataloaders(
            data_transforms=self.data_transforms, 
            data_dir=self.config.DATA.DATA_DIR, 
            train_dir=self.config.DATA.TRAIN_DIR,
            test_dir=self.config.DATA.TEST_DIR,
            batch_size=self.config.TRAIN.BATCH_SIZE, 
            num_workers=self.config.TRAIN.NUM_WORKERS,
            shuffle=shuffle
            )
        self.dataset_sizes = {
            'train' : len(self.train_loader.dataset.imgs),
            'test' : len(self.val_loader.dataset.imgs)
        }
        self.trainer_status['train_loader'] = True
        self.trainer_status['val_loader'] = True

    def create_model(self, model_func=None, model_args=None, load_to_device=True):

        if model_func is None:
            assert self.model_func is not None, '[ERROR] You must specify a model library loading function.'
        if model_args is None:
            assert self.model_args is not None, '[ERROR] You must specify model arguments if you loading from library.'

        # assert self.model is None, '[ERROR] You already have an active model.'
        self.model = self.model_func(**self.model_args)

        if self.config.MODEL.MODEL_LIBRARY == 'torchvision':
            # Change output classifier to be the number of classes in the current problem.
            self.model.fc = nn.Linear(self.model.fc.in_features, self.config.DATA.NUM_CLASSES)

        elif self.config.MODEL.MODEL_LIBRARY == 'pytorchcv':
            # Change output classifier to be the number of classes in the current problem.
            if isinstance(self.model.output, torch.nn.Linear):
                self.model.output = nn.Linear(self.model.output.in_features, self.config.DATA.NUM_CLASSES)
            elif isinstance(self.model.output.fc, torch.nn.Linear):
                self.model.output.fc = nn.Linear(self.model.output.fc.in_features, self.config.DATA.NUM_CLASSES)

            
        # Send the model to the computation device 
        if load_to_device:   
            self.model.to(self.device)
            print('[INFO] Successfully created model and pushed it to the device {}'.format(self.device))
            # Print summary of model
            summary(self.model, batch_size=self.config.TRAIN.BATCH_SIZE, input_size=( 3, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size))
        else:
            print('[INFO] Successfully created model but NOT pushed it to the device {}'.format(self.device))
            # Print summary of model
            summary(self.model, device='cpu', batch_size=self.config.TRAIN.BATCH_SIZE, input_size=( 3, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size))

        self.trainer_status['model'] = True

        

    def create_optimizer(self):

        assert self.trainer_status['model'], '[ERROR] You need to generate a model to create an optimizer.'

        # Make the optimizer
        assert self.optimizer_args is not None, '[ERROR] Optimizer arguments need to be supplied.'
        self.optimizer = self.optimizer(self.model.parameters(), **self.optimizer_args)

        self.trainer_status['optimizer'] = True
        print('[INFO] Successfully created optimizer object.')



    def create_scheduler(self):

        assert self.trainer_status['model'], '[ERROR] You need to generate a model to create a scheduler.'
        assert self.trainer_status['optimizer'], '[ERROR] You need to generate an optimizer to create a scheduler.'

        # Make the learning rate scheduler
        assert self.scheduler_args is not None, '[ERROR] Scheduler arguments need to be supplied.'
        self.scheduler = self.scheduler(self.optimizer, **self.scheduler_args)

        self.trainer_status['scheduler'] = True
        print('[INFO] Successfully created learning rate scheduler object.')

    def run(self):

        #assert self.model is not None, '[ERROR] No model object loaded. Please load a PyTorch model torch.nn object into the class object.'
        #assert (self.train_loader is not None) or (self.val_loader is not None), '[ERROR] You must specify data loaders.'

        for key in self.trainer_status.keys():
            assert self.trainer_status[key], '[ERROR] The {} has not been generated and you cannot proceed.'.format(key)
        print('[INFO] Trainer pass OK for training.')

        print('[INFO] Commencing model training.')
        # Train the model
        self.best_model, self.history = train_model(
            model=self.model,
            criterion=self.criterion, 
            optimizer=self.optimizer, 
            scheduler=self.scheduler, 
            device=self.device, 
            dataloaders={'train' : self.train_loader, 'test' : self.val_loader}, 
            dataset_sizes=self.dataset_sizes, 
            num_epochs=self.config.TRAIN.NUM_EPOCHS,
            return_history=True, 
            log_history=self.config.SYSTEM.LOG_HISTORY, 
            working_dir=self.config.DIRS.WORKING_DIR
            )

    def save_best_model(self, output_dir=None, filename_postfix=None):

        assert self.best_model is not None, '[ERROR] The best model attribute is empty, you likely need to train the model first.'

        # Ability to save to bespoke place or default to working directory.
        if output_dir is None:
            output_dir = self.config.DIRS.WORKING_DIR
        
        # Add some extra text to the filename.
        if filename_postfix is None:
            filename_postfix = ''

        full_path = os.path.join(output_dir,'caltech_birds_{}_full{}.pth'.format(self.config.MODEL.MODEL_NAME, filename_postfix))
        state_path = os.path.join(output_dir,'caltech_birds_{}_dict{}.pth'.format(self.config.MODEL.MODEL_NAME, filename_postfix))

        # Save out the best model and finish
        save_model_full(model=self.best_model, PATH=full_path)
        save_model_dict(model=self.best_model, PATH=state_path)

        print('[INFO] Model has been successfully saved to the following directory: {}'.format(output_dir))
        print('[INFO] The full model filename is: caltech_birds_{}_full.pth'.format(full_path))
        print('[INFO] The state dictionary filename is: caltech_birds_{}_full.pth'.format(state_path))

    def _select_torchvision_model(self):

        from torchvision import models

        if self.config.MODEL.MODEL_NAME == 'googlenet':
            self.model_func = models.googlenet
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED, 'aux_logits' : False}

        elif self.config.MODEL.MODEL_NAME == 'inception_v3':
            self.model_func = models.inception_v3
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED, 'aux_logits' : False}

        elif self.config.MODEL.MODEL_NAME == 'alexnet':
            self.model_func = models.alexnet
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'vgg11':
            self.model_func = models.vgg11
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'vgg13':
            self.model_func = models.vgg13
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'vgg16':
            self.model_func = models.vgg16
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'vgg19':
            self.model_func = models.vgg19
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'resnet34':
            self.model_func = models.resnet34
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'resnet50':
            self.model_func = models.resnet50
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'resnet101':
            self.model_func = models.resnet101
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'resnet152':
            self.model_func = models.resnet152
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'squeezenet1_0':
            self.model_func = models.squeezenet1_0
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'squeezenet1_1':
            self.model_func = models.squeezenet1_1
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'densenet121':
            self.model_func = models.densenet121
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'densenet169':
            self.model_func = models.densenet169
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'densenet161':
            self.model_func = models.densenet161
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'densenet201':
            self.model_func = models.densenet201
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'mobilenet_v2':
            self.model_func = models.mobilenet_v2
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'mobilenet_v3_large':
            self.model_func = models.mobilenet_v3_large
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'mobilenet_v3_small':
            self.model_func = models.mobilenet_v3_small
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'resnext50_32x4d':
            self.model_func = models.resnext50_32x4d
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'resnext101_32x8d':
            self.model_func = models.resnext101_32x8d
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'wide_resnet50_2':
            self.model_func = models.wide_resnet50_2
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'wide_resnet101_2':
            self.model_func = models.wide_resnet101_2
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'mnasnet0_5':
            self.model_func = models.mnasnet0_5
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'mnasnet0_75':
            self.model_func = models.mnasnet0_75
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'mnasnet1_0':
            self.model_func = models.mnasnet1_0
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

        elif self.config.MODEL.MODEL_NAME == 'mnasnet1_3':
            self.model_func = models.mnasnet1_3
            self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}