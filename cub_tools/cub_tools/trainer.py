from __future__ import print_function, division

import os
import shutil

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler as torch_scheduler

from cub_tools.data import create_dataloaders
from cub_tools.config import get_cfg_defaults
from cub_tools.transforms import makeDefaultTransforms, makeAggresiveTransforms
from cub_tools.train import train_model
from cub_tools.utils import save_model_dict, save_model_full


class Trainer():
    def __init__(self, config, model=None, device=None, optimizer=None, scheduler=None, criterion=None, train_loader=None, val_loader=None, data_transforms=None):

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

        # Load the model configuration
        self.config = get_cfg_defaults()
        self.config.merge_from_file(config)
        # Creating correct working directories
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
                from torchvision import models
                if self.config.MODEL.MODEL_NAME == 'googlenet':
                    self.model_func = models.googlenet
                    self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

                elif self.config.MODEL.MODEL_NAME == 'vgg16':
                    self.model_func = models.vgg16
                    self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

                elif self.config.MODEL.MODEL_NAME == 'resnext101_32x8d':
                    self.model_func = models.resnext101_32x8d
                    self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

                elif self.config.MODEL.MODEL_NAME == 'resnet152':
                    self.model_func = models.resnet152
                    self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

                elif self.config.MODEL.MODEL_NAME == 'inception_v3':
                    self.model_func = models.inception_v3
                    self.model_args = {'pretrained' : self.config.MODEL.PRETRAINED}

            if self.config.MODEL.MODEL_LIBRARY == 'timm':
                import timm
                self.model_func = timm.create_model
                self.model_args = {
                        'model_name' : self.config.MODEL.MODEL_NAME,
                        'pretrained' : self.config.MODEL.PRETRAINED,
                        'num_classes' : self.config.DATA.NUM_CLASSES
                        }
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
            self.scheduler_args = dict(self.config.TRAIN.SCHEDULER.PARAMS)
            if self.config.TRAIN.SCHEDULER.TYPE == 'StepLR':
                self.scheduler = torch_scheduler.StepLR
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
    def create_dataloaders(self):

        assert self.trainer_status['data_transforms'], '[ERROR] You need to specify data transformers to create data loaders.'
        
        self.train_loader, self.val_loader = create_dataloaders(
            data_transforms=self.data_transforms, 
            data_dir=self.config.DATA.DATA_DIR, 
            batch_size=self.config.TRAIN.BATCH_SIZE, 
            num_workers=self.config.TRAIN.NUM_WORKERS
            )
        self.dataset_sizes = {
            'train' : len(self.train_loader.dataset.imgs),
            'test' : len(self.val_loader.dataset.imgs)
        }
        self.trainer_status['train_loader'] = True
        self.trainer_status['val_loader'] = True

    def create_model(self, model_func=None, model_args=None):

        if model_func is None:
            assert self.model_func is not None, '[ERROR] You must specify a model library loading function.'
        if model_args is None:
            assert self.model_args is not None, '[ERROR] You must specify model arguments if you loading from library.'

        # assert self.model is None, '[ERROR] You already have an active model.'
        self.model = self.model_func(**self.model_args)

        if self.config.MODEL.MODEL_LIBRARY == 'torchvision':
            num_ftrs = self.model.fc.in_features
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
            self.model.fc = nn.Linear(num_ftrs, self.config.DATA.NUM_CLASSES)
            
        # Send the model to the computation device    
        self.model.to(self.device)

        self.trainer_status['model'] = True

        print('[INFO] Successfully created model and pushed it to the device {}'.format(self.device))

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