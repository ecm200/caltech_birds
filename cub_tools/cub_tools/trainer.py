from __future__ import print_function, division

import os
import shutil
import tempfile
import datetime
import pathlib
import furl

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


class Trainer():
    def __init__(self, config=None, cmd_args=None, framework=None, model=None, device=None, optimizer=None, scheduler=None, criterion=None, train_loader=None, val_loader=None, data_transforms=None):

        assert (config is not None) and (cmd_args is not None), '[ERROR] Configuration not found. You must specify at least a configuration [config] or a param value pair list [cmd_args], or both and have them merged.'
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

        

        

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix, Recall, Fbeta, Precision, TopKCategoricalAccuracy
from ignite.handlers import ModelCheckpoint, EarlyStopping, Checkpoint
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine, WeightsHistHandler, GradsHistHandler
from ignite.contrib.engines import common
from torch.cuda.amp import GradScaler, autocast
class Ignite_Trainer(Trainer):

    def __init__(self, config=None, cmd_args=None, framework='ignite', model=None, device=None, optimizer=None, scheduler=None, criterion=None, train_loader=None, val_loader=None, data_transforms=None):
        super().__init__(
            config=config,
            cmd_args=cmd_args, 
            framework=framework,
            model=model, 
            device=device, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            criterion=criterion, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            data_transforms=data_transforms
            )

        self.train_engine = None
        self.evaluator = None
        self.train_evaluator = None
        self.tb_logger = None

    
    
    def create_trainer(self):

        # Define any training logic for iteration update
        def train_step(engine, batch):

            # Get the images and labels for this batch
            x, y = batch[0].to(self.device), batch[1].to(self.device)

            # Set the model into training mode
            self.model.train()

            # Zero paramter gradients
            self.optimizer.zero_grad()

            # Update the model
            if self.config.MODEL.WITH_GRAD_SCALE:
                with autocast(enabled=self.config.MODEL.WITH_AMP):
                    y_pred = self.model(x)
                    loss = self.criterion(y_pred, y)
                scaler = GradScaler(enabled=self.config.MODEL.WITH_AMP)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                with torch.set_grad_enabled(True):
                    y_pred = self.model(x)
                    loss = self.criterion(y_pred, y)
                    loss.backward()
                    self.optimizer.step()

            return loss.item()

        # Define trainer engine
        trainer = Engine(train_step)

        return trainer

    
    def create_evaluator(self, metrics, tag='val'):

        # Evaluation step function
        @torch.no_grad()
        def evaluate_step(engine: Engine, batch):
            self.model.eval()
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            if self.config.MODEL.WITH_GRAD_SCALE:
                with autocast(enabled=self.config.MODEL.WITH_AMP):
                    y_pred = self.model(x)
            else:
                y_pred = self.model(x)
            return y_pred, y

        # Create the evaluator object
        evaluator = Engine(evaluate_step)

        # Attach the metrics
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        return evaluator

    
    def evaluate_model(self):
        epoch = self.train_engine.state.epoch
        # Training Metrics
        train_state = self.train_evaluator.run(self.train_loader)
        tr_accuracy = train_state.metrics['accuracy']
        tr_precision = train_state.metrics['precision']
        tr_recall = train_state.metrics['recall']
        tr_f1 = train_state.metrics['f1']
        tr_topKCatAcc = train_state.metrics['topKCatAcc']
        tr_loss = train_state.metrics['loss']
        # Validation Metrics
        val_state = self.evaluator.run(self.val_loader)
        val_accuracy = val_state.metrics['accuracy']
        val_precision = val_state.metrics['precision']
        val_recall = val_state.metrics['recall']
        val_f1 = val_state.metrics['f1']
        val_topKCatAcc = val_state.metrics['topKCatAcc']
        val_loss = val_state.metrics['loss']
        print("Epoch: {:0>4}  TrAcc: {:.3f} ValAcc: {:.3f} TrPrec: {:.3f} ValPrec: {:.3f} TrRec: {:.3f} ValRec: {:.3f} TrF1: {:.3f} ValF1: {:.3f} TrTopK: {:.3f} ValTopK: {:.3f} TrLoss: {:.3f} ValLoss: {:.3f}"
            .format(epoch, tr_accuracy, val_accuracy, tr_precision, val_precision, tr_recall, val_recall, tr_f1, val_f1, tr_topKCatAcc, val_topKCatAcc, tr_loss, val_loss))

    
    def add_logging(self):

        # Add validation logging
        self.train_engine.add_event_handler(Events.EPOCH_COMPLETED(every=1), self.evaluate_model)

        # Add step length update at the end of each epoch
        self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self.scheduler.step())

   
    def add_tensorboard_logging(self):

        # Add TensorBoard logging
        self.tb_logger = TensorboardLogger(log_dir=os.path.join(self.config.DIRS.WORKING_DIR,'tb_logs'))
        # Logging iteration loss
        self.tb_logger.attach_output_handler(
            engine=self.train_engine, 
            event_name=Events.ITERATION_COMPLETED, 
            tag='training', 
            output_transform=lambda loss: {"batch loss": loss}
            )
        # Logging epoch training metrics
        self.tb_logger.attach_output_handler(
            engine=self.train_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="training",
            metric_names=["loss", "accuracy", "precision", "recall", "f1", "topKCatAcc"],
            global_step_transform=global_step_from_engine(self.train_engine),
        )
        # Logging epoch validation metrics
        self.tb_logger.attach_output_handler(
            engine=self.evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=["loss", "accuracy", "precision", "recall", "f1", "topKCatAcc"],
            global_step_transform=global_step_from_engine(self.train_engine),
        )
        # Attach the logger to the trainer to log model's weights as a histogram after each epoch
        self.tb_logger.attach(
            self.train_engine,
            event_name=Events.EPOCH_COMPLETED,
            log_handler=WeightsHistHandler(self.model)
        )
        # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
        self.tb_logger.attach(
            self.train_engine,
            event_name=Events.EPOCH_COMPLETED,
            log_handler=GradsHistHandler(self.model)
        )
        print('Tensorboard Logging...', end='')
        print('done')


    def create_callbacks(self, best_model_only=True):

        ## SETUP CALLBACKS
        print('[INFO] Creating callback functions for training loop...', end='')
        # Early Stopping - stops training if the validation loss does not decrease after 5 epochs
        handler = EarlyStopping(patience=self.config.EARLY_STOPPING_PATIENCE, score_function=score_function_loss, trainer=self.train_engine)
        self.evaluator.add_event_handler(Events.COMPLETED, handler)
        print('Early Stopping ({} epochs)...'.format(self.config.EARLY_STOPPING_PATIENCE), end='')

        print('Model Checkpointing...', end='')
        if best_model_only:
            print('best model checkpointing...', end='')
        # best model checkpointer, based on validation accuracy.
            val_checkpointer = ModelCheckpoint(
                dirname=self.config.DIRS.WORKING_DIR, 
                filename_prefix='caltech_birds_ignite_best', 
                score_function=score_function_acc,
                score_name='val_acc',
                n_saved=2, 
                create_dir=True, 
                save_as_state_dict=True, 
                require_empty=False,
                global_step_transform=global_step_from_engine(self.train_engine)
                )
            self.evaluator.add_event_handler(Events.COMPLETED, val_checkpointer, {self.config.MODEL.MODEL_NAME: self.model})
        else:
            # Checkpoint the model
            # iteration checkpointer
            print('every iteration model checkpointing...', end='')
            checkpointer = ModelCheckpoint(
                dirname=self.config.DIRS.WORKING_DIR, 
                filename_prefix='caltech_birds_ignite', 
                n_saved=2, 
                create_dir=True, 
                save_as_state_dict=True, 
                require_empty=False
                )
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {self.config.MODEL.MODEL_NAME: self.model})
        
        print('Done')


    def run(self):

        #assert self.model is not None, '[ERROR] No model object loaded. Please load a PyTorch model torch.nn object into the class object.'
        #assert (self.train_loader is not None) or (self.val_loader is not None), '[ERROR] You must specify data loaders.'

        for key in self.trainer_status.keys():
            assert self.trainer_status[key], '[ERROR] The {} has not been generated and you cannot proceed.'.format(key)
        print('[INFO] Trainer pass OK for training.')

        # TRAIN ENGINE
        # Create the objects for training
        self.train_engine = self.create_trainer()

        # METRICS AND EVALUATION
        # Metrics - running average
        RunningAverage(output_transform=lambda x: x).attach(self.train_engine, 'loss')

        # Metrics - epochs
        metrics = {
            'accuracy':Accuracy(),
            'recall':Recall(average=True),
            'precision':Precision(average=True),
            'f1':Fbeta(beta=1),
            'topKCatAcc':TopKCategoricalAccuracy(k=5),
            'loss':Loss(self.criterion)
            }

        # Create evaluators
        self.evaluator = self.create_evaluator(metrics=metrics)
        self.train_evaluator = self.create_evaluator(metrics=metrics, tag='train')

        # LOGGING
        # Create logging to terminal
        self.add_logging()

        # Create Tensorboard logging
        self.add_tensorboard_logging()
        
        ## CALLBACKS
        self.create_callbacks()

        ## TRAIN
        # Train the model
        print('[INFO] Executing model training...')
        self.train_engine.run(self.train_loader, max_epochs=self.config.TRAIN.NUM_EPOCHS)
        print('[INFO] Model training is complete.')

    def update_model_from_checkpoint(self, checkpoint_file=None, load_to_device=True):
        '''
        Function to take a saved checkpoint of the models weights, and load it into the model.
        '''
        assert self.trainer_status['model'], '[ERROR] You must create the model to load the weights. Use Trainer.create_model() method to first create your model, then load weights.'
        assert checkpoint_file is not None, '[ERROR] You must provide the full path and name of the .pt file containing the saved weights of the model you want to update.'

        try:
            # Load the weights of the checkpointed model from the PT file
            self.model.load_state_dict(torch.load(f=checkpoint_file))
        except:
            raise Exception('[ERROR] Something went wrong with loading the weights into the model.')
        else:
            print('[INFO] Successfully loaded weights into the model from weights file:: {}'.format(checkpoint_file))

        if load_to_device:
            self.model.to(self.device)
            print('[INFO] Successfully updated model and pushed it to the device {}'.format(self.device))
            # Print summary of model
            summary(self.model, batch_size=self.config.TRAIN.BATCH_SIZE, input_size=( 3, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size))
        else:
            print('[INFO] Successfully updated model but NOT pushed it to the device {}'.format(self.device))
            # Print summary of model
            summary(self.model, device='cpu', batch_size=self.config.TRAIN.BATCH_SIZE, input_size=( 3, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size))
        
    def convert_to_torchscript(self, checkpoint_file=None, torchscript_model_path=None, method='trace', return_jit_model=False):

        assert self.trainer_status['model'], '[ERROR] You must create the model to load the weights. Use Trainer.create_model() method to first create your model, then load weights.'
        assert checkpoint_file is not None, '[ERROR] You must provide the path and name of a PyTorch Ignite checkpoint file of model weights [checkpoint_file].'

        # Update the Trainer class attribute model with model weights file
        self.update_model_from_checkpoint(checkpoint_file=checkpoint_file)

        if torchscript_model_path is None:
            torchscript_model_path = os.path.join(os.getcwd(),'torchscript_model.pt')

        if method == 'trace':
            assert self.trainer_status['val_loader'], '[ERROR] You must create the validation loader in order to load images. Use Trainer.create_dataloaders() method to create access to image batches.'

            # Create an image batch
            X, _ = next(iter(self.val_loader))
            # Push the input images to the device
            X = X.to(self.device)
            # Trace the model
            jit_model = torch.jit.trace(self.model, (X))
            # Write the trace module of the model to disk
            print('[INFO] Torchscript file being saved to temporary location:: {}'.format(torchscript_model_path))
            jit_model.save(torchscript_model_path)

        elif method == 'script':
            # Trace the model
            jit_model = torch.jit.script(self.model)
            # Write the trace module of the model to disk
            print('[INFO] Torchscript file being saved to temporary location:: {}'.format(temp_file_path))
            jit_model.save(temp_file_path)


        if return_jit_model:
            return jit_model



from ignite.contrib.handlers.clearml_logger import ClearMLSaver
from clearml import OutputModel
class ClearML_Ignite_Trainer(Ignite_Trainer):

    def __init__(self, task, config=None, cmd_args=None, framework='ignite', model=None, device=None, optimizer=None, scheduler=None, criterion=None, train_loader=None, val_loader=None, data_transforms=None):
        super().__init__(
            config=config,
            cmd_args=cmd_args, 
            framework=framework,
            model=model, 
            device=device, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            criterion=criterion, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            data_transforms=data_transforms
            )
        
        self.task = task

    def create_callbacks(self):

        ## SETUP CALLBACKS
        print('[INFO] Creating callback functions for training loop...', end='')
        # Early Stopping - stops training if the validation loss does not decrease after 5 epochs
        handler = EarlyStopping(patience=self.config.EARLY_STOPPING_PATIENCE, score_function=score_function_loss, trainer=self.train_engine)
        self.evaluator.add_event_handler(Events.COMPLETED, handler)
        print('Early Stopping ({} epochs)...'.format(self.config.EARLY_STOPPING_PATIENCE), end='')        

        val_checkpointer = Checkpoint(
            {"model": self.model},
            ClearMLSaver(),
            n_saved=1,
            score_function=score_function_acc,
            score_name="val_acc",
            filename_prefix='cub200_{}_ignite_best'.format(self.config.MODEL.MODEL_NAME),
            global_step_transform=global_step_from_engine(self.train_engine),
        )
        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, val_checkpointer)
        print('Model Checkpointing...', end='')
        print('Done')

    
    def create_config_pbtxt(self, config_pbtxt_file=None):

        platform = "pytorch_libtorch"
        input_name = 'INPUT__0'
        output_name = 'OUTPUT__0'
        input_data_type = "TYPE_FP32"
        output_data_type = "TYPE_FP32"
        input_dims = [-1, 3, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size] # "[ -1, 3, {}, {} ]".format(self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size)
        output_dims = [-1, self.config.DATA.NUM_CLASSES]
        
        #if self.config.MODEL.MODEL_LIBRARY == 'pytorchcv':
        #    if isinstance(self.model.output, torch.nn.Linear):
        #        output_dims = "[ "+str(self.model.output.out_features).replace("None", "-1")+" ]"
        #    elif isinstance(self.model.output.fc, torch.nn.Linear):
        #        output_dims = "[ "+str(self.model.output.fc.out_features).replace("None", "-1")+" ]"
        #
        # elif self.config.MODEL.MODEL_LIBRARY == 'timm':
        #     input_name = 'input_layer'
        #     output_name = self.model.default_cfg['classifier']
        #     input_data_type = "TYPE_FP32"
        #     output_data_type = "TYPE_FP32"
        #     input_dims = "[ 3, {}, {} ]".format(self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size)
        #     output_dims = "[ "+str(self.model.get_classifier().out_features).replace("None", "-1")+" ]"
        #
        # elif self.config.MODEL.MODEL_LIBRARY == 'torchvision':
        #     input_name = 'input_layer'
        #     output_name = 'fc'
        #     input_data_type = "TYPE_FP32"
        #     output_data_type = "TYPE_FP32"
        #     input_dims = "[ 3, {}, {} ]".format(self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size, self.config.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size)
        #     output_dims = "[ "+str(self.model.fc.out_features).replace("None", "-1")+" ]"


        self.config_pbtxt = """
            platform: "%s"
            input [
                {
                    name: "%s"
                    data_type: %s
                    dims: %s
                    dims: %s
                    dims: %s
                    dims: %s
                }
            ]
            output [
                {
                    name: "%s"
                    data_type: %s
                    dims: %s
                    dims: %s
                }
            ]
        """ % (
            platform,
            input_name, input_data_type, 
            str(input_dims[0]), str(input_dims[1]), str(input_dims[2]), str(input_dims[3]),
            output_name, output_data_type, str(output_dims[0]), str(output_dims[1])
        )

        if config_pbtxt_file is not None:
            with open(config_pbtxt_file, "w") as config_file:
                config_file.write(self.config_pbtxt)


    def trace_model_for_torchscript(self, dirname=None, fname=None, model_name_preamble=None):
        '''
        Function for tracing models to Torchscript.
        '''
        assert self.trainer_status['model'], '[ERROR] You must create the model to load the weights. Use Trainer.create_model() method to first create your model, then load weights.'
        assert self.trainer_status['val_loader'], '[ERROR] You must create the validation loader in order to load images. Use Trainer.create_dataloaders() method to create access to image batches.'

        if model_name_preamble is None:
            model_name_preamble = 'Torchscript Best Model'
        
        if dirname is None:
            dirname = tempfile.mkdtemp(prefix=f"ignite_torchscripts_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_')}")
        temp_file_path = os.path.join(dirname,'model.pt')

        # Get the best model weights file for this experiment
        for chkpnt_model in self.task.get_models()['output']:
            print('[INFO] Model Found. Model Name:: {0}'.format(chkpnt_model.name))
            print('[INFO] Model Found. Mode URI:: {0}'.format(chkpnt_model.url))
            if "best_model" in chkpnt_model.name:
                print('[INFO] Using this model weights for creating Torchscript model.')
                break
        
        # Get the model weights file locally and update the model
        local_cache_path = chkpnt_model.get_local_copy()

        # Convert the model to Torchscript
        self.convert_to_torchscript(checkpoint_file=local_cache_path, torchscript_model_path=local_cache_path, method='trace', return_jit_model=False)

        # Build the remote location of the torchscript file, based on the best model weights
        # Create furl object of existing model weights
        model_furl = furl.furl(chkpnt_model.url)
        # Strip off the model path
        model_path = pathlib.Path(model_furl.pathstr)
        # Get the existing model weights name, and split the name from the file extension.
        file_split = os.path.splitext(model_path.name)
        # Create the torchscript filename
        if fname is None:
            fname = file_split[0]+"_torchscript"+file_split[1]
        # Construct the new full uri with the new filename
        new_model_furl = furl.furl(origin=model_furl.origin, path=os.path.join(model_path.parent,fname))

        # Upload the torchscript model file to the clearml-server
        print('[INFO] Pushing Torchscript model as artefact to ClearML Task:: {}'.format(self.task.id))
        new_output_model = OutputModel(
            task=self.task, 
            name=model_name_preamble+' '+self.task.name, 
            tags=['Torchscript','Deployable','Best Model', 'CUB200', self.config.MODEL.MODEL_NAME, self.config.MODEL.MODEL_LIBRARY, 'PyTorch', 'Ignite', 'Azure Blob Storage']
            )
        print('[INFO] New Torchscript model artefact added to experiment with name:: {}'.format(new_output_model.name))
        print('[INFO] Torchscript model local temporary file location:: {}'.format(temp_file_path))
        print('[INFO] Torchscript model file remote location:: {}'.format(new_model_furl.url))
        new_output_model.update_weights(
            weights_filename=temp_file_path,
            target_filename=fname
            )
        print('[INFO] Torchscript model file remote upload complete. Model saved to ID:: {}'.format(new_output_model.id))

        


        


def score_function_loss(engine):
        '''
        Scoring function on loss for the early termination handler.
        As an increase in a metric is deemed to be a positive improvement, then negative of the loss needs to be returned.
        '''
        val_loss = engine.state.metrics['loss']
        return -val_loss

def score_function_acc(engine):
    return engine.state.metrics["accuracy"]





