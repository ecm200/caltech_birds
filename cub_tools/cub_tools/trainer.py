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
    def __init__(self, config, framework=None, model=None, device=None, optimizer=None, scheduler=None, criterion=None, train_loader=None, val_loader=None, data_transforms=None):

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
            # Change output classifier to be the number of classes in the current problem.
            self.model.fc = nn.Linear(self.model.fc.in_features, self.config.DATA.NUM_CLASSES)

        elif self.config.MODEL.MODEL_LIBRARY == 'pytorchcv':
            # Change output classifier to be the number of classes in the current problem.
            self.model.output.fc = nn.Linear(self.model.output.fc.in_features, self.config.DATA.NUM_CLASSES)

            
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
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine, WeightsHistHandler, GradsHistHandler
from ignite.contrib.engines import common
from torch.cuda.amp import GradScaler, autocast
class Ignite_Trainer(Trainer):

    
    def __init__(self, config, framework='ignite', model=None, device=None, optimizer=None, scheduler=None, criterion=None, train_loader=None, val_loader=None, data_transforms=None):
        super().__init__(
            config, 
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


    def create_callbacks(self):

        ## SETUP CALLBACKS
        print('[INFO] Creating callback functions for training loop...', end='')
        # Early Stopping - stops training if the validation loss does not decrease after 5 epochs
        handler = EarlyStopping(patience=self.config.EARLY_STOPPING_PATIENCE, score_function=score_function_loss, trainer=self.train_engine)
        self.evaluator.add_event_handler(Events.COMPLETED, handler)
        print('Early Stopping ({} epochs)...'.format(self.config.EARLY_STOPPING_PATIENCE), end='')

        # Checkpoint the model
        # iteration checkpointer
        checkpointer = ModelCheckpoint(
            dirname=self.config.DIRS.WORKING_DIR, 
            filename_prefix='caltech_birds_ignite', 
            n_saved=2, 
            create_dir=True, 
            save_as_state_dict=True, 
            require_empty=False
            )
        self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {self.config.MODEL.MODEL_NAME: self.model})
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
        print('Model Checkpointing...', end='')
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


def score_function_loss(engine):
        '''
        Scoring function on loss for the early termination handler.
        As an increase in a metric is deemed to be a positive improvement, then negative of the loss needs to be returned.
        '''
        val_loss = engine.state.metrics['loss']
        return -val_loss

def score_function_acc(engine):
    return engine.state.metrics["accuracy"]