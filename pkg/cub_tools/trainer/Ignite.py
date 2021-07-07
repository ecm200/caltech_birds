from __future__ import print_function, division

import os

from torchsummary import summary

import torch
from cub_tools.trainer import Trainer

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, Recall, Fbeta, Precision, TopKCategoricalAccuracy
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine, WeightsHistHandler, GradsHistHandler
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
                    # With ReduceLROnPlateau, the step() call needs validation loss at the end epoch, so this is handled through an evaluator event handler rather than here.
                    if not self.config.TRAIN.SCHEDULER.TYPE == 'ReduceLROnPlateau':
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

   
    def add_tensorboard_logging(self, logging_dir=None):

        # Add TensorBoard logging
        if logging_dir is None:
            os.path.join(self.config.DIRS.WORKING_DIR,'tb_logs')
        else:
            os.path.join(logging_dir,'tb_logs')
        print('Tensorboard logging saving to:: {} ...'.format(logging_dir), end='')

        self.tb_logger = TensorboardLogger(log_dir=logging_dir)
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

        # If using ReduceLROnPlateau then need to add event to handle the step() call with loss:
        if self.config.TRAIN.SCHEDULER.TYPE == 'ReduceLROnPlateau':
            self.evaluator.add_event_handler(Events.COMPLETED, self.scheduler)
        else:
            print('No checkpointing required for LR Scheduler....', end='')

        # Early Stopping - stops training if the validation loss does not decrease after 5 epochs
        handler = EarlyStopping(patience=self.config.EARLY_STOPPING_PATIENCE, score_function=score_function_loss, trainer=self.train_engine)
        self.evaluator.add_event_handler(Events.COMPLETED, handler)
        print('Early Stopping ({} epochs)...'.format(self.config.EARLY_STOPPING_PATIENCE), end='')

        # Model checkpointing
        self._create_ingite_model_checkpointer(best_model_only=best_model_only)


    def run(self, logging_dir=None, best_model_only=True):

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
        self.add_tensorboard_logging(logging_dir=logging_dir)
        
        ## CALLBACKS
        self.create_callbacks(best_model_only=best_model_only)

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
            print('[INFO] Torchscript file being saved to temporary location:: {}'.format(torchscript_model_path))
            jit_model.save(torchscript_model_path)


        if return_jit_model:
            return jit_model
    
    def _create_ingite_model_checkpointer(self, best_model_only=True):
        '''
        Function to create an ingite model checkpointer based on validation accuracy (best model == True), or at every epoch (best model == False)
        '''

        print('Model Checkpointing...', end='')
        if best_model_only:
            print('best model checkpointing...', end='')
            # best model checkpointer, based on validation accuracy.
            self.model_checkpointer = ModelCheckpoint(
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
            self.evaluator.add_event_handler(Events.COMPLETED, self.model_checkpointer, {self.config.MODEL.MODEL_NAME: self.model})
        else:
            # Checkpoint the model
            # iteration checkpointer
            print('every iteration model checkpointing...', end='')
            self.model_checkpointer = ModelCheckpoint(
                dirname=self.config.DIRS.WORKING_DIR, 
                filename_prefix='caltech_birds_ignite', 
                n_saved=2, 
                create_dir=True, 
                save_as_state_dict=True, 
                require_empty=False
                )
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, self.model_checkpointer, {self.config.MODEL.MODEL_NAME: self.model})
        
        print('Done')


def score_function_loss(engine):
        '''
        Scoring function on loss for the early termination handler.
        As an increase in a metric is deemed to be a positive improvement, then negative of the loss needs to be returned.
        '''
        val_loss = engine.state.metrics['loss']
        return -val_loss

def score_function_acc(engine):
    return engine.state.metrics["accuracy"]