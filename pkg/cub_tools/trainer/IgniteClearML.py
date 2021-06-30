from __future__ import print_function, division

import os
import tempfile
import datetime
import pathlib
import furl

from ignite.engine import Events
from ignite.handlers import EarlyStopping, Checkpoint
from ignite.contrib.handlers.tensorboard_logger import  global_step_from_engine
from ignite.contrib.handlers.clearml_logger import ClearMLSaver

from clearml import OutputModel

from .Ignite import Ignite_Trainer
from .Ignite import score_function_acc, score_function_loss

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
        self.convert_to_torchscript(checkpoint_file=local_cache_path, torchscript_model_path=temp_file_path, method='trace', return_jit_model=False)

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

