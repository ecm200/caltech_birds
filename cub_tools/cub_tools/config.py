# my_project/config.py
from yacs.config import CfgNode as CN
import yacs

_C = CN()

# Model library description
_C.MODEL = CN()
# Model library being used: currently torchvision, timm or pytorchcv.
_C.MODEL.MODEL_LIBRARY = 'torchvision'
# Name of the particular model you want to train e.g. googlenet
_C.MODEL.MODEL_NAME = 'test'
# Pretrain weights
_C.MODEL.PRETRAINED = True
# With nVidia AMP
_C.MODEL.WITH_AMP = False
# With Grad Scaling
_C.MODEL.WITH_GRAD_SCALE = False


# Training parameters
_C.TRAIN = CN()
# Number of images to train in each batch
_C.TRAIN.BATCH_SIZE = 16
# Number of worker processes to load images
_C.TRAIN.NUM_WORKERS = 4
# Total number of epochs to train for
_C.TRAIN.NUM_EPOCHS = 40
# Early stopping in epochs 
_C.EARLY_STOPPING_PATIENCE = 5
# Loss Criterion
_C.TRAIN.LOSS = CN()
# Loss function
_C.TRAIN.LOSS.CRITERION = 'CrossEntropy'  # other types are: 
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'SGD' # other types are: 
_C.TRAIN.OPTIMIZER.PARAMS = CN()
_C.TRAIN.OPTIMIZER.PARAMS.lr = 0.001
_C.TRAIN.OPTIMIZER.PARAMS.momentum = 0.9
_C.TRAIN.OPTIMIZER.PARAMS.nesterov = True
# Learning Rate Scheduluer
_C.TRAIN.SCHEDULER = CN()
_C.TRAIN.SCHEDULER.TYPE = 'StepLR'
_C.TRAIN.SCHEDULER.PARAMS = ['step_size', 7, 'gamma', 0.1] # For StepLR
#_C.TRAIN.SCHEDULER.PARAMS = ['lr_lambda', lmbda = lambda epoch: 0.95] # For MultiplicativeLR
#_C.TRAIN.SCHEDULER.PARAMS = ['milestones', [30, 80], 'gamma', 0.1] # For MultiStepLR
#_C.TRAIN.SCHEDULER.PARAMS = ['gamma', 0.1] # For ExponentialLR
#_C.TRAIN.SCHEDULER.PARAMS = ['T_max', _C.TRAIN.NUM_EPOCHS, 'eta_min', 0.0001] # For CosineAnnealingLR
#_C.TRAIN.SCHEDULER.PARAMS = ['mode', 'min', 'factor', 0.1, 'patience', 10, 'threshold', 0.0001, 'threshold_mode', 'rel', 'cooldown', 0, 'min_lr', 0, 'eps', 1e-08] # For ReduceLROnPlateau


# Model directories
_C.DIRS = CN()
# Root of the project
_C.DIRS.ROOT_DIR = '../'
# Output dir for model training results
_C.DIRS.WORKING_DIR = 'models/classification'
# Before running training, delete existing model output directory.
_C.DIRS.CLEAN_UP = True

# Data parameters
_C.DATA = CN()
# Dir containing train test folders of images
_C.DATA.DATA_DIR = 'data/images'
_C.DATA.TRAIN_DIR = 'train'
_C.DATA.TEST_DIR = 'test'
# Number of classes
_C.DATA.NUM_CLASSES = 200
# Image transforms
_C.DATA.TRANSFORMS = CN()
_C.DATA.TRANSFORMS.TYPE = 'default' # Other option is aggresive
_C.DATA.TRANSFORMS.PARAMS = CN()
# Options for default transforms
_C.DATA.TRANSFORMS.PARAMS.DEFAULT = CN()
_C.DATA.TRANSFORMS.PARAMS.DEFAULT.img_crop_size = 224
_C.DATA.TRANSFORMS.PARAMS.DEFAULT.img_resize = 256
# Additional options for aggresive
_C.DATA.TRANSFORMS.PARAMS.AGGRESIVE = CN()
_C.DATA.TRANSFORMS.PARAMS.AGGRESIVE.type = 'all' # Other options are perspective and rotation
_C.DATA.TRANSFORMS.PARAMS.AGGRESIVE.persp_distortion_scale = 0.25
_C.DATA.TRANSFORMS.PARAMS.AGGRESIVE.rotation_range = (-10.0,10.0)


# Anything else that needs setting
_C.SYSTEM = CN()
# Save history to disk as pkl file
_C.SYSTEM.LOG_HISTORY = True



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()



def get_key_value_dict(cfg):
    cfg_params = {}
    for key_layer_1 in cfg.keys():
        if isinstance(cfg[key_layer_1], yacs.config.CfgNode):
            for key_layer_2 in cfg[key_layer_1].keys():
                if isinstance(cfg[key_layer_1][key_layer_2], yacs.config.CfgNode):
                    for key_layer_3 in cfg[key_layer_1][key_layer_2].keys():
                        if isinstance(cfg[key_layer_1][key_layer_2][key_layer_3], yacs.config.CfgNode):
                            for key_layer_4 in cfg[key_layer_1][key_layer_2][key_layer_3].keys():
                                if isinstance(cfg[key_layer_1][key_layer_2][key_layer_3][key_layer_4], yacs.config.CfgNode):
                                    for key_layer_5 in cfg[key_layer_1][key_layer_2][key_layer_3][key_layer_4].keys():
                                        cfg_params[key_layer_1+'.'+key_layer_2+'.'+key_layer_3+'.'+key_layer_4+'.'+key_layer_5] = cfg[key_layer_1][key_layer_2][key_layer_3][key_layer_4][key_layer_5]
                                else:
                                    cfg_params[key_layer_1+'.'+key_layer_2+'.'+key_layer_3+'.'+key_layer_4] = cfg[key_layer_1][key_layer_2][key_layer_3][key_layer_4]
                        else:
                            cfg_params[key_layer_1+'.'+key_layer_2+'.'+key_layer_3] = cfg[key_layer_1][key_layer_2][key_layer_3]
                else:
                    cfg_params[key_layer_1+'.'+key_layer_2] = cfg[key_layer_1][key_layer_2]
        else:
            cfg_params[key_layer_1] = cfg[key_layer_1]
    
    return cfg_params