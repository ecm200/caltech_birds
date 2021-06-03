#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division
import pathlib

# Clear ML experiment
from clearml import Task, Dataset

# Local modules
from cub_tools.trainer import ClearML_Ignite_Trainer
from cub_tools.args import get_parser
from cub_tools.config import get_cfg_defaults, get_key_value_dict


# Get the arguments from the command line, including configuration file and any overrides.
parser = get_parser()
parser.print_help()
args = parser.parse_args()

if args.run_local:
    print('[INFO] Running the job locally and logging to ClearML-Server')

#print('[INFO] Optional Arguments from CLI:: {}'.format(args.opts))
#if args.opts == '[]':
#    args.opts = list()
#    print('[INFO] Setting empty CLI args to an explicit empty list')

## CLEAR ML
# Tmp config load for network name
cfg = get_cfg_defaults()
cfg.merge_from_file(args.config)
params = get_key_value_dict(cfg)

# Connecting with the ClearML process
# First add the repo package requirements that aren't on CONDA / PYPI
Task.add_requirements('git+https://github.com/ecm200/caltech_birds.git@clearml_dev#egg=cub_tools&subdirectory=cub_tools/')
Task.add_requirements('git+https://github.com/rwightman/pytorch-image-models.git')
# Now connect the script to ClearML Server as an experiment.
task = Task.init(
    project_name='Caltech Birds',
    task_name='[Network: '+cfg.MODEL.MODEL_NAME+', Library: '+cfg.MODEL.MODEL_LIBRARY+'] Ignite Train PyTorch CNN on CUB200', 
    task_type=Task.TaskTypes.training,
    output_uri='azure://clearmllibrary/artefacts'
    )

# Add tags to the experiment to show in the ClearML GUI for better grouping
task.add_tags(['CUB200', cfg.MODEL.MODEL_NAME, cfg.MODEL.MODEL_LIBRARY, 'PyTorch', 'Ignite', 'Deployable', 'Azure Blob Storage'])

# Setup ability to add configuration parameters control.
# Pass the YACS configuration object directly to task object for storting of all parameters with model on clearml-server
params = task.connect(cfg, name='YACS')  # enabling configuration override by clearml
#print(params)  # printing actual configuration (after override in remote mode)
# Convert Params dictionary into a set of key value pairs in a list
params_list = []
for key in params:
    params_list.extend([key,params[key]])

# Run the training remotely on ClearML Server. 
# If True, this will create an experiment argument on the ClearML Server and terminate running locally.
# If False, the model will train locally, and logging and artefacts will be capture as normal.
if not args.run_local:
    # Execute task remotely
    task.execute_remotely()

    # Get the dataset from the clearml-server and cache locally.
    print('[INFO] Getting a local copy of the CUB200 birds datasets')
    # Train
    train_dataset = Dataset.get(dataset_project='Caltech Birds', dataset_name='cub200_2011_train_dataset__AZURE_BLOB_VERSION')
    #train_dataset.get_mutable_local_copy(target_folder='./data/images/train')
    print('[INFO] Default location of training dataset:: {}'.format(train_dataset.get_default_storage()))
    train_dataset_base = train_dataset.get_local_copy()
    print('[INFO] Default location of training dataset:: {}'.format(train_dataset_base))

    # Test
    test_dataset = Dataset.get(dataset_project='Caltech Birds', dataset_name='cub200_2011_test_dataset__AZURE_BLOB_VERSION')
    #train_dataset.get_mutable_local_copy(target_folder='./data/images/train')
    print('[INFO] Default location of testing dataset:: {}'.format(test_dataset.get_default_storage()))
    test_dataset_base = test_dataset.get_local_copy()
    print('[INFO] Default location of testing dataset:: {}'.format(test_dataset_base))

    # Amend the input data directories and output directories for remote execution
    # Modify experiment root dir
    params_list = params_list + ['DIRS.ROOT_DIR', '']
    # Add data root dir
    params_list = params_list + ['DATA.DATA_DIR', str(pathlib.PurePath(train_dataset_base).parent)]
    # Add data train dir
    params_list = params_list + ['DATA.TRAIN_DIR', str(pathlib.PurePath(train_dataset_base).name)]
    # Add data test dir
    params_list = params_list + ['DATA.TEST_DIR', str(pathlib.PurePath(test_dataset_base).name)]
    # Add working dir
    params_list = params_list + ['DIRS.WORKING_DIR', str(task.cache_dir)]
    print('[INFO] Task output destination:: {}'.format(task.get_output_destination()))

print('[INFO] Final parameter list passed to Trainer object:: {}'.format(params_list))

# Create the trainer object
trainer = ClearML_Ignite_Trainer(config=args.config, cmd_args=params_list) # NOTE: disabled cmd line argument passing but using it to pass ClearML configs.

# Setup the data transformers
print('[INFO] Creating data transforms...')
trainer.create_datatransforms()

# Setup the dataloaders
print('[INFO] Creating data loaders...')
trainer.create_dataloaders()

# Setup the model
print('[INFO] Creating the model...')
trainer.create_model()

# Setup the optimizer
print('[INFO] Creating optimizer...')
trainer.create_optimizer()

# Setup the scheduler
print('[INFO] Creating LR Scheduler...')
trainer.create_scheduler()

# Train the model
print('[INFO] Training the model...')
trainer.run()

# Create the deployment script and add it to the experiment.
print('[INFO] Creating deployment configuration...')
trainer.create_config_pbtxt(config_pbtxt_file='config.pbtxt')
task.connect_configuration(configuration=pathlib.Path('config.pbtxt'), name='config.pbtxt')

## Save the best model
#trainer.save_best_model()