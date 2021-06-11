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

# Add any additional arguments you'd like to pass
parser.add_argument(
    '--clearml-project',
    dest='clearml_project',
    type=str,
    help='Name of the ClearML project that you want the experiment to be logged in. [Caltech Birds/Training]', 
    default='Caltech Birds/Training')

parser.add_argument(
    '--clearml-dataset-project',
    dest='clearml_dataset_project',
    type=str,
    help='Name of the ClearML project where the dataset for training and test is located. [Caltech Birds/Datasets]', 
    default='Caltech Birds/Datasets')

parser.add_argument(
    '--clearml-dataset-train',
    dest='clearml_dataset_train',
    type=str,
    help='Name of the ClearML training dataset. [cub200_2011_train_dataset]', 
    default='cub200_2011_train_dataset')

parser.add_argument(
    '--clearml-dataset-test',
    dest='clearml_dataset_test',
    type=str,
    help='Name of the ClearML testing dataset. [cub200_2011_test_dataset]', 
    default='cub200_2011_test_dataset')

parser.add_argument(
    '--clearml-output-url',
    dest='clearml_output_url',
    type=str,
    help='Location of where the output files should be stored. Default is Azure Blob Storage. Format is azure://storage_account/container [azure://clearmllibrary/artefacts]', 
    default='azure://clearmllibrary/artefacts')

parser.add_argument(
        '--clearml-task-clone',
        dest='clearml_task_clone',
        action='store_true',
        help='Create a clone of the task to be run on the remote resource, rather than running this experiment. [False]',
        default=False
    )
parser.add_argument(
    '--clearml-queue',
    dest='clearml_queue',
    type=str,
    help='Name of the ClearML-Server queue that the task will be enqueded with for remote execution. Use "none" for no queuing of job. Default [gpu]', 
    default='gpu')

parser.print_help()
args = parser.parse_args()

# If no queue provided by specifying 'none', then make it None
if args.clearml_queue.lower() == 'none':
    args.clearml_queue = None

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
    project_name=args.clearml_project,
    task_name='TRAIN [Network: '+cfg.MODEL.MODEL_NAME+', Library: '+cfg.MODEL.MODEL_LIBRARY+'] Ignite Train PyTorch CNN on CUB200', 
    task_type=Task.TaskTypes.training,
    output_uri=args.clearml_output_url
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

    # Execute task remotely, control whether task is cloned, or queued.
    task.execute_remotely(
        queue_name=args.clearml_queue,
        clone=args.clearml_task_clone,
        exit_process=True)

    # Get the dataset from the clearml-server and cache locally.
    print('[INFO] Getting a local copy of the CUB200 birds datasets')
    # Train
    train_dataset = Dataset.get(dataset_project=args.clearml_dataset_project, dataset_name=args.clearml_dataset_train)
    #train_dataset.get_mutable_local_copy(target_folder='./data/images/train')
    print('[INFO] Default location of training dataset:: {}'.format(train_dataset.get_default_storage()))
    train_dataset_base = train_dataset.get_local_copy()
    print('[INFO] Default location of training dataset:: {}'.format(train_dataset_base))

    # Test
    test_dataset = Dataset.get(dataset_project=args.clearml_dataset_project, dataset_name=args.clearml_dataset_test)
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

# NOTE: Placeholder here, call conversion to torchscript and save to file, and upload to clearml-server / remote storage.

## Save the best model
#trainer.save_best_model()