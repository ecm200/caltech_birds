import os
import argparse

# ClearML modules
from clearml import Dataset

parser = argparse.ArgumentParser(description='CUB200 2011 ClearML data uploader - Ed Morris (c) 2021')
parser.add_argument(
    '--dataset-basedir',
    dest='dataset_basedir',
    type=str,
    help='The directory to the root of the dataset', 
    default='/home/edmorris/projects/image_classification/caltech_birds/data/images')
parser.add_argument(
    '--clearml-project',
    dest='clearml_project',
    type=str,
    help='The name of the clearml project that the dataset will be stored and published to.', 
    default='Caltech Birds/Datasets')
parser.add_argument(
    '--clearml-dataset-url',
    dest='clearml_dataset_url',
    type=str,
    help='Location of where the dataset files should be stored. Default is Azure Blob Storage. Format is azure://storage_account/container', 
    default='azure://clearmllibrary/datasets')
args = parser.parse_args()

for task_type in ['train','test']:
    print('[INFO] Versioning and uploading {0} dataset for CUB200 2011'.format(task_type))
    dataset = Dataset.create('cub200_2011_{0}_dataset'.format(task_type), dataset_project=args.clearml_project)
    dataset.add_files(path=os.path.join(args.dataset_basedir,task_type), verbose=False)
    dataset.upload(output_url=args.clearml_dataset_url)
    print('[INFO] {0} Dataset finalized....'.format(task_type), end='')
    dataset.finalize()
    print('done.')

    print('[INFO] {0} Dataset published....'.format(task_type), end='')
    dataset.publish()
    print('done.')