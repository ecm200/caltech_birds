import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path




def create_dataloaders(data_transforms, data_dir, batch_size, num_workers, train_dir=None, test_dir=None, shuffle=None, test_batch_size=2):
    '''
    Given directories of images for train and test datasets, organized in sub folders of class under train and test directories, build dataloader objects to serve to the network.

    Cub Tools
    Ed Morris (c) 2021
    '''

    # If no bespoke folders are set for images, then assume train and test folders are in the root data dir.
    if train_dir is None:
        train_dir = 'train'
    if test_dir is None:
        test_dir = 'test'

    # Set the name of the train and test image directories
    images_dirs = {'train' : train_dir, 'test' : test_dir}

    # Set the batch sizes for each operation
    batch_size = {'train' : batch_size, 'test' : test_batch_size}

    # Set the shuffle option dict for each operation
    if shuffle == None:
        shuffle = {'train' : True, 'test' : False}

    # Setup data loaders with augmentation transforms
    image_datasets = {x: ImageFolder(os.path.join(data_dir, images_dirs[x]), data_transforms[x])
                    for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size[x],
                                 shuffle=shuffle[x], num_workers=num_workers)
                for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    print('***********************************************')
    print('**            DATASET SUMMARY                **')
    print('***********************************************')
    for dataset in dataset_sizes.keys():
        print(dataset,' size:: ', dataset_sizes[dataset],' images')
    print('Number of classes:: ', len(class_names))
    print('***********************************************')
    print('[INFO] Created data loaders.')

    return dataloaders['train'], dataloaders['test']