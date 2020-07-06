'''
Script to sort the images into train and test directories.

All other directory structures remain the same.

Train Test Split defined by the file **train_test_split.txt**

Ed Morris
(c) 2020
'''

import os
import pandas as pd
import shutil

# Script runtime options
root_dir = 'data'
data_dir = os.path.join(root_dir,'images')

image_fnames = pd.read_csv(filepath_or_buffer=os.path.join(root_dir,'images.txt'), 
                          header=None, 
                          delimiter=' ', 
                          names=['Img ID', 'file path'])

image_fnames['is training image?'] = pd.read_csv(filepath_or_buffer=os.path.join(root_dir,'train_test_split.txt'), 
                                                 header=None, delimiter=' ', 
                                                 names=['Img ID','is training image?'])['is training image?']

os.makedirs(os.path.join(data_dir,'train'), exist_ok=True)
os.makedirs(os.path.join(data_dir,'test'), exist_ok=True)

for i_image, image_fname in enumerate(image_fnames['file path']):
    if image_fnames['is training image?'].iloc[i_image]:
        new_dir = os.path.join(data_dir,'train',image_fname.split('/')[0])
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(src=os.path.join(data_dir,image_fname), dst=os.path.join(new_dir, image_fname.split('/')[1]))
        print(i_image, ':: Image is in training set. [', bool(image_fnames['is training image?'].iloc[i_image]),']')
        print('Image:: ', image_fname)
        print('Destination:: ', new_dir)
    else:
        new_dir = os.path.join(data_dir,'test',image_fname.split('/')[0])
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(src=os.path.join(data_dir,image_fname), dst=os.path.join(new_dir, image_fname.split('/')[1]))
        print(i_image, ':: Image is in testing set. [', bool(image_fnames['is training image?'].iloc[i_image]),']')
        print('Source Image:: ', image_fname)
        print('Destination:: ', new_dir)