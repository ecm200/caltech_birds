from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

import skorch
from skorch import NeuralNetClassifier

from pathlib import Path
import os
import sys
import time
import copy

import pandas as pd
import matplotlib.pylab as plt

import numpy as np
from numpy import dstack

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


from cub_tools.utils import save_pickle

class stackedEnsemble():
    '''
    Class of Stacked Ensemble for creating an ensemble of PyTorch learners.

    Takes in a set of pre-trained CNNs for classification and a meta-learner and produces an ensemble prediction of class.

    The meta-learner default method is Logistic Regression, but a scikit-learn type classifier object can be passed to the class object on calling.
    Custom meta-learner object parameter values can also be passed using a dictionary of arguments.

    E.g.

    
    '''

    # TODO: 1. Add functionality to train the stacked models?
    
    def __init__(self, meta_learner=None, meta_learner_options=None, models=None, device=None):
        
        if meta_learner is None:
            self.meta_learner = LogisticRegression
        else:
            self.meta_learner = meta_learner
        self.meta_learner_options = meta_learner_options
        self.models = models
        self.device = device
        self.train_dataloader = None
        self.test_dataloader = None
        self.stackX = None
        self.stacky = None
        self.test_stackX = None
        self.yhat = None

        self.meta_learner_fit = False
        
    

    def fit(self, dataloader):
        '''
        Function to fit the meta-leaner by first generating the stacked dataset of probabilities.

        Assuming there N images in the set, and M models in the ensemble, the expected size of the array is as follows:

        nx (cols): number of classes by number of models e.g. 4 models and 200 classes, 800 columns. ny (rows): number of images

        The array is arrange as follows:

        image 1: |---200 class probability columns model 1---|---200 class probability columns model 2---|---***---|---200 class probability columns model M---|
        image 2: |---200 class probability columns model 1---|---200 class probability columns model 2---|---***---|---200 class probability columns model M---|
        ***
        ***
        image N: |---200 class probability columns model 1---|---200 class probability columns model 2---|---***---|---200 class probability columns model M---|
        '''

        assert self.models is not None,  'Fitting meta-learner with images requires the models of the ensemble to be provided so that class probabilities can be calculated. \n Reinitialise the class object as follows: stackedEnsemble(models=models_dict)'
        assert self.device is not None, 'Fitting meta-leaner with images requires specification of the PyTorch device to run the inference of the training images. \n Reinitialise the class object with as follows: stackedEnsemble(device=device)'

        self.train_dataloader = dataloader

        # generate stacked dataset from input images in dataloader
        print('[INFO] Creating the meta learner inputs (probabilities from individual models) as none provided.')
        self.stackX, self.stacky = stacked_dataset_from_dataloader(self.models, self.train_dataloader, self.device)

        # fit standalone model
        fit_meta_learner()
        self.meta_learner_fit = True
        print('..Complete')
    
    

    def fit_stacked(self, stackX, stacky):
        '''
        Function to fit the meta-leaner by using a pre-stacked dataset of probabilities from the ensemble of models.

        Assuming there N images in the set, and M models in the ensemble, the expected size of the array is as follows:

        nx (cols): number of classes by number of models e.g. 4 models and 200 classes, 800 columns. ny (rows): number of images

        The array is arrange as follows:

        image 1: |---200 class probability columns model 1---|---200 class probability columns model 2---|---***---|---200 class probability columns model M---|
        image 2: |---200 class probability columns model 1---|---200 class probability columns model 2---|---***---|---200 class probability columns model M---|
        ***
        ***
        image N: |---200 class probability columns model 1---|---200 class probability columns model 2---|---***---|---200 class probability columns model M---|
        '''

        print('[INFO] Stacked input table and labels found, using these to train meta learner.')
        self.stackX = stackX
        self.stacky = stacky
        
        # fit standalone model
        fit_meta_learner()
        self.meta_learner_fit = True
        print('..Complete')
        
    
    
    def predict(self, dataloader=None, stackX=None):
        '''
        Predict class of input images, first converting the input dataset to a stacked dataset.
        '''
        assert self.meta_learner_fit is True, 'Meta Leaner has not been fit. Please run the stackedEnsemble.fit() method to fit the meta learner before trying to predict'
        self.test_dataloader=dataloader

        # create dataset using ensemble
        print('[INFO] Creating the meta learner inputs (probabilities from individual models) as none provided.')
        self.test_stackX, _ = stacked_dataset_from_dataloader(self.models, self.test_dataloader, self.device)
        
        # predict using the trained meta learner
        print('[INFO] Predicting with the meta learner...', end='')
        self.yhat = self.meta_model.predict(X=self.test_stackX)
        
        print('..Complete')



    def predict_stacked(self, stackX):
        '''
        Predict class using a pre-stacked dataset as input.
        '''

        assert self.meta_learner_fit is True, 'Meta Leaner has not been fit. Please run the stackedEnsemble.fit() method to fit the meta learner before trying to predict'

        print('[INFO] Stacked input table and labels found, using these to train meta learner.')
        self.test_stackX=stackX

        # predict using the trained meta learner
        print('[INFO] Predicting with the meta learner...', end='')
        self.yhat = self.meta_model.predict(X=self.test_stackX)


    def save_stacked_dataset(self, dstack_type='train', fname_pre=None, fname_path=''):
        '''
        Function to save a stacked dataset to file.

        Possible to save either TRAIN or TEST datasets to file.
        Default is saving TRAIN dataset to current directory as train_stack.pkl

        dstack_type = Stacked dataset output can be either "train" or "test".

        '''

        assert (dstack_type == 'train') or (dstack_type == 'test'), 'dstack_type must be specified as either "train" or "test".'

        if dstack_type == 'train':
            assert (self.stackX is not None) or (self.stacky is not None), 'There is no training stacked dataset. Cannot save to file.'
            if fname_pre == None:
                fname_pre = 'train'
            outf_X = os.path.join(fname_path,fname_pre+'_stackX.pkl')
            outf_y = os.path.join(fname_path,fname_pre+'_stacky.pkl')
            save_pickle(self.stackX, outf_X)
            print('[IO] Saved stacked training set input data to: {}'.format(outf_X))
            save_pickle(self.stacky, outf_y)
            print('[IO] Saved stacked training set labels data to: {}'.format(outf_y))
        elif dstack_type == 'test':
            assert self.test_stackX is not None, 'There is no test stacked dataset. Cannot save to file.'
            if fname_pre == None:
                fname_pre = 'test'
            outf = os.path.join(fname_path,fname_pre+'_stackX.pkl')
            save_pickle(self.stackX, out_f)
            print('[IO] Saved stacked test set input data to: {}'.format(outf))

        
    def class_report(self, y_true):
        assert self.yhat is not None, 'No predictions found to compare to true classes. Run predict or predict_stacked method to generate predictions.'
        from sklearn.metrics import classification_report
        print(classification_report(y_pred=self.yhat, y_true=y_true))



    def fit_meta_learner(self):
        '''
        Fit the scikit learn classification object type meta learner with the stacked dataset and labels.
        '''

        print('[INFO] Training the meta learner...', end='')
        if self.meta_learner_options is not None:
            self.meta_model = self.meta_learner(**self.meta_learner_options)
        else:
            self.meta_model = self.meta_learner()
        
        self.meta_model.fit(self.stackX, self.stacky)
        
        

    def stacked_dataset_from_dataloader(self, models, dataloader, device):
        '''
        Stack the input dataset of images as output probabilbities from each model.

        Assuming there N images in the set, and M models in the ensemble, the expected size of the array is as follows:

        nx (cols): number of classes by number of models e.g. 4 models and 200 classes, 800 columns. ny (rows): number of images

        The array is arrange as follows:

        image 1: |---200 class probability columns model 1---|---200 class probability columns model 2---|---***---|---200 class probability columns model M---|
        image 2: |---200 class probability columns model 1---|---200 class probability columns model 2---|---***---|---200 class probability columns model M---|
        ***
        ***
        image N: |---200 class probability columns model 1---|---200 class probability columns model 2---|---***---|---200 class probability columns model M---|

        '''
        stackX = None
        stacky = None
        print('[INFO] Starting StackX', end='')
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                if i < (len(dataloader)-1):
                    temp_stack = None
                    for model_name, model in models.items():
                        # make prediction
                        if isinstance(model, skorch.classifier.NeuralNetClassifier):
                            yhat = model.predict_proba(inputs)
                        else:
                            model.eval()
                            inputs = inputs.to(device)
                            yhat = model(inputs)
                            yhat = yhat.cpu().numpy()

                        # Convert score to probability
                        for ind in np.arange(0, yhat.shape[0], 1):
                            yhat[ind, ::] = softmax(yhat[ind,::])

                        # stack predictions into [rows, members, probabilities]
                        if temp_stack is None:
                            temp_stack = yhat
                        else:
                            temp_stack = dstack((temp_stack, yhat))

                    # flatten predictions to [rows, members x probabilities]
                    temp_stack = temp_stack.reshape((temp_stack.shape[0], temp_stack.shape[1]*temp_stack.shape[2]))
                    # stack the batch of model probabilities onto the bottom of the results table
                    if stackX is None:
                        stackX = temp_stack
                    else:
                        stackX = np.vstack((stackX, temp_stack))

                    # stack the output truth labels to bottom of truth labels table
                    if stacky is None:
                        stacky = labels.cpu().numpy().ravel()
                    else:
                        stacky = np.vstack((stacky, labels.cpu().numpy().ravel()))

                    if i % 5 == 0:
                        print('..{}'.format(i), end='')

        print('..Complete')
        return stackX, stacky