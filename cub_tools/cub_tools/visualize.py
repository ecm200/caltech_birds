from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

#from imutils import paths
from pathlib import Path
import os
import time
import copy

import pandas as pd
import matplotlib.pylab as plt
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import io
import base64
from skimage.util import img_as_uint, img_as_ubyte
from skimage.io import imsave


def imshow(inp, title=None, figsize=(20,8)):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=figsize)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, class_names, device, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                ax.set_title('actual:: '+class_names[labels[j]]+' \n predicted:: '+class_names[preds[j]])
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
def visualize_model_grid(model, class_names, device, dataloaders, num_images=6, figsize=(20,20), images_per_row=3):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=figsize)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                
                inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                
                ax = plt.subplot(num_images//images_per_row, images_per_row, images_so_far)
                ax.axis('off')
                plt.imshow(inp)
                
                if labels[j] == preds[j]:
                    predict_statement = '**CORRECT PREDICTION**'
                else:
                    predict_statement = '!!ERROR IN PREDICTION!!'
                
                plt.title('{} \n Actual: {} \n Predicted: {}'.format(predict_statement, class_names[labels[j]], class_names[preds[j]]))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def get_title_string_from_classes(classes, class_names):
    title_str = ''
    for i_x,x in enumerate(classes):
        if i_x !=0 and i_x % 4 == 0:
            title_str = title_str+' ||\n'
        title_str = title_str + ' || ' + class_names[x]
    return title_str


def encode_image(im):
    '''
    Function to encode an image to base64 representation.

    Inputs:

        im, PIL image   PIL image object loaded using PIL.Image.open(image).

    Returns:

        encoded image string in Base64 encoding.

    '''
    rawBytes = io.BytesIO()    
    im = Image.fromarray(img_as_ubyte(np.array(im)))
    im.save(rawBytes, "PNG")
    rawBytes.seek(0) 
    encoded_image = base64.b64encode(rawBytes.read()).decode()
    return f'data:image/png;base64,{encoded_image}'


def add_class_to_image(im, classname, textfill=(255,255,255,255), textsize=0.75, textpos=(10,10)):
    '''
    Add the class name to the image.
    '''

    txt = Image.new('RGBA', (int(im.size[0]*textsize),int(im.size[1]*textsize)), (255,255,255,0))
    d = ImageDraw.Draw(txt)
    font = ImageFont.load_default()
    d.text(textpos, classname, font=font, fill=textfill)
    return Image.alpha_composite(im.convert('RGBA'), txt.resize(im.size))