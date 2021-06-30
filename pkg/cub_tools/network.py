import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

import numpy as np

from lucent.optvis import render, param, transform, objectives
from lucent.misc.channel_reducer import ChannelReducer
from lucent.misc.io import show


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def runDiversity(layerName, layerNeuron, imageSize=256, batch=4, weight=1e2):
    '''
    Function to run Lucent neuron diversity optimisation for a given Layer and Neuron (Channel) in a PyTorch CNN.

    '''
    batch_param_f = lambda: param.image(imageSize, batch=batch)
    obj = objectives.channel(layerName, layerNeuron) - weight * objectives.diversity(layerName)
    _ = render.render_vis(model_, obj, batch_param_f, show_inline=True)


def runDiversitywithTransforms(layerName, layerNeuron, transforms=None, imageSize=256, batch=4, weight=1e2):
    '''
    Function to run Lucent neuron diversity optimisation for a given Layer and Neuron (Channel) in a PyTorch CNN.
    This function uses image augmentation transforms to help improve the clarity and resolution of the produced neuron maximisations.

    '''
    if transforms == None:
        transforms = [
            transform.pad(16),
            transform.jitter(8),
            transform.random_scale([n/100. for n in range(80, 120)]),
            transform.random_rotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2))),
            transform.jitter(2),
        ]
    batch_param_f = lambda: param.image(imageSize, batch=batch)
    obj = objectives.channel(layerName, layerNeuron) - weight * objectives.diversity(layerName)
    _ = render.render_vis(model_, obj, batch_param_f, transforms=transforms, show_inline=True)


@torch.no_grad()
def get_layer(model, layer, X):
    hook = render.ModuleHook(getattr(model, layer))
    model(X)
    hook.close()
    return hook.features


@objectives.wrap_objective()
def dot_compare(layer, acts, batch=1):
    def inner(T):
        pred = T(layer)[batch]
        return -(pred * acts).sum(dim=0, keepdims=True).mean()

    return inner


def render_activation_grid(
    img,
    model,
    device,
    layer="mixed4d",
    cell_image_size=60,
    n_groups=6,
    n_steps=1024,
    batch_size=64,
    img_crop_size=224,
    img_resize=512,
    img_transforms=None):

    '''
    Activation Grid renderer from the Lucent library, modified for more general PyTorch models.
    This version has been adapted to allow image transforms from thr Torchvision.transforms module to be used.
    This allows bespoke image transformations, to be applied to the input image, to match those on which the network under investigation has been trained with.
    Currently, the default is resizing and cropping to the input size expected by the network, and standard scaled using the ImageNet coefficients.

    This work uses the Lucent library, which borrow heavily from the Google Brain Lucide library for Tensorflow.
    https://github.com/greentfrapp/lucent

    Inputs:

        Required inputs:

        img                 Pil image object.

        model               The torch.nn model object containing the CNN trained model.

        device              The device the computations are to be run. GPU STRONGLY ADVISED!.

        layer               The name of the layer that the spatial activations will be analysed from.

        Optional arguments:

        See Lucent descriptions execept for following.

        img_crop_size       The default transform image crop size, which should match the expected input size of the network under investigation.

        img_resize          The size the image should be resized to before cropping. Usually a good idea to keep it to the 2s-compliment size that's larger to the input size.
                            e.g. for an input size of 224, a resize of 256 would suffice. For 299, then 512 would be a good choice.

        img_transforms      An object of Torchvision.transforms modules, composed into a object using Torchvision.transforms.Compose(), as shown in the defaults.

        
    '''
    
    
    # First wee need, to normalize and resize the image.
    # Bespoke image transforms can be implemented and passed through the img_transforms argument.
    if img_transforms is None:
        img_transforms = transforms.Compose([
                            transforms.Resize(img_resize),
                            transforms.CenterCrop(img_crop_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    img = img_transforms(img).to(device)
    img = img.unsqueeze(0)
    # shape: (1, 3, 224, 224)
    #img = transforms_f(img)
    
    

    # Here we compute the activations of the layer `layer` using `img` as input
    # shape: (layer_channels, layer_height, layer_width), the shape depends on the layer
    acts = get_layer(model, layer, img)[0]
    # shape: (layer_height, layer_width, layer_channels)
    acts = acts.permute(1, 2, 0)
    # shape: (layer_height*layer_width, layer_channels)
    acts = acts.view(-1, acts.shape[-1])
    acts_np = acts.cpu().numpy()
    nb_cells = acts.shape[0]

    # negative matrix factorization `NMF` is used to reduce the number
    # of channels to n_groups. This will be used as the following.
    # Each cell image in the grid is decomposed into a sum of
    # (n_groups+1) images. First, each cell has its own set of parameters
    #  this is what is called `cells_params` (see below). At the same time, we have
    # a of group of images of size 'n_groups', which also have their own image parametrized
    # by `groups_params`. The resulting image for a given cell in the grid
    # is the sum of its own image (parametrized by `cells_params`)
    # plus a weighted sum of the images of the group. Each each image from the group
    # is weighted by `groups[cell_index, group_idx]`. Basically, this is a way of having
    # the possibility to make cells with similar activations have a similar image, because
    # cells with similar activations will have a similar weighting for the elements
    # of the group.
    if n_groups > 0:
        reducer = ChannelReducer(n_groups, "NMF")
        groups = reducer.fit_transform(acts_np)
        groups /= groups.max(0)
    else:
        groups = np.zeros([])
    # shape: (layer_height*layer_width, n_groups)
    groups = torch.from_numpy(groups)

    # Parametrization of the images of the groups (we have 'n_groups' groups)
    groups_params, groups_image_f = param.fft_image(
        [n_groups, 3, cell_image_size, cell_image_size]
    )
    # Parametrization of the images of each cell in the grid (we have 'layer_height*layer_width' cells)
    cells_params, cells_image_f = param.fft_image(
        [nb_cells, 3, cell_image_size, cell_image_size]
    )

    # First, we need to construct the images of the grid
    # from the parameterizations

    def image_f():
        groups_images = groups_image_f()
        cells_images = cells_image_f()
        X = []
        for i in range(nb_cells):
            x = 0.7 * cells_images[i] + 0.5 * sum(
                groups[i, j] * groups_images[j] for j in range(n_groups)
            )
            X.append(x)
        X = torch.stack(X)
        return X

    # make sure the images are between 0 and 1
    image_f = param.to_valid_rgb(image_f, decorrelate=True)

    # After constructing the cells images, we sample randomly a mini-batch of cells
    # from the grid. This is to prevent memory overflow, especially if the grid
    # is large.
    def sample(image_f, batch_size):
        def f():
            X = image_f()
            inds = torch.randint(0, len(X), size=(batch_size,))
            inputs = X[inds]
            # HACK to store indices of the mini-batch, because we need them
            # in objective func. Might be better ways to do that
            sample.inds = inds
            return inputs

        return f

    image_f_sampled = sample(image_f, batch_size=batch_size)

    # Now, we define the objective function

    def objective_func(model):
        # shape: (batch_size, layer_channels, cell_layer_height, cell_layer_width)
        pred = model(layer)
        # use the sampled indices from `sample` to get the corresponding targets
        target = acts[sample.inds].to(pred.device)
        # shape: (batch_size, layer_channels, 1, 1)
        target = target.view(target.shape[0], target.shape[1], 1, 1)
        dot = (pred * target).sum(dim=1).mean()
        return -dot

    obj = objectives.Objective(objective_func)

    def param_f():
        # We optimize the parametrizations of both the groups and the cells
        params = list(groups_params) + list(cells_params)
        return params, image_f_sampled

    results = render.render_vis(
        model,
        obj,
        param_f,
        thresholds=(n_steps,),
        show_image=False,
        progress=True,
        fixed_image_size=cell_image_size,
    )
    # shape: (layer_height*layer_width, 3, grid_image_size, grid_image_size)
    imgs = image_f()
    imgs = imgs.cpu().data
    imgs = imgs[:, :, 2:-2, 2:-2]
    # turn imgs into a a grid
    grid = torchvision.utils.make_grid(imgs, nrow=int(np.sqrt(nb_cells)), padding=0)
    grid = grid.permute(1, 2, 0)
    grid = grid.numpy()
    render.show(grid)
    return imgs