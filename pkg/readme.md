# cub tools package installation

Cub_tools (in pkg/) has now been [published to PYPI](https://pypi.org/project/cub-tools/) and can be easily installed in any environment using following commands:

```shell
pip install cub-tools
```

The head version can still be installed by cloning this repository and running the following:

```shell
cd pkg/

pip install .
```

# Installing a python training environment

An environment file for installing the runtime environment has also been provided, in the root of the repository [conda dependencies](https://github.com/ecm200/caltech_birds/blob/master/conda_dependencies.yml).

**This should be run on a machine with the relevant version of the CUDA drivers installed, at current time of writing 11.1. To change the CUDA version, ensure that the cudatoolkit version is correct for the version you have installed, and also check the PyTorch version**. 

It is currently recommended that PyTorch be installed through Conda, as the PYPI versions were not working correctly (_at the time of writing, v1.8.1 when installed through PYPI was causing some computation errors due to linking errors with backend libraries. This was not happening when using Conda installed depdendencies_).

To install an environment using the conda dependencies file, run:

```shell
conda env create -f conda_depdencies.yml
```
