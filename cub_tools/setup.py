from setuptools import setup

setup(name='cub_tools',
      version='2.2.0',
      description='Caltech UCSD Birds Database Tools for PyTorch Image Classification',
      url='https://github.com/ecm200/caltech_birds',
      author='Ed Morris',
      author_email='ecm200@gmail.com',
      license='MIT',
      packages=['cub_tools'],
      zip_safe=False,
      install_requires=[
            #'pytorch', # was set to 1.4, try latest
            #'torchvision',
            'imutils',
            'pandas',
            'matplotlib',
            'numpy',
            'torch-lucent',
            'pytorchcv',
            'scikit-image',
            'Pillow']
      )
